import contextlib
import gc
import json
import re
from pathlib import Path

import safetensors
import torch
import torch.nn as nn
try:
    from transformers import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4DecoderLayer,
        DeepseekV4RMSNorm,
    )
except (ImportError, KeyError):
    DeepseekV4Config = None
    DeepseekV4DecoderLayer = None
    DeepseekV4RMSNorm = None

_FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# FP8 weight dtypes that need block-wise scale application
_FP8_DTYPES = set()
if hasattr(torch, "float8_e4m3fn"):
    _FP8_DTYPES.add(torch.float8_e4m3fn)
if hasattr(torch, "float8_e5m2"):
    _FP8_DTYPES.add(torch.float8_e5m2)


def dequantize_fp4_weight(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if quantized.dim() < 2 or scales.dim() < 2:
        raise ValueError(
            f"Quantized tensor dims ({quantized.dim()}) and scales dims ({scales.dim()}) must be >= 2"
        )
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=quantized.device)
    u8 = quantized.contiguous().view(torch.uint8)
    low = (u8 & 0xF).long()
    high = ((u8 >> 4) & 0xF).long()
    unpacked = torch.stack([lut[low], lut[high]], dim=-1)
    quantized_fp32 = unpacked.reshape(*quantized.shape[:-1], 2 * quantized.shape[-1])

    rows, cols = quantized_fp32.shape[-2:]
    scale_rows, scale_cols = scales.shape[-2:]
    block_m = rows // scale_rows
    block_n = cols // scale_cols

    if cols % scale_cols != 0:
        raise ValueError(
            f"Quantized tensor columns ({cols}) must be divisible by "
            f"scale block size ({scale_cols})"
        )
    if scale_rows * block_m != rows:
        raise ValueError(
            f"Shape mismatch: {rows} rows with {scale_rows} scale rows "
            f"and {block_m} rows per block"
        )

    s_fp32 = scales.to(torch.float32)
    q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = s_fp32.reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    result = (q * s).to(torch.bfloat16)
    return result.reshape(*quantized.shape[:-2], rows, cols)


def dequantize_fp8_weight(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight with E8M0 block scales.
    
    weight: float8_e4m3fn, shape [R, C]
    scales: float8_e8m0fnu, shape [R//block_m, C//block_n]
    """
    w_bf16 = weight.to(torch.bfloat16)
    s_f32 = scales.to(torch.float32)

    rows, cols = w_bf16.shape[-2:]
    scale_rows, scale_cols = s_f32.shape[-2:]
    block_m = rows // scale_rows
    block_n = cols // scale_cols

    q = w_bf16.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = s_f32.reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    result = (q * s)
    return result.reshape(*weight.shape[:-2], rows, cols).to(torch.bfloat16)


class V4BlockDiskLoader:
    EXPERT_W_MAP = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

    # Map disk prefix → model state_dict prefix for layer tensors
    _RENAME_MAP = {
        "attn.wq_a.": "self_attn.q_a_proj.",
        "attn.wq_b.": "self_attn.q_b_proj.",
        "attn.wkv.": "self_attn.kv_proj.",
        "attn.wo_a.": "self_attn.o_a_proj.",
        "attn.wo_b.": "self_attn.o_b_proj.",
        "attn.q_norm.": "self_attn.q_a_norm.",
        "attn.kv_norm.": "self_attn.kv_norm.",
        "attn.attn_sink": "self_attn.sinks",
        "attn_norm.": "input_layernorm.",
        "ffn_norm.": "post_attention_layernorm.",
"hc_attn_fn": "self_attn.attn_hc.fn",
         "hc_attn_base": "self_attn.attn_hc.base",
         "hc_attn_scale": "self_attn.attn_hc.scale",
         "hc_ffn_fn": "ffn_hc.fn",
         "hc_ffn_base": "ffn_hc.base",
         "hc_ffn_scale": "ffn_hc.scale",
    }

    # Compressor/indexer renames (CSA layers, nested path)
    _RENAME_COMPRESSOR_MAP = {
        "attn.compressor.ape": "self_attn.compressor.position_bias",
        "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
        "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
        "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
        "attn.indexer.compressor.ape": "self_attn.compressor.indexer.position_bias",
        "attn.indexer.compressor.norm.weight": "self_attn.compressor.indexer.kv_norm.weight",
        "attn.indexer.compressor.wgate.weight": "self_attn.compressor.indexer.gate_proj.weight",
        "attn.indexer.compressor.wkv.weight": "self_attn.compressor.indexer.kv_proj.weight",
        "attn.indexer.weights_proj.": "self_attn.compressor.indexer.weights_proj.",
        "attn.indexer.wq_b.": "self_attn.compressor.indexer.q_b_proj.",
    }

    @staticmethod
    def _apply_rename_map(key):
        for old, new in V4BlockDiskLoader._RENAME_COMPRESSOR_MAP.items():
            if key.startswith(old):
                return key.replace(old, new, 1)
        for old, new in V4BlockDiskLoader._RENAME_MAP.items():
            if key.startswith(old):
                return key.replace(old, new, 1)
        if key.startswith("ffn."):
            return "mlp." + key[4:]
        return key

    @staticmethod
    def _classify_tensors(tensor_names):
        per_expert = {}
        stacked = {}
        shared = {}
        gate = {}
        hc = {}
        compressor = {}
        fp8_pairs = {}
        fallthrough = []

        for name in tensor_names:
            stripped = re.sub(r"^layers\.\d+\.", "", name)

            # per-expert: ffn.experts.N.w{1,2,3}.{weight|scale}
            m = re.match(r"ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)", stripped)
            if m:
                idx = int(m.group(1))
                w_type = m.group(2)
                suffix = m.group(3)
                if idx not in per_expert:
                    per_expert[idx] = {}
                if w_type not in per_expert[idx]:
                    per_expert[idx][w_type] = {}
                per_expert[idx][w_type][suffix] = name
                continue

            # stacked expert: ffn.experts.w{1,2,3}.{weight|scale}
            m = re.match(r"ffn\.experts\.(w[123])\.(weight|scale)", stripped)
            if m:
                w_type = m.group(1)
                suffix = m.group(2)
                if w_type not in stacked:
                    stacked[w_type] = {}
                stacked[w_type][suffix] = name
                continue

            # shared expert: ffn.shared_experts.w{1,2,3}.{weight|scale}
            m = re.match(r"ffn\.shared_experts\.(w[123])\.(weight|scale)", stripped)
            if m:
                w_type = m.group(1)
                suffix = m.group(2)
                if w_type not in shared:
                    shared[w_type] = {}
                shared[w_type][suffix] = name
                continue

            # gate: ffn.gate.{weight|bias|tid2eid}
            m = re.match(r"ffn\.gate\.(weight|bias|tid2eid)", stripped)
            if m:
                gate[m.group(1)] = name
                continue

            # compressor/indexer (CSA layers)
            if stripped.startswith("attn.compressor") or stripped.startswith("attn.indexer"):
                compressor[stripped] = name
                continue

            # HC parameters
            if stripped.startswith("hc_attn") or stripped.startswith("hc_ffn"):
                hc[stripped] = name
                continue

            # FP8 attention weight/scale pairs
            m = re.match(r"attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.(weight|scale)", stripped)
            if m:
                base = m.group(1)
                suffix = m.group(2)
                if base not in fp8_pairs:
                    fp8_pairs[base] = {}
                fp8_pairs[base][suffix] = name
                continue

            fallthrough.append(name)

        return per_expert, stacked, shared, gate, hc, compressor, fp8_pairs, fallthrough

    def __init__(self, model_path, config=None):
        self.model_path = self._resolve_path(model_path)
        self._shard_cache = {}

        if DeepseekV4Config is None:
            raise ImportError(
                "DeepSeek V4 support requires transformers >= 5.9.0. "
                "Install with: pip install transformers>=5.9.0"
            )

        index_path = self.model_path / "model.safetensors.index.json"
        with open(index_path) as f:
            self.index = json.load(f)

        self.layer_map = self._build_layer_tensor_map()

        if config is None:
            config = DeepseekV4Config.from_pretrained(str(self.model_path))
        self.config = config

    @staticmethod
    def _resolve_path(path):
        p = Path(path)
        if p.exists():
            return p.resolve()

        try:
            from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
            cache_name = f"models--{str(path).replace('/', '--')}"
            cache_dir = Path(HUGGINGFACE_HUB_CACHE) / cache_name / "snapshots"
            if cache_dir.exists():
                snapshots = sorted(cache_dir.iterdir())
                if snapshots:
                    return snapshots[-1]
        except (ImportError, Exception):
            pass

        raise FileNotFoundError(
            f"Model path '{path}' not found locally. "
            f"Download with: huggingface-cli download {path}"
        )

    def _build_layer_tensor_map(self):
        layer_map = {}
        for tensor_name in self.index["weight_map"]:
            m = re.match(r"layers\.(\d+)\.", tensor_name)
            if m:
                layer_idx = int(m.group(1))
                layer_map.setdefault(layer_idx, []).append(tensor_name)
        return layer_map

    def _load_tensor(self, tensor_name):
        shard_file = self.index["weight_map"][tensor_name]
        if shard_file not in self._shard_cache:
            self._shard_cache[shard_file] = safetensors.safe_open(
                str(self.model_path / shard_file),
                framework="pt",
                device="cpu",
            )
        return self._shard_cache[shard_file].get_tensor(tensor_name)

    def load_non_backbone_modules(self, model=None):
        embed_weight = self._load_tensor("embed.weight")
        embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        norm_weight = self._load_tensor("norm.weight")
        norm = DeepseekV4RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        norm.weight.data = norm_weight.to(norm.weight.dtype)
        lm_weight = self._load_tensor("head.weight")
        lm = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        lm.weight.data = lm_weight.to(lm.weight.dtype)
        result = {
            "embed_tokens": embed,
            "norm": norm,
            "lm_head": lm,
        }
        if model is not None:
            model.model.embed_tokens = embed
            model.model.norm = norm
            model.lm_head = lm

        # Load HC head parameters if present on disk
        hc_head_keys = {
            "hc_head_fn": "hc_head.hc_fn",
            "hc_head_base": "hc_head.hc_base",
            "hc_head_scale": "hc_head.hc_scale",
        }
        hc_head_state = {}
        for disk_key, model_key in hc_head_keys.items():
            if disk_key in self.index["weight_map"]:
                tensor = self._load_tensor(disk_key)
                hc_head_state[model_key] = tensor
        if hc_head_state and model is not None:
            model.model.hc_head.load_state_dict(hc_head_state, strict=False, assign=True)
        if hc_head_state:
            result["hc_head"] = hc_head_state

        # Materialize rotary_emb buffers (inv_freq is computed in __init__,
        # not stored in safetensors). When loaded on meta, these are meta
        # tensors and need to be recreated on CPU.
        if model is not None:
            self._materialize_rope(
                model.model.rotary_emb,
                theta=self.config.rope_theta,
            )

        return result

    @staticmethod
    def _materialize_rope(rope: nn.Module, theta: float = 10000.0) -> None:
        main_dim = rope.main_inv_freq.shape[-1] * 2
        compress_dim = rope.compress_inv_freq.shape[-1] * 2
        device = "cpu"

        def _make_inv_freq(dim: int) -> torch.Tensor:
            return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

        main_inv = _make_inv_freq(main_dim)
        compress_inv = _make_inv_freq(compress_dim)

        for name in ("main_inv_freq", "main_original_inv_freq"):
            rope._buffers[name] = main_inv
        for name in ("compress_inv_freq", "compress_original_inv_freq"):
            rope._buffers[name] = compress_inv

    def _dequant_weight(self, weight_name, scale_name):
        """Dequantize a weight tensor, auto-detecting FP4 vs FP8 format from dtype."""
        tensor = self._load_tensor(weight_name)
        if scale_name is None:
            return self._to_bf16(tensor)
        scales = self._load_tensor(scale_name)
        if tensor.dtype == torch.int8:
            return dequantize_fp4_weight(tensor, scales)
        if tensor.dtype in _FP8_DTYPES:
            return dequantize_fp8_weight(tensor, scales)
        return self._to_bf16(tensor)

    def _dequant_experts_into_lists(self, per_expert_tensors):
        sorted_idx = sorted(per_expert_tensors.keys())
        gate_parts = []
        up_parts = []
        down_parts = []

        for idx in sorted_idx:
            data = per_expert_tensors[idx]
            for w_type in ["w1", "w2", "w3"]:
                if w_type not in data:
                    continue
                weight_name = data[w_type].get("weight")
                scale_name = data[w_type].get("scale")
                if weight_name is None or scale_name is None:
                    continue
                result = self._dequant_weight(weight_name, scale_name)
                if w_type == "w1":
                    gate_parts.append(result)
                elif w_type == "w3":
                    up_parts.append(result)
                elif w_type == "w2":
                    down_parts.append(result)
        return gate_parts, up_parts, down_parts

    @staticmethod
    def _stack_expert_lists(gate_parts, up_parts, down_parts):
        if gate_parts and up_parts:
            gate = torch.stack(gate_parts, dim=0)
            up = torch.stack(up_parts, dim=0)
            del gate_parts, up_parts
            return {"mlp.experts.gate_up_proj": torch.cat([gate, up], dim=1)}
        if down_parts:
            return {"mlp.experts.down_proj": torch.stack(down_parts, dim=0)}
        return {}

    def _process_stacked_expert_tensors(self, stacked_expert_tensors, state_dict):
        gate_part = None
        up_part = None
        down_part = None
        for w_type in ["w1", "w2", "w3"]:
            if w_type not in stacked_expert_tensors:
                continue
            weight_name = stacked_expert_tensors[w_type].get("weight")
            scale_name = stacked_expert_tensors[w_type].get("scale")
            if weight_name is None or scale_name is None:
                continue
            result = self._dequant_weight(weight_name, scale_name)
            if w_type == "w1":
                gate_part = result
            elif w_type == "w3":
                up_part = result
            elif w_type == "w2":
                down_part = result

        if gate_part is not None and up_part is not None:
            state_dict["mlp.experts.gate_up_proj"] = torch.cat([gate_part, up_part], dim=-2)
        if down_part is not None:
            state_dict["mlp.experts.down_proj"] = down_part

    def _to_bf16(self, tensor):
        if tensor.dtype != torch.bfloat16:
            return tensor.to(torch.bfloat16)
        return tensor

    def _process_shared_experts(self, shared_tensors, state_dict):
        for w_type in ["w1", "w2", "w3"]:
            if w_type not in shared_tensors:
                continue
            weight_name = shared_tensors[w_type].get("weight")
            scale_name = shared_tensors[w_type].get("scale")
            if weight_name is None:
                continue
            result = self._dequant_weight(weight_name, scale_name)
            proj_name = self.EXPERT_W_MAP.get(w_type, w_type)
            state_dict["mlp.shared_experts." + proj_name + ".weight"] = result

    def _process_gate_tensors(self, gate_tensors, state_dict):
        if "weight" in gate_tensors:
            tensor = self._load_tensor(gate_tensors["weight"])
            state_dict["mlp.gate.weight"] = self._to_bf16(tensor)
        if "bias" in gate_tensors:
            tensor = self._load_tensor(gate_tensors["bias"])
            state_dict["mlp.gate.e_score_correction_bias"] = tensor
        if "tid2eid" in gate_tensors:
            tensor = self._load_tensor(gate_tensors["tid2eid"])
            state_dict["mlp.gate.tid2eid"] = tensor

    def _process_hc_tensors(self, hc_tensors, state_dict):
        for stripped, tensor_name in hc_tensors.items():
            tensor = self._load_tensor(tensor_name)
            model_key = self._apply_rename_map(stripped)
            state_dict[model_key] = tensor

    def _process_compressor_tensors(self, compressor_tensors, state_dict):
        for stripped, tensor_name in compressor_tensors.items():
            tensor = self._load_tensor(tensor_name)
            model_key = self._apply_rename_map(stripped)
            state_dict[model_key] = tensor

    def _process_fp8_pairs(self, fp8_pairs, state_dict):
        for base_name, pair in fp8_pairs.items():
            weight_name = pair.get("weight")
            scale_name = pair.get("scale")
            if weight_name is None:
                continue
            tensor = self._dequant_weight(weight_name, scale_name)
            stripped = re.sub(r"^layers\.\d+\.", "", weight_name)
            stripped = re.sub(r"\.(weight|scale)$", "", stripped)
            model_key = self._apply_rename_map(stripped + ".weight")
            state_dict[model_key] = tensor

    def _process_fallthrough_tensors(self, fallthrough_tensors, state_dict):
        for name in fallthrough_tensors:
            stripped = re.sub(r"^layers\.\d+\.", "", name)
            model_key = self._apply_rename_map(stripped)
            tensor = self._load_tensor(name)
            state_dict[model_key] = self._to_bf16(tensor)

    def _build_layer_state_dict(self, layer_idx):
        tensor_names = self.layer_map.get(layer_idx, [])
        state_dict = {}

        per_expert, stacked, shared, gate, hc, compressor, fp8_pairs, fallthrough = self._classify_tensors(tensor_names)

        # Process non-expert tensors (small, no memory issue)
        if stacked:
            self._process_stacked_expert_tensors(stacked, state_dict)
        if shared:
            self._process_shared_experts(shared, state_dict)
        if gate:
            self._process_gate_tensors(gate, state_dict)
        if hc:
            self._process_hc_tensors(hc, state_dict)
        if compressor:
            self._process_compressor_tensors(compressor, state_dict)
        if fp8_pairs:
            self._process_fp8_pairs(fp8_pairs, state_dict)
        if fallthrough:
            self._process_fallthrough_tensors(fallthrough, state_dict)

        # Process experts in batches to avoid OOM from shard mmap + accumulated lists
        if per_expert:
            sorted_idx = sorted(per_expert.keys())
            batch_size = 32
            gate_parts_all, up_parts_all, down_parts_all = [], [], []
            for batch_start in range(0, len(sorted_idx), batch_size):
                batch_keys = sorted_idx[batch_start:batch_start + batch_size]
                batch = {k: per_expert[k] for k in batch_keys}
                gp, up, dp = self._dequant_experts_into_lists(batch)
                # Close shard to free mmap before stacking this batch
                self.close()
                gc.collect()
                gate_stacked = None
                up_stacked = None
                down_stacked = None
                if gp and up:
                    gate_stacked = torch.stack(gp, dim=0)
                    up_stacked = torch.stack(up, dim=0)
                    gate_parts_all.append(torch.cat([gate_stacked, up_stacked], dim=1))
                if dp:
                    down_stacked = torch.stack(dp, dim=0)
                    down_parts_all.append(down_stacked)
                del gp, up, dp, gate_stacked, up_stacked, down_stacked
                gc.collect()

            if gate_parts_all:
                state_dict["mlp.experts.gate_up_proj"] = torch.cat(gate_parts_all, dim=0)
            if down_parts_all:
                state_dict["mlp.experts.down_proj"] = torch.cat(down_parts_all, dim=0)
        gc.collect()
        return state_dict

    def load_layer(self, layer_idx, device="cuda"):
        state_dict = self._build_layer_state_dict(layer_idx)

        with torch.device("meta"):
            layer = DeepseekV4DecoderLayer(self.config, layer_idx)

        layer.load_state_dict(state_dict, strict=False, assign=True)
        layer.to(device)
        layer.eval()
        return layer

    def load_into_block(self, block, layer_idx, device="cpu"):
        """Load real BF16 weights from disk into an existing meta block's parameters."""
        state_dict = self._build_layer_state_dict(layer_idx)
        
        # Materialize meta tensors by moving to target device
        # Must do this AFTER building state_dict but BEFORE load_state_dict
        block.to_empty(device=device)
        
        block.load_state_dict(state_dict, strict=False, assign=True)
        return block

    def unload_layer(self, layer, clear_shard_cache=False):
        layer.to("cpu")
        del layer
        gc.collect()
        if clear_shard_cache:
            self.close()

    def close(self):
        for handle in self._shard_cache.values():
            with contextlib.suppress(Exception):
                handle.close()
        self._shard_cache.clear()
        gc.collect()
