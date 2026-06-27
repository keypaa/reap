import gc
import json
import re
from pathlib import Path

import safetensors
import torch
import torch.nn as nn
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4RMSNorm,
)

_FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequantize_fp4_weight(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
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

    q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = scales.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    result = (q * s).to(torch.bfloat16)
    return result.reshape(*quantized.shape[:-2], rows, cols)


class V4BlockDiskLoader:
    EXPERT_W_MAP = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    def __init__(self, model_path, config=None):
        self.model_path = Path(model_path)
        self._shard_cache = {}

        index_path = self.model_path / "model.safetensors.index.json"
        with open(index_path) as f:
            self.index = json.load(f)

        self.layer_map = self._build_layer_tensor_map()

        if config is None:
            config = DeepseekV4Config.from_pretrained(str(self.model_path))
        self.config = config

    def _build_layer_tensor_map(self):
        layer_map = {}
        for tensor_name in self.index["weight_map"]:
            m = re.match(r"model\.layers\.(\d+)\.", tensor_name)
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
        embed_weight = self._load_tensor("model.embed_tokens.weight")
        embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        norm_weight = self._load_tensor("model.norm.weight")
        norm = DeepseekV4RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        norm.weight.data = norm_weight.to(norm.weight.dtype)
        lm_weight = self._load_tensor("lm_head.weight")
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
        return result

    def load_layer(self, layer_idx, device="cuda"):
        tensor_names = self.layer_map.get(layer_idx, [])
        state_dict = {}

        per_expert_tensors = {}
        stacked_expert_tensors = {}
        shared_tensors = {}

        for name in tensor_names:
            m = re.match(
                r"model\.layers\.\d+\.mlp\.experts\.(\d+)\.(w[123])\.(weight|scale)", name
            )
            if m:
                expert_idx = int(m.group(1))
                w_type = m.group(2)
                suffix = m.group(3)
                if expert_idx not in per_expert_tensors:
                    per_expert_tensors[expert_idx] = {}
                if w_type not in per_expert_tensors[expert_idx]:
                    per_expert_tensors[expert_idx][w_type] = {}
                per_expert_tensors[expert_idx][w_type][suffix] = name
                continue

            m = re.match(
                r"model\.layers\.\d+\.mlp\.experts\.(w[123])\.(weight|scale)", name
            )
            if m:
                w_type = m.group(1)
                suffix = m.group(2)
                if w_type not in stacked_expert_tensors:
                    stacked_expert_tensors[w_type] = {}
                stacked_expert_tensors[w_type][suffix] = name
                continue

            m = re.match(r"model\.layers\.\d+\.mlp\.shared_experts\.(.+)", name)
            if m:
                shared_key = m.group(1)
                shared_tensors[shared_key] = name
                continue

            stripped = re.sub(r"^model\.layers\.\d+\.", "", name)
            tensor = self._load_tensor(name)
            state_dict[stripped] = self._to_bf16(tensor)

        # Process per-expert tensors
        if per_expert_tensors:
            self._process_per_expert_tensors(per_expert_tensors, state_dict)

        # Process stacked expert tensors
        if stacked_expert_tensors:
            self._process_stacked_expert_tensors(stacked_expert_tensors, state_dict)

        # Process shared expert tensors
        for shared_key, tensor_name in shared_tensors.items():
            tensor = self._load_tensor(tensor_name)
            tensor = self._to_bf16(tensor)
            state_key = None
            for w_name, proj_name in self.EXPERT_W_MAP.items():
                if shared_key.startswith(w_name + "."):
                    state_key = shared_key.replace(w_name, proj_name, 1)
                    break
            if state_key is None:
                state_key = shared_key
            state_dict["mlp.shared_experts." + state_key] = tensor

        with torch.device("meta"):
            layer = DeepseekV4DecoderLayer(self.config, layer_idx)

        layer.load_state_dict(state_dict, strict=False, assign=True)
        layer.to(device)
        layer.eval()
        return layer

    def _to_bf16(self, tensor):
        if hasattr(torch, "float8_e4m3fn") and tensor.dtype == torch.float8_e4m3fn:
            return tensor.to(torch.bfloat16)
        return tensor

    def _process_per_expert_tensors(self, per_expert_tensors, state_dict):
        gate_parts = []
        up_parts = []
        down_parts = []

        sorted_idx = sorted(per_expert_tensors.keys())
        for idx in sorted_idx:
            data = per_expert_tensors[idx]
            for w_type in ["w1", "w2", "w3"]:
                if w_type not in data:
                    continue
                weight_name = data[w_type].get("weight")
                scale_name = data[w_type].get("scale")
                if weight_name is None or scale_name is None:
                    continue
                quant = self._load_tensor(weight_name)
                scales = self._load_tensor(scale_name)
                result = dequantize_fp4_weight(quant, scales)
                if w_type == "w1":
                    gate_parts.append(result)
                elif w_type == "w3":
                    up_parts.append(result)
                elif w_type == "w2":
                    down_parts.append(result)

        if gate_parts and up_parts:
            gate = torch.stack(gate_parts, dim=0)
            up = torch.stack(up_parts, dim=0)
            state_dict["mlp.experts.gate_up_proj"] = torch.cat([gate, up], dim=1)
        if down_parts:
            state_dict["mlp.experts.down_proj"] = torch.stack(down_parts, dim=0)

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
            quant = self._load_tensor(weight_name)
            scales = self._load_tensor(scale_name)
            result = dequantize_fp4_weight(quant, scales)
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

    def unload_layer(self, layer):
        layer.to("cpu")
        del layer
        gc.collect()

    def close(self):
        self._shard_cache.clear()
        gc.collect()
