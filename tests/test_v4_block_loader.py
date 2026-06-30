import json
import re
import tempfile
from pathlib import Path

import safetensors
import pytest
import torch

from reap.v4_block_loader import V4BlockDiskLoader, dequantize_fp4_weight

try:
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    HAS_HF = True
except ImportError:
    HAS_HF = False


class TestFP4Dequantize:
    def test_fp4_dequantize_shape(self):
        packed = torch.zeros(2, 16, dtype=torch.int8)
        packed[0, :2] = torch.tensor([0x01, 0x23], dtype=torch.int8)
        packed[1, :2] = torch.tensor([0x45, 0x67], dtype=torch.int8)
        scales = torch.ones(1, 16, dtype=torch.float32)
        scales[0, :2] = torch.tensor([2.0, 4.0])
        result = dequantize_fp4_weight(packed, scales)
        assert result.shape == (2, 32)
        assert result.dtype == torch.bfloat16

    def test_fp4_dequantize_values(self):
        lut = (
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        )
        packed = torch.zeros(2, 16, dtype=torch.int8)
        packed[0, 0] = 0x01
        packed[1, 0] = 0x23
        scales = torch.ones(1, 16, dtype=torch.float32)
        result = dequantize_fp4_weight(packed, scales)

        row0_low = lut[0x01 & 0xF]
        row0_high = lut[(0x01 >> 4) & 0xF]
        row1_low = lut[0x23 & 0xF]
        row1_high = lut[(0x23 >> 4) & 0xF]
        expected = torch.zeros(2, 32, dtype=torch.bfloat16)
        expected[0, :2] = torch.tensor([row0_low, row0_high])
        expected[1, :2] = torch.tensor([row1_low, row1_high])
        assert torch.allclose(result, expected)

    def test_fp4_dequantize_scales_applied(self):
        packed = torch.zeros(1, 32, dtype=torch.int8)
        packed[0, 0] = 0x01
        packed[0, 1] = 0x23
        scales = torch.ones(1, 32, dtype=torch.float32)
        scales[0, 0] = 2.0
        scales[0, 1] = 3.0
        result = dequantize_fp4_weight(packed, scales)
        lut = (
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        )
        expected = torch.zeros(1, 64, dtype=torch.bfloat16)
        expected[0, 0] = lut[0x01 & 0xF] * 2.0
        expected[0, 1] = lut[(0x01 >> 4) & 0xF] * 2.0
        expected[0, 2] = lut[0x23 & 0xF] * 3.0
        expected[0, 3] = lut[(0x23 >> 4) & 0xF] * 3.0
        assert torch.allclose(result, expected)

    def test_fp4_dequantize_3d(self):
        packed = torch.zeros(1, 2, 16, dtype=torch.int8)
        packed[0, 0, :2] = torch.tensor([0x01, 0x23], dtype=torch.int8)
        packed[0, 1, :2] = torch.tensor([0x45, 0x67], dtype=torch.int8)
        scales = torch.ones(1, 1, 16, dtype=torch.float32)
        scales[0, 0, :2] = torch.tensor([2.0, 4.0])
        result = dequantize_fp4_weight(packed, scales)
        assert result.shape == (1, 2, 32)
        assert result.dtype == torch.bfloat16

    def test_fp4_dequantize_e8m0_scales(self):
        """E8M0 scale format: float8_e8m0fnu.to(float32) gives decoded value."""
        packed = torch.zeros(1, 16, dtype=torch.int8)
        packed[0, 0] = 0x01
        arr = bytearray([130])
        scales = torch.frombuffer(arr, dtype=torch.uint8).view(torch.float8_e8m0fnu).reshape(1, 1)
        result = dequantize_fp4_weight(packed, scales)
        assert result.shape == (1, 32)
        # lut[1] = 0.5, scale for byte 130 = 2**(130-127) = 8.0 → 0.5 * 8.0 = 4.0
        assert abs(result[0, 0].item() - 4.0) < 1e-6


class TestRenameMap:
    def test_attention_projection_rename(self):
        assert V4BlockDiskLoader._apply_rename_map("attn.wq_a.weight") == "self_attn.q_a_proj.weight"
        assert V4BlockDiskLoader._apply_rename_map("attn.wq_b.scale") == "self_attn.q_b_proj.scale"
        assert V4BlockDiskLoader._apply_rename_map("attn.wkv.weight") == "self_attn.kv_proj.weight"
        assert V4BlockDiskLoader._apply_rename_map("attn.wo_a.weight") == "self_attn.o_a_proj.weight"
        assert V4BlockDiskLoader._apply_rename_map("attn.wo_b.weight") == "self_attn.o_b_proj.weight"

    def test_attention_norm_rename(self):
        assert V4BlockDiskLoader._apply_rename_map("attn.q_norm.weight") == "self_attn.q_a_norm.weight"
        assert V4BlockDiskLoader._apply_rename_map("attn.kv_norm.weight") == "self_attn.kv_norm.weight"
        assert V4BlockDiskLoader._apply_rename_map("attn.attn_sink") == "self_attn.sinks"

    def test_layer_norm_rename(self):
        assert V4BlockDiskLoader._apply_rename_map("attn_norm.weight") == "input_layernorm.weight"
        assert V4BlockDiskLoader._apply_rename_map("ffn_norm.weight") == "post_attention_layernorm.weight"

    def test_hc_rename(self):
        assert V4BlockDiskLoader._apply_rename_map("hc_attn_fn") == "self_attn.attn_hc.fn"
        assert V4BlockDiskLoader._apply_rename_map("hc_attn_base") == "self_attn.attn_hc.base"
        assert V4BlockDiskLoader._apply_rename_map("hc_attn_scale") == "self_attn.attn_hc.scale"
        assert V4BlockDiskLoader._apply_rename_map("hc_ffn_fn") == "ffn_hc.fn"
        assert V4BlockDiskLoader._apply_rename_map("hc_ffn_base") == "ffn_hc.base"
        assert V4BlockDiskLoader._apply_rename_map("hc_ffn_scale") == "ffn_hc.scale"

    def test_ffn_to_mlp_rename(self):
        assert V4BlockDiskLoader._apply_rename_map("ffn.gate.weight") == "mlp.gate.weight"
        assert V4BlockDiskLoader._apply_rename_map("ffn.gate.bias") == "mlp.gate.bias"

    def test_compressor_rename(self):
        assert V4BlockDiskLoader._apply_rename_map("attn.compressor.ape") == "self_attn.compressor.position_bias"
        assert V4BlockDiskLoader._apply_rename_map("attn.compressor.norm.weight") == "self_attn.compressor.kv_norm.weight"
        assert V4BlockDiskLoader._apply_rename_map("attn.compressor.wgate.weight") == "self_attn.compressor.gate_proj.weight"
        assert V4BlockDiskLoader._apply_rename_map("attn.compressor.wkv.weight") == "self_attn.compressor.kv_proj.weight"

    def test_indexer_rename(self):
        assert V4BlockDiskLoader._apply_rename_map("attn.indexer.compressor.ape") == "self_attn.compressor.indexer.position_bias"
        assert V4BlockDiskLoader._apply_rename_map("attn.indexer.wq_b.weight") == "self_attn.compressor.indexer.q_b_proj.weight"

    def test_fallthrough_no_match(self):
        assert V4BlockDiskLoader._apply_rename_map("some_unknown.key") == "some_unknown.key"


class TestClassifyTensors:
    def _classify(self, names):
        return V4BlockDiskLoader._classify_tensors(names)

    def test_classify_per_expert(self):
        names = [
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w1.scale",
            "layers.0.ffn.experts.0.w2.weight",
            "layers.0.ffn.experts.1.w1.weight",
        ]
        per_expert, stacked, shared, gate, hc, compressor, fp8, fallthrough = self._classify(names)
        assert 0 in per_expert
        assert 1 in per_expert
        assert "w1" in per_expert[0]
        assert "w2" in per_expert[0]
        assert len(fallthrough) == 0

    def test_classify_shared(self):
        names = [
            "layers.0.ffn.shared_experts.w1.weight",
            "layers.0.ffn.shared_experts.w1.scale",
        ]
        per_expert, stacked, shared, gate, hc, compressor, fp8, fallthrough = self._classify(names)
        assert "w1" in shared
        assert len(per_expert) == 0

    def test_classify_gate(self):
        names = [
            "layers.0.ffn.gate.weight",
            "layers.0.ffn.gate.bias",
        ]
        per_expert, stacked, shared, gate, hc, compressor, fp8, fallthrough = self._classify(names)
        assert "weight" in gate
        assert "bias" in gate

    def test_classify_hc(self):
        names = [
            "layers.0.hc_attn_fn",
            "layers.0.hc_attn_base",
            "layers.0.hc_ffn_fn",
        ]
        per_expert, stacked, shared, gate, hc, compressor, fp8, fallthrough = self._classify(names)
        assert "hc_attn_fn" in hc
        assert "hc_attn_base" in hc
        assert "hc_ffn_fn" in hc

    def test_classify_fallthrough(self):
        names = [
            "layers.0.attn_norm.weight",
            "layers.0.ffn_norm.weight",
        ]
        per_expert, stacked, shared, gate, hc, compressor, fp8, fallthrough = self._classify(names)
        assert len(fallthrough) == 2

    def test_classify_compressor(self):
        names = [
            "layers.10.attn.compressor.ape",
            "layers.10.attn.compressor.norm.weight",
            "layers.10.attn.indexer.compressor.wgate.weight",
        ]
        per_expert, stacked, shared, gate, hc, compressor, fp8, fallthrough = self._classify(names)
        assert len(compressor) == 3

    def test_classify_tid2eid(self):
        names = ["layers.0.ffn.gate.tid2eid"]
        per_expert, stacked, shared, gate, hc, compressor, fp8, fallthrough = self._classify(names)
        assert "tid2eid" in gate

    def test_classify_fp8_pairs(self):
        names = [
            "layers.0.attn.wq_a.weight",
            "layers.0.attn.wq_a.scale",
            "layers.0.attn.wkv.weight",
            "layers.0.attn.wkv.scale",
        ]
        per_expert, stacked, shared, gate, hc, compressor, fp8, fallthrough = self._classify(names)
        assert "wq_a" in fp8
        assert "wkv" in fp8
        assert len(fallthrough) == 0


class TestV4BlockDiskLoader:
    SAMPLE_WEIGHT_MAP = {
        "layers.0.ffn.experts.0.w1.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.0.w1.scale": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.0.w2.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.0.w2.scale": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.0.w3.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.0.w3.scale": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.1.w1.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.1.w1.scale": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.1.w2.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.1.w2.scale": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.1.w3.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.experts.1.w3.scale": "model-00001-of-00006.safetensors",
        "layers.0.ffn.gate.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.gate.bias": "model-00001-of-00006.safetensors",
        "layers.0.ffn.shared_experts.w1.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.shared_experts.w1.scale": "model-00001-of-00006.safetensors",
        "layers.0.ffn.shared_experts.w2.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.shared_experts.w2.scale": "model-00001-of-00006.safetensors",
        "layers.0.ffn.shared_experts.w3.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn.shared_experts.w3.scale": "model-00001-of-00006.safetensors",
        "layers.0.attn.wq_a.weight": "model-00001-of-00006.safetensors",
        "layers.0.attn.wq_a.scale": "model-00001-of-00006.safetensors",
        "layers.0.attn.wq_b.weight": "model-00001-of-00006.safetensors",
        "layers.0.attn.wq_b.scale": "model-00001-of-00006.safetensors",
        "layers.0.attn.wkv.weight": "model-00001-of-00006.safetensors",
        "layers.0.attn.wkv.scale": "model-00001-of-00006.safetensors",
        "layers.0.attn.wo_a.weight": "model-00001-of-00006.safetensors",
        "layers.0.attn.wo_a.scale": "model-00001-of-00006.safetensors",
        "layers.0.attn.wo_b.weight": "model-00001-of-00006.safetensors",
        "layers.0.attn.wo_b.scale": "model-00001-of-00006.safetensors",
        "layers.0.attn.q_norm.weight": "model-00001-of-00006.safetensors",
        "layers.0.attn.kv_norm.weight": "model-00001-of-00006.safetensors",
        "layers.0.attn.attn_sink": "model-00001-of-00006.safetensors",
        "layers.0.attn_norm.weight": "model-00001-of-00006.safetensors",
        "layers.0.ffn_norm.weight": "model-00001-of-00006.safetensors",
        "layers.0.hc_attn_fn": "model-00001-of-00006.safetensors",
        "layers.0.hc_attn_base": "model-00001-of-00006.safetensors",
        "layers.0.hc_attn_scale": "model-00001-of-00006.safetensors",
        "layers.0.hc_ffn_fn": "model-00001-of-00006.safetensors",
        "layers.0.hc_ffn_base": "model-00001-of-00006.safetensors",
        "layers.0.hc_ffn_scale": "model-00001-of-00006.safetensors",
        "layers.42.ffn.experts.0.w1.weight": "model-00006-of-00006.safetensors",
        "layers.42.ffn.gate.weight": "model-00006-of-00006.safetensors",
        "layers.42.attn.wq_a.weight": "model-00006-of-00006.safetensors",
        "layers.42.attn_norm.weight": "model-00006-of-00006.safetensors",
        "embed.weight": "model-00001-of-00006.safetensors",
        "norm.weight": "model-00006-of-00006.safetensors",
        "head.weight": "model-00006-of-00006.safetensors",
    }

    @pytest.fixture
    def mock_model_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            index = {
                "metadata": {"total_size": 1000000},
                "weight_map": self.SAMPLE_WEIGHT_MAP,
            }
            index_file = tmpdir_path / "model.safetensors.index.json"
            with open(index_file, "w") as f:
                json.dump(index, f)

            for shard in set(self.SAMPLE_WEIGHT_MAP.values()):
                shard_path = tmpdir_path / shard
                shard_path.write_bytes(b"")

            yield tmpdir_path

    def test_init(self, mock_model_dir):
        loader = V4BlockDiskLoader(mock_model_dir)
        assert loader.model_path == mock_model_dir
        assert "weight_map" in loader.index
        assert set(loader.layer_map.keys()) == {0, 42}

    def test_layer_tensor_map(self, mock_model_dir):
        loader = V4BlockDiskLoader(mock_model_dir)
        layer_0_tensors = loader.layer_map[0]
        assert len(layer_0_tensors) == 41
        assert all(t.startswith("layers.0.") for t in layer_0_tensors)

        layer_42_tensors = loader.layer_map[42]
        assert len(layer_42_tensors) == 4
        assert all(t.startswith("layers.42.") for t in layer_42_tensors)

    def test_load_tensor_raises_on_missing_shard(self, mock_model_dir):
        loader = V4BlockDiskLoader(mock_model_dir)
        with pytest.raises(safetensors.SafetensorError):
            loader._load_tensor("layers.0.ffn.gate.weight")

    def test_resolve_path_local_dir(self, mock_model_dir):
        resolved = V4BlockDiskLoader._resolve_path(mock_model_dir)
        assert resolved == mock_model_dir.resolve()
        assert resolved.is_absolute()

    def test_resolve_path_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found locally"):
            V4BlockDiskLoader._resolve_path("nonexistent/model-id")

    def test_resolve_path_hf_cache(self, tmp_path, monkeypatch):
        if not HAS_HF:
            pytest.skip("huggingface_hub not installed")
        cache_root = tmp_path / "hf_cache"
        snapshot_dir = cache_root / "models--deepseek-ai--DeepSeek-V4-Flash" / "snapshots" / "abc123def"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model.safetensors.index.json").write_text("{}")

        monkeypatch.setattr("huggingface_hub.constants.HUGGINGFACE_HUB_CACHE", str(cache_root))
        resolved = V4BlockDiskLoader._resolve_path("deepseek-ai/DeepSeek-V4-Flash")
        assert resolved == snapshot_dir

    def test_build_layer_tensor_map_ignores_non_layer_tensors(self, mock_model_dir):
        loader = V4BlockDiskLoader(mock_model_dir)
        for layer_idx in loader.layer_map:
            for tensor_name in loader.layer_map[layer_idx]:
                assert re.match(r"layers\.\d+\.", tensor_name)
        assert "embed.weight" not in loader.layer_map
        assert "norm.weight" not in loader.layer_map
        assert "head.weight" not in loader.layer_map

    def test_classify_smoke(self, mock_model_dir):
        loader = V4BlockDiskLoader(mock_model_dir)
        names = loader.layer_map[0]
        per_expert, stacked, shared, gate, hc, compressor, fp8, fallthrough = loader._classify_tensors(names)
        assert len(per_expert) == 2
        assert len(shared) == 3
        assert "weight" in gate
        assert "bias" in gate
        assert len(hc) == 6
        assert len(fp8) == 5
        assert len(fallthrough) >= 4


class TestDequantizeEdgeCases:
    def test_zero_scale(self):
        packed = torch.zeros(1, 16, dtype=torch.int8)
        packed[0, 0] = 0x01
        packed[0, 1] = 0x23
        scales = torch.zeros(1, 16, dtype=torch.float32)
        result = dequantize_fp4_weight(packed, scales)
        assert torch.all(result == 0.0)

    def test_identity_scale(self):
        packed = torch.zeros(1, 16, dtype=torch.int8)
        packed[0, 0] = 0x01
        scales = torch.ones(1, 16, dtype=torch.float32)
        result = dequantize_fp4_weight(packed, scales)
        lut = (
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        )
        expected = torch.zeros(1, 32, dtype=torch.bfloat16)
        expected[0, :2] = torch.tensor([lut[0x01 & 0xF], lut[(0x01 >> 4) & 0xF]])
        assert torch.allclose(result, expected)

    def test_negative_fp4_values(self):
        packed = torch.zeros(1, 16, dtype=torch.uint8)
        packed[0, 0] = 0x89
        packed = packed.to(torch.int8)
        scales = torch.ones(1, 16, dtype=torch.float32)
        result = dequantize_fp4_weight(packed, scales)
        lut = (
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        )
        expected = torch.zeros(1, 32, dtype=torch.bfloat16)
        expected[0, :2] = torch.tensor([lut[0x89 & 0xF], lut[(0x89 >> 4) & 0xF]])
        assert torch.allclose(result, expected)

    def test_fp4_dequantize_non_divisible_raises(self):
        packed = torch.tensor([[0x01, 0x23]], dtype=torch.int8)
        scales = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="must be divisible by"):
            dequantize_fp4_weight(packed, scales)
