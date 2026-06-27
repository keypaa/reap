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


class TestV4BlockDiskLoader:
    SAMPLE_WEIGHT_MAP = {
        "model.layers.0.mlp.experts.0.w1.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.0.w1.scale": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.0.w2.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.0.w2.scale": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.0.w3.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.0.w3.scale": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.1.w1.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.1.w1.scale": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.1.w2.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.1.w2.scale": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.1.w3.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.experts.1.w3.scale": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.gate.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.shared_experts.w1.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.shared_experts.w2.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.mlp.shared_experts.w3.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.self_attn.q_a_proj.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.self_attn.q_a_norm.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.self_attn.q_b_proj.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.self_attn.kv_proj.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.self_attn.kv_norm.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.self_attn.o_a_proj.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.self_attn.o_b_proj.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.self_attn.sinks": "model-00001-of-00006.safetensors",
        "model.layers.0.input_layernorm.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00006.safetensors",
        "model.layers.0.attn_hc.fn": "model-00001-of-00006.safetensors",
        "model.layers.0.attn_hc.base": "model-00001-of-00006.safetensors",
        "model.layers.0.attn_hc.scale": "model-00001-of-00006.safetensors",
        "model.layers.0.ffn_hc.fn": "model-00001-of-00006.safetensors",
        "model.layers.0.ffn_hc.base": "model-00001-of-00006.safetensors",
        "model.layers.0.ffn_hc.scale": "model-00001-of-00006.safetensors",
        "model.layers.42.mlp.experts.0.w1.weight": "model-00006-of-00006.safetensors",
        "model.layers.42.mlp.gate.weight": "model-00006-of-00006.safetensors",
        "model.layers.42.self_attn.q_a_proj.weight": "model-00006-of-00006.safetensors",
        "model.layers.42.input_layernorm.weight": "model-00006-of-00006.safetensors",
        "embed_tokens.weight": "model-00001-of-00006.safetensors",
        "model.norm.weight": "model-00006-of-00006.safetensors",
        "lm_head.weight": "model-00006-of-00006.safetensors",
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
        assert len(layer_0_tensors) == 32
        assert all(t.startswith("model.layers.0.") for t in layer_0_tensors)

        layer_42_tensors = loader.layer_map[42]
        assert len(layer_42_tensors) == 4
        assert all(t.startswith("model.layers.42.") for t in layer_42_tensors)

    def test_load_tensor_raises_on_missing_shard(self, mock_model_dir):
        loader = V4BlockDiskLoader(mock_model_dir)
        with pytest.raises(safetensors.SafetensorError):
            loader._load_tensor("model.layers.0.mlp.gate.weight")

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
                assert re.match(r"model\.layers\.\d+\.", tensor_name)
        assert "embed_tokens.weight" not in loader.layer_map
        assert "model.norm.weight" not in loader.layer_map
        assert "lm_head.weight" not in loader.layer_map


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
        # packed (1,2) → dequant (1,4), scales (1,3): 4 % 3 = 1 → error
        packed = torch.tensor([[0x01, 0x23]], dtype=torch.int8)
        scales = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="must be divisible by"):
            dequantize_fp4_weight(packed, scales)
