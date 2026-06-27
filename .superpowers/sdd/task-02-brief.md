# Task 2: Phase 1 — Block-From-Disk Loader

## Context
Task 1 (Phase 0) registered `DeepseekV4ForCausalLM` in MODEL_ATTRS and OBSERVER_CONFIG_REGISTRY. Now we build the core block-from-disk loader that reads one decoder layer at a time from safetensor shards, decompresses FP4→BF16, and constructs a `DeepseekV4DecoderLayer` on GPU.

**Key constraint:** Lightning AI RTX PRO 6000 has 180 GB CPU RAM. Full `from_pretrained(device_map="cpu")` decompresses all 43 layers to BF16 → ~560 GB — doesn't fit. We must load and decompress one layer at a time.

## Files to Create
- `src/reap/v4_block_loader.py` — `V4BlockDiskLoader` class

## Files to Edit
- `src/reap/__init__.py` — add `from .v4_block_loader import V4BlockDiskLoader`

## Implementation Requirements

### The FP4→BF16 Format

Validated evidence confirms the exact storage format:

| Weight tensor | Safetensor dtype | Shape | Scale tensor | Scale shape |
|--------------|-------------------|-------|--------------|-------------|
| `experts.N.w1.weight` (gate_up) | I8 (packed FP4) | [2048, 2048] | `.w1.scale` | [2048, 128] |
| `experts.N.w2.weight` (down) | I8 (packed FP4) | [4096, 1024] | `.w2.scale` | [4096, 64] |
| `experts.N.w3.weight` (up) | I8 (packed FP4) | [2048, 2048] | `.w3.scale` | [2048, 128] |
| `shared_experts.w1.weight` | F8_E4M3 | [2048, 4096] | (none needed) | — |
| `gate.weight` (router) | BF16 | [256, 4096] | (none needed) | — |

FP4 format details:
- **FP4 E2M1**: 4-bit floating point: 1 sign bit, 2 exponent bits, 1 mantissa bit
- 16 possible values from the LUT: `(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)`
- **Packing**: Each I8 byte packs 2 FP4 values: low nibble (bits 0-3) = first FP4, high nibble (bits 4-7) = second FP4
- **Scales**: F8_E8M0 format (8-bit exponent, 0-bit mantissa = pure power of 2)
- **Block structure**: Block size = 32 columns (block_m=1, block_n=32). Each column group of 32 FP4 values shares one F8_E8M0 scale factor
- After I8→FP4 unpack, the column dimension doubles (e.g., [2048, 2048] → [2048, 4096] BF16)

### Decompression Algorithm (from transformers `finegrained_fp8.py`)

```python
_FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

def _dequantize_fp4_weight(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decompress a single FP4-quantized weight tensor to BF16.
    
    Args:
        quantized: I8 tensor of packed FP4 values, shape [out_dim, in_dim]
        scales: F8_E8M0 scale tensor, shape [out_dim, in_dim // 32]
    Returns:
        Decompressed BF16 tensor, shape [out_dim, 2 * in_dim]
    """
    # Step 1: Unpack I8 → FP4 via LUT lookup
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=quantized.device)
    u8 = quantized.contiguous().view(torch.uint8)
    low = (u8 & 0xF).long()
    high = ((u8 >> 4) & 0xF).long()
    unpacked = torch.stack([lut[low], lut[high]], dim=-1)
    quantized_fp32 = unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])
    
    # Step 2: Apply per-block F8_E8M0 scales
    rows, cols = quantized_fp32.shape[-2:]
    scale_rows, scale_cols = scales.shape[-2:]
    block_m = rows // scale_rows  # 1
    block_n = cols // scale_cols  # 32
    
    q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = scales.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    result = (q * s).to(torch.bfloat16)
    return result.reshape(rows, cols)
```

### `V4BlockDiskLoader` Class

```python
class V4BlockDiskLoader:
    """Load one DeepseekV4DecoderLayer at a time from disk.
    
    Strategy:
    1. Read model.safetensors.index.json to map tensor names → shard files
    2. Load non-backbone modules (embed, norm, lm_head) once — ~2 GB
    3. For each decoder layer: read I8+F8 scale tensors, decompress to BF16,
       construct DeepseekV4DecoderLayer on GPU, forward, free.
    """
    
    def __init__(self, model_path: str, config: DeepseekV4Config):
        # Load safetensors index
        # Cache shard file handles (lazy open)
        # Build mapping: layer_idx → list of tensor names in that layer
    
    def load_non_backbone_modules(self, device: str = "cpu") -> dict:
        """Load embed_tokens, norm, lm_head — loaded once, kept in CPU memory."""
    
    def load_layer(self, layer_idx: int, device: str) -> nn.Module:
        """Load one decoder layer from disk to GPU.
        
        1. Read all I8 weight tensors + F8 scale tensors for this layer from shard
        2. Decompress FP4→BF16 for expert weight tensors
        3. Build state_dict for the layer (BF16 weights)
        4. Create empty DeepseekV4DecoderLayer(config) on meta device
        5. load_state_dict(layer_state_dict, assign=True)
        6. Move to GPU
        7. Set eval mode
        """
    
    def unload_layer(self, layer: nn.Module):
        """Free layer from GPU memory."""
```

Key details:
- **Tensor naming pattern**: `model.layers.N.mlp.experts.gate_up_proj`, `model.layers.N.mlp.experts.down_proj`, `model.layers.N.mlp.gate.weight`, `model.layers.N.self_attn.*`, etc.
- **Scale tensor naming**: Expert weights have companion `.w1.scale`, `.w2.scale`, `.w3.scale` tensors (NOT `.gate_up_proj.scale`)
- **Alternate naming for safetensors**: The safetensor index uses `experts.w1.weight` not `experts.gate_up_proj`. Map accordingly: (w1=gate_up_proj, w2=down_proj, w3=up_proj)
- **Shared experts**: Stored as `model.layers.N.mlp.shared_experts.w1.weight` in F8_E4M3 (no decompression needed, just cast to BF16)
- **Router**: `model.layers.N.mlp.gate.weight` in BF16 — no conversion
- **Layer construction**: Use `DeepseekV4DecoderLayer(config)` on meta device (`torch.device("meta")`), then `load_state_dict(strict=False, assign=True)`
- **Non-backbone modules**: Load via `from_pretrained(model_path, device_map="cpu", torch_dtype=torch.bfloat16)` and extract `model.model.embed_tokens`, `model.model.norm`, `model.lm_head`
- **Safetensor index file**: `model.safetensors.index.json` contains `{"metadata": {...}, "weight_map": {"tensor_name": "shard_file.safetensors"}}`

### Pseudo-code for `_build_layer_tensor_map()`

```python
def _build_layer_tensor_map(self):
    """Map layer_idx → list of tensor names belonging to that layer."""
    layer_map = {}
    for tensor_name in self.index["weight_map"]:
        m = re.match(r"model\.layers\.(\d+)\.", tensor_name)
        if m:
            layer_idx = int(m.group(1))
            layer_map.setdefault(layer_idx, []).append(tensor_name)
    return layer_map
```

### Pseudo-code for `_load_tensor_from_shard()`

```python
def _load_tensor_from_shard(self, tensor_name: str) -> torch.Tensor:
    shard_file = self.index["weight_map"][tensor_name]
    if shard_file not in self._shard_cache:
        self._shard_cache[shard_file] = safetensors.safe_open(
            f"{self.model_path}/{shard_file}",
            framework="pt",
            device="cpu",
        )
    return self._shard_cache[shard_file].get_tensor(tensor_name)
```

### Layer tensor name map (safetensor index → state_dict)

The mapping from safetensor index key to DeepseekV4DecoderLayer state_dict key:

| Index key | state_dict key | Dtype after decompress |
|-----------|---------------|----------------------|
| `model.layers.N.mlp.experts.w1.weight` | `mlp.experts.gate_up_proj` | BF16 |
| `model.layers.N.mlp.experts.w1.scale` | (skip — consumed by decompress) | — |
| `model.layers.N.mlp.experts.w2.weight` | `mlp.experts.down_proj` | BF16 |
| `model.layers.N.mlp.experts.w2.scale` | (skip — consumed by decompress) | — |
| `model.layers.N.mlp.experts.w3.weight` | `mlp.experts.up_proj` | BF16 |
| `model.layers.N.mlp.experts.w3.scale` | (skip — consumed by decompress) | — |
| `model.layers.N.mlp.shared_experts.w1.weight` | `mlp.shared_experts.linear.weight` | BF16 |
| `model.layers.N.mlp.gate.weight` | `mlp.gate.weight` | BF16 |
| `model.layers.N.mlp.gate.e_score_correction_bias` | `mlp.gate.e_score_correction_bias` | F32 |
| `model.layers.N.self_attn.*` | `self_attn.*` | BF16 |
| `model.layers.N.input_layernorm.*` | `input_layernorm.*` | BF16 |
| `model.layers.N.post_attention_layernorm.*` | `post_attention_layernorm.*` | BF16 |
| `model.layers.N.pre_moe_layernorm.*` | `pre_moe_layernorm.*` | BF16 |
| `model.layers.N.post_moe_layernorm.*` | `post_moe_layernorm.*` | BF16 |
| `model.layers.N.mhc_comb.*` | `mhc_comb.*` | BF16 |

**IMPORTANT**: The safetensor index uses `experts.w1`, `w2`, `w3` naming convention, but the model's actual `state_dict` expects `experts.gate_up_proj`, `experts.down_proj`, `experts.up_proj`. You must map these between reading from safetensor and loading into the model.

## Non-backbone Module Loading

Instead of loading these from individual safetensor tensors, use `from_pretrained` to extract them:

```python
def load_non_backbone_modules(self):
    """Load embed_tokens, norm, lm_head using from_pretrained."""
    full_model = AutoModelForCausalLM.from_pretrained(
        self.model_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    non_backbone = {
        "embed_tokens": full_model.model.embed_tokens,
        "norm": full_model.model.norm,
        "lm_head": full_model.lm_head,
    }
    del full_model
    gc.collect()
    return non_backbone
```

These three modules consume ~2 GB total and stay on CPU.

## Integration Test

Write `tests/test_v4_block_loader.py`:
- `test_fp4_dequantize()` — Unit test the FP4→BF16 decompression:
  1. Create a small I8 tensor with known FP4 nibble values (e.g., `torch.tensor([[0x01, 0x23], [0x45, 0x67]], dtype=torch.int8)`)
  2. Create the matching scale tensor (e.g., `torch.tensor([[1.0, 1.0]], dtype=torch.float32)`)
  3. Decompress and verify the output values match the LUT at expected positions
  4. Verify output shape: I8 [M, N] → BF16 [M, 2*N]
- `test_v4_block_loader_init()` — Test initialization with a mock config
- `test_layer_tensor_map()` — Parse a mock `model.safetensors.index.json` and verify layer mapping

## Pre-existing test to verify against

Run `C:\Users\pauma\miniconda3\envs\py310\python.exe -m pytest tests/test_v4_model_registration.py -v` to confirm Phase 0 changes are not broken.

## Report

Write results to `.superpowers/sdd/task-02-report.md` with: status, commits, test results, and any concerns.
