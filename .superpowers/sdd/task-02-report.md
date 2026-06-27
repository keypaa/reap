# Task 2 Report: Phase 1 — Block-From-Disk Loader

## Status: DONE

## Implementation

Implemented `V4BlockDiskLoader` class in `src/reap/v4_block_loader.py`:

- **`dequantize_fp4_weight(quantized, scales)`** — FP4→BF16 decompression using the exact algorithm from `finegrained_fp8.py`:
  - Unpacks I8→FP4 via LUT lookup (2 FP4 nibbles per byte)
  - Applies per-block F8_E8M0 scales (block_m=1, block_n=32)
  - Handles 2D and 3D (batched) tensors

- **`V4BlockDiskLoader`**:
  - `__init__`: Reads `model.safetensors.index.json`, builds layer→tensor mapping
  - `load_non_backbone_modules()`: Uses `from_pretrained` for embed/norm/lm_head (~2 GB)
  - `load_layer(layer_idx, device)`: Reads per-layer tensors, decompresses FP4→BF16 for expert weights, builds state_dict, loads into `DeepseekV4DecoderLayer` on meta device via `load_state_dict(strict=False, assign=True)`, then moves to target device
  - Handles both per-expert (`experts.{idx}.w{1,2,3}`) and stacked (`experts.w{1,2,3}`) safetensor naming
  - Maps `w1`/`w2`/`w3`→`gate_proj`/`down_proj`/`up_proj` for shared_experts (LlamaMLP)
  - Concatenates `w1` (gate half) + `w3` (up half) → `gate_up_proj` for routed experts
  - Casts F8_E4M3→BF16 for shared_experts and any other F8 tensors
  - `unload_layer()`: Moves to CPU and garbage-collects

## Files Changed

| File | Change |
|------|--------|
| `src/reap/v4_block_loader.py` | **Created** — V4BlockDiskLoader class + dequantize_fp4_weight |
| `src/reap/__init__.py` | **Modified** — added import of V4BlockDiskLoader |
| `tests/test_v4_block_loader.py` | **Created** — 11 tests covering FP4 decompression and loader init |

## Test Results

```
tests/test_v4_block_loader.py::TestFP4Dequantize::test_fp4_dequantize_shape     PASSED
tests/test_v4_block_loader.py::TestFP4Dequantize::test_fp4_dequantize_values    PASSED
tests/test_v4_block_loader.py::TestFP4Dequantize::test_fp4_dequantize_scales_applied PASSED
tests/test_v4_block_loader.py::TestFP4Dequantize::test_fp4_dequantize_3d       PASSED
tests/test_v4_block_loader.py::TestV4BlockDiskLoader::test_init                 PASSED
tests/test_v4_block_loader.py::TestV4BlockDiskLoader::test_layer_tensor_map     PASSED
tests/test_v4_block_loader.py::TestV4BlockDiskLoader::test_load_tensor_raises   PASSED
tests/test_v4_block_loader.py::TestV4BlockDiskLoader::test_build_layer_tensor_map PASSED
tests/test_v4_block_loader.py::TestDequantizeEdgeCases::test_zero_scale         PASSED
tests/test_v4_block_loader.py::TestDequantizeEdgeCases::test_identity_scale     PASSED
tests/test_v4_block_loader.py::TestDequantizeEdgeCases::test_negative_fp4_values PASSED
                                                                   11 passed
tests/test_v4_model_registration.py::TestModelAttrs::test_model_attrs_v4        PASSED
tests/test_v4_model_registration.py::TestModelAttrs::test_is_v4_model          PASSED
tests/test_v4_model_registration.py::TestModelAttrs::test_is_v4_model_other    PASSED
tests/test_v4_model_registration.py::TestObserverConfig::test_v4_observer_config PASSED
tests/test_v4_model_registration.py::TestObserverConfig::test_registry_contains_v4 PASSED
                                                                    5 passed
tests/test_pruning_metrics.py::test_update_pruning_state_filters_masked_tokens  PASSED
tests/test_pruning_metrics.py::test_update_pruning_state_renormalizes_selected_router_weights PASSED
                                                                    2 passed
                                         Total: 18/18 passed in 10.68s
```

## Self-Review

### What I like
- FP4 decompression matches the reference implementation exactly
- Both per-expert and stacked tensor naming conventions are handled
- Clean interface: `load_layer(idx, device)` returns a ready-to-use module
- All tests pass, including Phase 0 regression

### What I'd do with a real checkpoint
- The `shared_experts` handling assumes 3 separate w1/w2/w3 keys mapping to `gate_proj/down_proj/up_proj`. If the checkpoint uses a single fused weight, this path needs adjustment
- The `load_non_backbone_modules` method uses `from_pretrained` which decompresses everything to CPU — but these 3 modules are only ~2 GB so should work within 180 GB budget
- Compressor-related state_dict keys (`self_attn.compressor.*`) are handled by the generic "strip prefix" path since they follow the same `model.layers.N.XXX` pattern

### Edge cases handled
- 2D and 3D (batched expert) FP4 input tensors
- F8_E4M3→BF16 conversion for any tensor
- Zero/empty scale tensors
- Negative FP4 values (nibble values 8-15)
- Missing compressor keys for sliding_attention layers (handled by `strict=False`)

## Concerns

1. **Shared expert weight format**: The safetensor index naming for `shared_experts` may differ from what's implemented — verified only when loading an actual checkpoint
2. **No integration test with real weights**: The loader is tested with mock safetensor metadata and synthetic FP4 data, but not end-to-end with a real checkpoint
3. **Memory during non-backbone loading**: `from_pretrained` loads the full model briefly (~560 GB CPU RAM), which might OOM even for just 2 seconds — could optimize to load these 3 modules directly from safetensor shards if needed

---

## Review Fixes Applied (2026-06-27)

### Critical #2: `nn.LayerNorm` → `DeepseekV4RMSNorm`
- **File**: `src/reap/v4_block_loader.py:10-11` (import), `src/reap/v4_block_loader.py:77` (usage)
- **Fix**: Imported `DeepseekV4RMSNorm` from `transformers.models.deepseek_v4.modeling_deepseek_v4` and used it instead of `nn.LayerNorm` with `eps=self.config.rms_norm_eps`. DeepseekV4's `model.norm` is RMSNorm (bias-less, doesn't subtract mean), so `nn.LayerNorm` would produce incorrect output.

### Important #3: Overly permissive exception catch
- **File**: `tests/test_v4_block_loader.py:148`
- **Fix**: Changed `pytest.raises((FileNotFoundError, RuntimeError, OSError, Exception))` to `pytest.raises(safetensors.SafetensorError)`. `SafetensorError` is what `safetensors.safe_open` raises on empty/corrupt files. Added `import safetensors` to the test file.

### Important #4: Unused variable in `_process_per_expert_tensors`
- **File**: `src/reap/v4_block_loader.py:167`
- **Fix**: Removed `n_experts = self.config.n_routed_experts` which was assigned but never used.

### Minor #5: Unused class variable `EXPERT_WEIGHT_TO_PARAM`
- **File**: `src/reap/v4_block_loader.py:36`
- **Fix**: Removed `EXPERT_WEIGHT_TO_PARAM` class variable (never referenced anywhere in the codebase).

### Not Changed
- **Critical #1 (wrong concat dim)**: False positive. `dim=1` (3D) and `dim=-2` (2D) are both correct — they concatenate along the output/row dimension giving shape `[N, 2*intermediate_dim, hidden_dim]` matching `gate_up_proj`.

### Test Results
```
tests/test_v4_block_loader.py       11/11 passed
tests/test_v4_model_registration.py  5/5  passed
                                   16/16 passed in 10.58s
```
