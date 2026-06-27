# Task 4: Phase 3 — V4-Specific Pruning

**Status:** DONE

## Commits

```
092410e Task 4: Phase 3 — V4-specific pruning with hash router remap
```

## Files Created
- `src/reap/v4_prune_utils.py` — `_prune_v4_layer()` and `_remap_hash_router_tid2eid()`
- `tests/test_v4_prune.py` — 14 tests in 7 test classes

## Files Modified
- `src/reap/prune.py` — V4 interception branch + V4 config update

## Test Results

All 48 tests pass across all 4 V4 test files:

```
tests/test_v4_prune.py ..............                         14 passed
tests/test_v4_moe_observer.py ...................             19 passed
tests/test_v4_block_loader.py ...........                     11 passed
tests/test_v4_model_registration.py ....                      4 passed
```

## Test Coverage (test_v4_prune.py)

| Test | What it verifies |
|------|-----------------|
| `test_prune_v4_layer_shapes` | 3D param shapes after pruning 8→4 |
| `test_prune_v4_layer_no_retained` | 4→1 edge case |
| `test_prune_v4_layer_keeps_correct_weights` | Retained weights match originals |
| `test_remap_all_retained` | tid2eid unchanged when all retained |
| `test_remap_half_retained` | tid2eid remapped + pruned experts → 0 |
| `test_remap_with_unused_tids` | -1 entries preserved in tid2eid |
| `test_remap_tid2eid_buffer_preserved` | `data.copy_()` keeps buffer registration |
| `test_bias_pruned_with_topk_router` | e_score_correction_bias indexed correctly |
| `test_bias_guard_no_crash_without_bias` | Hash router (no bias) doesn't error |
| `test_bias_guard_no_crash_without_device` | Generic gate without bias doesn't error |
| `test_config_fields_updated` | Config n_routed/num_local not touched by prune layer |
| `test_shared_experts_left_untouched` | Weights unchanged after prune |
| `test_shared_experts_modules_preserved` | Module structure intact |
| `test_hash_router_prune_roundtrip` | Full prune flow with hash router |

## Concerns

None. The implementation closely follows the brief spec. Notable design decisions:

1. **`original_num_experts` saved before mutation** — the hash router's `old_to_new` mapping needs the original count, which would be lost after updating `moe.experts.num_experts`
2. **`.data.copy_()` for buffer preservation** — `_remap_hash_router_tid2eid` updates `tid2eid.data.copy_()` instead of replacing the buffer registration, keeping the buffer's identity intact
3. **`hasattr` guards** — `e_score_correction_bias` (TopKRouter only), `num_experts` (may vary by transformers version), `tid2eid` (HashRouter only) all use `hasattr` for safe access
