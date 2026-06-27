# Task 3 Report: Phase 2 — DeepseekV4MoEObserver with Incremental Expert Loop

**Date:** 2026-06-27
**Status:** DONE

## What Was Implemented

### 1. `src/reap/metrics.py` — `_partial_update()` on `OnlineStatsTracker`
- Added `_partial_update(self, expert_idx, new_mean, new_count)` method
- Uses Welford + Kahan summation to update a single expert index without touching others
- Mirrors the existing `update()` logic but operates on scalars at specific indices

### 2. `src/reap/pruning_metrics.py` — `update_pruning_state_single_expert()`
- New function for accumulating pruning saliency metrics for ONE expert at a time
- Accepts `expert_output`, `router_logits`, `selected_experts`, optional `valid_token_mask`
- Computes routing weights, finds active tokens, computes EAN norm, updates all layer_state fields
- Skips experts with no active tokens (early return)
- Handles top-k selection and renormalize_router_weights

### 3. `src/reap/layerwise_observer.py` — `input_ids` in `ReplayBatch`
- Added `input_ids: Optional[torch.Tensor] = None` field to `ReplayBatch` dataclass
- Updated `ReplayCache.append()` to accept `input_ids` parameter
- Updated `ReplayCache.materialize()` to propagate `input_ids` into kwargs
- Updated `intercept_entry_inputs()` to capture `input_ids` from model kwargs and store in `ReplayBatch`

### 4. `src/reap/v4_moe_observer.py` — New file
- **`DeepseekV4MoEObserver(LayerwiseMoEObserver)`** — overrides `_process_moe_activations()` with:
  - Router dispatch: TopKRouter (no input_ids) vs HashRouter (requires input_ids)
  - Incremental expert loop: `F.linear(flat_input, gate_up_proj[idx])` → SiLU gate*up → `F.linear(_, down_proj[idx])`
  - Gate/up clamping via `limit` attribute
  - Calls `update_pruning_state_single_expert()` per expert
  - Memory cleanup every 32 experts
- **`register_v4_standard_hooks()`** — hooks DeepseekV4TopKRouter/HashRouter submodules directly (avoiding broken MoE block hook path)

### 5. `tests/test_v4_moe_observer.py` — 18 tests
- 6 tests for `OnlineStatsTracker._partial_update()`
- 6 tests for `update_pruning_state_single_expert()`
- 3 tests for `DeepseekV4MoEObserver` class structure
- 3 tests for `ReplayBatch` `input_ids` propagation

## Commands Run & Results

```bash
# New V4 observer tests: 18 passed
pytest tests/test_v4_moe_observer.py -v

# Regression tests (Tasks 1+2): 16 passed
pytest tests/test_v4_model_registration.py tests/test_v4_block_loader.py -v

# All relevant tests: 50 passed, 7 pre-existing failures
pytest tests/test_v4_moe_observer.py tests/test_v4_model_registration.py \
  tests/test_v4_block_loader.py tests/test_pruning_metrics.py \
  tests/test_layerwise_observer.py tests/test_layerwise_model_utils.py -v

# Lint: no new issues (1 pre-existing SIM108 warning unchanged)
ruff check src/reap/v4_moe_observer.py src/reap/metrics.py \
  src/reap/pruning_metrics.py src/reap/layerwise_observer.py \
  tests/test_v4_moe_observer.py
```

**Pre-existing failures (7 total, all unrelated):**
- `test_layerwise_observer`: Qwen3 `num_experts` attr, Ernie experts not iterable
- `test_layerwise_model_utils`: DeepseekV2Config strict dataclass, Mixtral block name mismatch

## Files Changed

| File | Status |
|------|--------|
| `src/reap/v4_moe_observer.py` | **Created** (174 lines) |
| `tests/test_v4_moe_observer.py` | **Created** (185 lines) |
| `src/reap/metrics.py` | **Edited** — added `_partial_update()` (19 lines) |
| `src/reap/pruning_metrics.py` | **Edited** — added `update_pruning_state_single_expert()` (62 lines) |
| `src/reap/layerwise_observer.py` | **Edited** — `input_ids` in ReplayBatch/ReplayCache/intercept_entry_inputs (15 lines) |

## Self-Review Findings

1. **`total_tokens` not updated in `update_pruning_state_single_expert`** — The brief mentions updating `total_tokens`, but doing so would overcount (each token would be counted for every expert it selected). The caller (`_process_moe_activations`) tracks `total_tokens` once per batch instead.

2. **`pairwise_expert_frequency` not updated** — Requires knowing all expert frequencies simultaneously; can't be computed incrementally per-expert. Would need to be computed in the caller from `selected_experts`.

3. **`_process_moe_activations` signature change** — Added `input_ids` parameter to support HashRouter. This means the call site (`_after_forward`) needs updating when the V4 observer is used in the pipeline — currently `_after_forward` only passes `(target_device, attention_mask)`.

4. **`register_v4_standard_hooks` is structural but unverified against real model** — The hook factory captures router logits from `output[0]` and runs the incremental expert loop. Needs end-to-end testing with a real V4 model.

## Concerns

1. **`_after_forward` in `_record_activations_for_block`** — Currently passes `(target_device, attention_mask)` to `_process_moe_activations`. The V4 override needs `input_ids` too. The `_after_forward` closure would need modification to capture `input_ids` from replay kwargs when using the V4 observer.

2. **HashRouter detection** — Uses `hasattr(moe_module.gate, 'is_hash')` which depends on V4's modeling. Alternative: use `config.mlp_layer_types[layer_idx] == "hash_moe"` with the model config passed through.

3. **`limit` fallback** — Defaults to `10.0` (matching V4 Flash's `swiglu_limit`), but should ideally be read from `moe_module.experts.limit` or `model.config.swiglu_limit`.

## Report Path
`.superpowers/sdd/task-03-report.md`
