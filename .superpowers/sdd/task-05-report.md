# Task 5: Phase 5.1 — Integration Wiring Report

## Status: DONE

## Commits
- `a2618fd` — Phase 5.1: wire V4 components into layerwise pipeline

## Files Changed
- **`src/reap/layerwise_prune.py`** — V4 dispatch in `record_activations_layerwise()` (observer selection), `main()` model loading (meta device + `V4BlockDiskLoader`), and pruning reload. Imports `_is_v4_model` and `_is_v4_model_from_name` from `model_util`.
- **`src/reap/model_util.py`** — Added `_is_v4_model_from_name()` helper alongside existing `_is_v4_model`.
- **`src/reap/v4_block_loader.py`** — `load_non_backbone_modules()` now accepts optional `model` parameter; when provided, sets loaded modules (`embed_tokens`, `norm`, `lm_head`) on the model.
- **`tests/test_v4_pipeline_dispatch.py`** — New: 17 tests covering `_is_v4_model_from_name`, `_is_v4_model`, observer class hierarchy, dispatch source patterns, and block loader interface.

## Deviations from Brief
1. **`_is_v4_model_from_name` moved to `model_util.py`** instead of staying in `layerwise_prune.py`. Reason: it belongs alongside `_is_v4_model` and is testable without heavy deps (vllm, lm_eval). `layerwise_prune.py` imports it from `model_util`.
2. **Observer dispatch tested via source AST analysis** instead of runtime mock. Reason: importing `reap.layerwise_prune` requires vllm, lm_eval, and other heavy dependencies not installed in the test environment. AST verification confirms the dispatch pattern exists and is syntactically correct.

## Test Results
- **V4 tests (65/65 pass)**: `test_v4_prune.py`, `test_v4_moe_observer.py`, `test_v4_block_loader.py`, `test_v4_model_registration.py`, `test_v4_pipeline_dispatch.py`
- **Full suite (86/102 pass, 16 pre-existing failures)**:
  - 2 collection errors: missing `vllm` (pre-existing)
  - 14 test failures: transformers API incompatibilities (DeepSeekV2 strict dataclasses, Qwen3MoeExperts non-iterable, etc.) — all pre-existing, none related to this change

## Lint Results
- All checks passed on changed files (`src/reap/layerwise_prune.py`, `src/reap/model_util.py`, `src/reap/v4_block_loader.py`, `tests/test_v4_pipeline_dispatch.py`)
- 10 pre-existing warnings in other files (SIM108, SIM105, SIM210, SIM102, SIM910)

## Concerns
- The `V4BlockDiskLoader.load_non_backbone_modules` creates fresh modules on CPU. For the layerwise pipeline, the model is on meta device, and these modules replace the meta counterparts. This should work but hasn't been integration-tested with an actual V4 model (no GPU available).
