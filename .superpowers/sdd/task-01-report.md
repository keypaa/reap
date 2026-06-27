# Task 1 Report: Phase 0 — V4 Model Registration

**Status:** DONE

## What Was Implemented

1. **`src/reap/model_util.py`** — Added `DeepseekV4ForCausalLM` entry to `MODEL_ATTRS` (lines 119-128) with the 3D parameter path scheme (`gate_proj`, `up_proj` on `experts` object, `fused=False`). Added `_is_v4_model()` helper function at line 120.

2. **`src/reap/observer.py`** — Added `DeepseekV4MoEObserverHookConfig` dataclass (lines 528-533) with `module_class_name_to_hook_regex="DeepseekV4SparseMoeBlock"`, `num_experts_attr_name="experts.num_experts"`, `top_k_attr_name="gate.top_k"`, `fused_experts=False`. Registered in `OBSERVER_CONFIG_REGISTRY` at line 545.

3. **`tests/test_v4_model_registration.py`** — New file with 5 tests:
   - `test_model_attrs_v4()` — asserts MODEL_ATTRS entry keys/values
   - `test_is_v4_model()` — asserts `_is_v4_model()` returns True for class name containing "DeepseekV4"
   - `test_is_v4_model_other()` — asserts False for Qwen3MoeForCausalLM, MixtralForCausalLM
   - `test_v4_observer_config()` — asserts DeepseekV4MoEObserverHookConfig field values
   - `test_registry_contains_v4()` — asserts OBSERVER_CONFIG_REGISTRY entry

## Commands Run & Test Results

```powershell
# New tests (5/5 pass)
$env:PYTHONPATH="src"; & "C:\Users\pauma\miniconda3\envs\py310\python.exe" -m pytest tests/test_v4_model_registration.py -v
# 5 passed in 11.46s

# Relevant existing tests also pass
$env:PYTHONPATH="src"; & "C:\Users\pauma\miniconda3\envs\py310\python.exe" -m pytest tests/test_v4_model_registration.py tests/test_pruning_metrics.py -v
# 7 passed in 11.60s

# Full suite: 16 failed (all pre-existing issues), 24 passed
# Pre-existing failures: Qwen3MoeExperts not iterable, DeepseekV2 n_shared_experts=None validation, Mixtral module naming
```

V4 model class verified in transformers 5.9.0: `DeepseekV4ForCausalLM`, `DeepseekV4Config`, `DeepseekV4SparseMoeBlock` all confirmed available.

## Files Changed

| File | Change |
|------|--------|
| `src/reap/model_util.py` | +16 lines: MODEL_ATTRS entry + `_is_v4_model()` |
| `src/reap/observer.py` | +9 lines: dataclass + registry entry |
| `tests/test_v4_model_registration.py` | New file, 69 lines, 5 tests |

## Self-Review Findings

- Values match the spec exactly: `moe_block="mlp"`, `gate_proj="gate_proj"`, `up_proj="up_proj"`, `down_proj="down_proj"`, `experts="experts"`, `fused=False`, `router="gate"`, `num_experts="num_local_experts"`, `num_experts_per_tok="num_experts_per_tok"`
- V4's `DeepseekV4SparseMoeBlock.mlp.experts` returns a `DeepseekV4Experts` object (not iterable `ModuleList`), and `experts.gate_up_proj` / `experts.down_proj` are 3D `nn.Parameter` tensors — confirmed via runtime introspection
- `_is_v4_model()` uses string containment check (`"DeepseekV4" in cls_name`), which covers all V4 variants without hardcoding every class name
- No comments added (project convention)
- DeepseekV4Config uses `attribute_map` (maps `num_local_experts` → `n_routed_experts`, `intermediate_size` → `moe_intermediate_size`), so `num_local_experts` and `num_experts_per_tok` resolve correctly from config
- Verified: `moe.experts.num_experts = 2`, `moe.gate.top_k = 1` with the test config

## Concerns

None. All spec values match, all new tests pass, existing pass rate unchanged.
