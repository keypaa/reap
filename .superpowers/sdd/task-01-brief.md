# Task 1: Phase 0 — V4 Model Registration

## Context
This is the first task implementing REAP pruning support for DeepSeek V4 Flash (284B). Two files need edits; one new test file.

## Plan Text
(From `docs/superpowers/specs/2026-06-27-block-from-disk-plan.md` sections 0.1–0.6)

### 0.1 Add `DeepseekV4ForCausalLM` to `MODEL_ATTRS`
```python
"DeepseekV4ForCausalLM": {
    "moe_block": "mlp",
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
    "experts": "experts",
    "fused": False,  # Nominal placeholder; V4 intercepts before fused check
    "router": "gate",
    "num_experts": "num_local_experts",
    "num_experts_per_tok": "num_experts_per_tok",
},
```

**Design choice:** Do NOT set `fused` to True. Instead, V4-specific interception in prune.py and observers uses `_is_v4_model()` to bypass both fused branches.

### 0.2 Create `_is_v4_model()` Helper

```python
def _is_v4_model(model) -> bool:
    """Check if model is a DeepSeek V4 variant by class name."""
    return "DeepseekV4" in model.__class__.__name__
```

Used by observers (Phase 2) and prune.py (Phase 3). Place in `model_util.py`.

### 0.3 Add `DeepseekV4MoEObserverHookConfig` Dataclass

In `observer.py`:
```python
@dataclass
class DeepseekV4MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "DeepseekV4SparseMoeBlock"
    num_experts_attr_name: str = "experts.num_experts"
    top_k_attr_name: str = "gate.top_k"
    fused_experts: bool = False  # Nominal only; V4 observer bypasses this
```

### 0.4 Register in `OBSERVER_CONFIG_REGISTRY`
```python
"DeepseekV4ForCausalLM": DeepseekV4MoEObserverHookConfig,
```

### 0.5 Block Detection — No Code Change
`DECODER_BLOCK_PATTERNS` in `layerwise_model_utils.py:37` includes `r"\.layers\.\d+$"` which matches V4's `model.layers.N`. Already works. Verify with `find_decoder_blocks(model)`.

### 0.6 Write Unit Test

New test file `tests/test_v4_model_registration.py`:
- `test_model_attrs_v4()` — assert `MODEL_ATTRS["DeepseekV4ForCausalLM"]` has correct keys and values
- `test_is_v4_model()` — assert `_is_v4_model()` returns True for mock with class name containing `DeepseekV4`
- `test_is_v4_model_other()` — assert returns False for `Qwen3MoeForCausalLM` and `MixtralForCausalLM`
- `test_v4_observer_config()` — assert `DeepseekV4MoEObserverHookConfig` has correct field values
- `test_registry_contains_v4()` — assert `OBSERVER_CONFIG_REGISTRY["DeepseekV4ForCausalLM"]` is `DeepseekV4MoEObserverHookConfig`

## Interfaces

### Files to Edit
- `src/reap/model_util.py` — Add MODEL_ATTRS entry + `_is_v4_model()`
- `src/reap/observer.py` — Add dataclass + registry entry

### Files to Create
- `tests/test_v4_model_registration.py`

## Global Constraints
- Follow existing code style (ruff SIM rules, line-length 88, no mypy)
- No comments unless required by project convention
- `fused` must be `False` as placeholder (not True)
- Existing tests must still pass after edits

## Report
Write results to `.superpowers/sdd/task-01-report.md` with: status (DONE/NEEDS_CONTEXT/BLOCKED), commits, test results summary, and any concerns.
