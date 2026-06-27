# Fix A: Phase 0 — MODEL_ATTRS + Standard Path Guard

## Files to Edit
- `src/reap/model_util.py`
- `src/reap/observer.py`
- `src/reap/v4_moe_observer.py`
- `src/reap/main.py`

## Issues to Fix

### A1: MODEL_ATTRS gate/up_proj wrong for V4

V4's `DeepseekV4Experts` stores weights as a single fused `gate_up_proj` 3D param `[N, 2*D, D]`, not separate `gate_proj`/`up_proj`. The registry has `"gate_proj": "gate_proj"`, `"up_proj": "up_proj"`.

**Fix:** Add V4-specific resolution in `assert_merge()` and `assert_tied_weights()`. When `_is_v4_model(model)`, check `gate_up_proj` directly instead of separate proj attrs:

```python
def get_moe_attr_names(model):
    """Return (gate_proj, up_proj, down_proj) handling V4's fused gate_up_proj."""
    attrs = MODEL_ATTRS.get(model.__class__.__name__)
    if not attrs:
        raise ValueError(f"Unknown model class: {model.__class__.__name__}")
    gate = attrs["gate_proj"]
    up = attrs.get("up_proj", gate)
    down = attrs["down_proj"]
    if _is_v4_model(model):
        gate = "gate_up_proj"
        up = "gate_up_proj"
    return gate, up, down
```

Use this in `assert_merge()` and `assert_tied_weights()` to resolve the correct attribute names for V4.

### A2: Block V4 from standard (non-layerwise) observer path

The standard observer at `main.py` → `_setup_observer()` → `observer.py:MoETransformerObserver` doesn't handle V4's architecture. It would crash on `enumerate(module.experts)` because V4's experts are `DeepseekV4Experts` (3D params), not a `ModuleList`.

**Fix:** In `src/reap/main.py` `_setup_observer()` (or wherever the standard observer is created), add:

```python
from reap.model_util import _is_v4_model
if _is_v4_model(model):
    raise RuntimeError(
        "DeepSeek V4 does not support the standard observation pipeline. "
        "Use `python -m reap.layerwise_prune` instead."
    )
```

This gives users a clear error before they hit an inscrutable crash.

### A3: register_v4_standard_hooks is orphaned — either wire it or document it

`v4_moe_observer.py:register_v4_standard_hooks()` exists but is never called from any pipeline code path.

**Fix:** Document that this function is for standalone/custom usage, not part of the automatic pipeline. Add a docstring explaining its purpose and usage. Do NOT wire it into the standard pipeline — that would require rewriting `MoETransformerObserver` which is out of scope.

## Tests

Update existing tests to reflect the changes. At minimum verify:
- `_is_v4_model(model)` works after assert_merge changes
- Error raised when V4 enters standard path
- register_v4_standard_hooks docstring exists

## Environment

PYTHONPATH="src"
Python: `C:\Users\pauma\miniconda3\envs\py310\python.exe`

## Commit

Message: "Fix A: correct MODEL_ATTRS for V4 fused gate_up_proj, guard standard observer path"
