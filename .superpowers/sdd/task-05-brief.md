# Task 5: Phase 5.1 — Integration Wiring

Wire V4 components into the layerwise pipeline dispatch. This is code-only — no GPU execution.

## Files to Edit
- `src/reap/layerwise_prune.py` — Observer dispatch, model loading dispatch
- `src/reap/layerwise_observer.py` — Minimal changes if any

## Files to Create
- `tests/test_v4_pipeline_dispatch.py` — Unit tests for dispatch logic

## Requirements

### 1. Observer Dispatch in `record_activations_layerwise()`

In `src/reap/layerwise_prune.py`, the `record_activations_layerwise()` function currently creates a `LayerwiseMoEObserver` at line 179:

```python
observer = LayerwiseMoEObserver(
    model=model,
    hook_config=hook_config,
)
```

Add V4 dispatch using `_is_v4_model()`:

```python
from reap.model_util import _is_v4_model
from reap.v4_moe_observer import DeepseekV4MoEObserver

if _is_v4_model(model):
    observer = DeepseekV4MoEObserver(
        model=model,
        hook_config=hook_config,
    )
else:
    observer = LayerwiseMoEObserver(
        model=model,
        hook_config=hook_config,
    )
```

### 2. Model Loading Dispatch in `main()`

Currently lines 280-293 load the full model on CPU. For V4, this OOMs (560 GB BF16). Add V4 dispatch:

```python
if _is_v4_model_from_name(model_name):
    # V4: load on meta device, use V4BlockDiskLoader for non-backbone
    logger.info("Loading DeepSeek V4 model on meta device...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    from reap.v4_block_loader import V4BlockDiskLoader
    v4_loader = V4BlockDiskLoader(model_name, config=config)
    v4_loader.load_non_backbone_modules(model)
    # Store loader on model for use by observer
    model._v4_block_loader = v4_loader
else:
    # Existing code
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        ...
    )
```

For this registration we need `_is_v4_model_from_name()` or just check the model name string.

### 3. Pruning Model Loading Dispatch in `main()`

Lines 357-369 reload model on GPU for pruning. For V4:

```python
if _is_v4_model_from_name(model_name):
    # V4: reuse model or reload from shards
    from reap.v4_block_loader import V4BlockDiskLoader
    v4_loader = getattr(model, '_v4_block_loader', None) or V4BlockDiskLoader(model_name)
    v4_loader.load_non_backbone_modules(model)
    # Keep on meta + disk loader — actual pruning uses _prune_v4_layer in prune.py
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        ...
    )
```

### 4. `_is_v4_model_from_name()` helper

Add a simple helper in `layerwise_prune.py`:

```python
def _is_v4_model_from_name(model_name: str) -> bool:
    return "DeepSeek-V4" in model_name or "deepseek-v4" in model_name
```

### 5. Tests

Write `tests/test_v4_pipeline_dispatch.py` with:
- `test_is_v4_model_from_name()` — Verify helper returns True for V4 names, False for others
- `test_is_v4_model()` — Verify `_is_v4_model()` returns True for V4 model mock
- `test_observer_dispatch_v4()` — Mock a V4 model, verify `record_activations_layerwise` creates `DeepseekV4MoEObserver`
- `test_observer_dispatch_non_v4()` — Mock a non-V4 model, verify creates `LayerwiseMoEObserver`

**Note:** These are unit tests — mock the model objects, don't try to load actual models.

## Constraints

1. Don't break existing non-V4 pipeline flow
2. All existing tests must pass
3. Follow existing code style (ruff SIM, line-length 88)

## Environment

Python: `C:\Users\pauma\miniconda3\envs\py310\python.exe`
Test commands:
```powershell
$env:PYTHONPATH="src"; & "C:\Users\pauma\miniconda3\envs\py310\python.exe" -m pytest tests/test_v4_pipeline_dispatch.py -v
$env:PYTHONPATH="src"; & "C:\Users\pauma\miniconda3\envs\py310\python.exe" -m pytest tests/test_v4_prune.py tests/test_v4_moe_observer.py tests/test_v4_block_loader.py tests/test_v4_model_registration.py tests/test_v4_pipeline_dispatch.py -v
$env:PYTHONPATH="src"; & "C:\Users\pauma\miniconda3\envs\py310\python.exe" -m pytest tests/ -v --timeout 60
```

## Report

Write to `.superpowers/sdd/task-05-report.md` with:
- Status
- Commits
- Test results (all V4 + full suite)
- Concerns
