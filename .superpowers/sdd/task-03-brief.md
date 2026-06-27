# Task 3: Phase 2 — DeepseekV4MoEObserver with Incremental Expert Loop

## Context

Task 1 registered V4 in MODEL_ATTRS/OBSERVER_CONFIG_REGISTRY. Task 2 built the block-from-disk loader. Now we implement the V4-specific observer that collects pruning metrics incrementally (one expert at a time) without materializing the 17 GB `[E, T, D]` activation tensor.

**Key architecture difference:** V4's `DeepseekV4Experts` stores weights as 3D `nn.Parameter` tensors (e.g., `gate_up_proj: [N, 2*D, D]`), not as `ModuleList` of per-expert modules. So `enumerate(moe_module.experts)` iterates over `nn.Parameter` attributes (gate_up_proj, down_proj, act_fn, limit), not experts. We must use `moe_module.experts.gate_up_proj[expert_idx]` syntax for per-expert `F.linear`.

**Task scope:** 4 files to edit, 1 new file, ~11 checkboxes from the plan.

## Files to Create
- `src/reap/v4_moe_observer.py` — `DeepseekV4MoEObserver` class

## Files to Edit
- `src/reap/pruning_metrics.py` — Add `update_pruning_state_single_expert()`
- `src/reap/metrics.py` — Add `_partial_update()` to `OnlineStatsTracker`
- `src/reap/layerwise_observer.py` — Add `input_ids` to `ReplayBatch`, update `intercept_entry_inputs`, V4 branching in `_process_moe_activations`
- `tests/` — New or updated test file for V4 observer

## Detailed Requirements

### 1. `pruning_metrics.py`: Add `update_pruning_state_single_expert()`

Add a new function alongside the existing `update_pruning_state`. This function accumulates metrics for ONE expert at a time instead of iterating over an `[E, T, D]` tensor:

```python
def update_pruning_state_single_expert(
    layer_state: dict[str, Any],
    expert_idx: int,
    expert_output: torch.Tensor,
    router_logits: torch.Tensor,
    selected_experts: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
    renormalize_router_weights: bool = False,
) -> None:
```

Logic:
1. Compute routing weights via `F.softmax(router_logits, dim=1, dtype=torch.float)`
2. Find tokens that selected this expert: `active_mask = (selected_experts == expert_idx).any(dim=-1)`
3. Apply valid_token_mask if provided
4. If no active tokens, return early
5. Compute EAN norm for active outputs: `torch.linalg.norm(active_outputs, dim=-1)`
6. Update layer_state fields: `ean_sum`, `ean_mean` (via `_partial_update`), `weighted_ean_sum`, `weighted_expert_frequency_sum`, `reap`, `max_activations`
7. Also update `total_tokens` and `expert_frequency`

NOTE: The `_partial_update` method on `OnlineStatsTracker` doesn't exist yet. It will be added in requirement 2 below. For now, design the function so it calls `layer_state["ean_mean"]._partial_update(expert_idx, value, count)` and `layer_state["reap"]._partial_update(expert_idx, value, count)`.

### 2. `metrics.py`: Add `_partial_update()` to `OnlineStatsTracker`

```python
def _partial_update(self, expert_idx: int, new_mean: torch.Tensor, new_count: torch.Tensor):
    """Update statistics for a SINGLE expert index, not the full shape.
    
    This is a simplified version of `update()` that updates only one
    index in the OnlineStatsTracker, which is needed for V4's incremental
    expert loop where we process one expert at a time.
    """
```

Logic:
- `updated_count = self.count + new_count` (but for the specific index — need to think about how count works)
- Actually, since the tracker stores per-expert state, we need to update just ONE expert's entry

Looking at the existing `update()` method (metrics.py:258-284):
- `self.count` has shape `(num_experts,)` — each count is per-expert
- `self.mean` has shape `(num_experts,)` — each mean is per-expert
- For `_partial_update(expert_idx, new_mean, new_count)`:
  - `updated_count = self.count[expert_idx] + new_count`
  - Apply Welford + Kahan update for `self.mean[expert_idx]` only

### 3. `v4_moe_observer.py`: `DeepseekV4MoEObserver`

Extend `LayerwiseMoEObserver` with V4-specific overrides:

```python
class DeepseekV4MoEObserver(LayerwiseMoEObserver):
    """V4-specific observer with incremental expert loop for 3D parameters."""
```

Key overrides:

**`_process_moe_activations()`** — Complete override:
1. Get num_experts from `moe_module.experts.num_experts` (not via `reduce(getattr, ...)`)
2. Get top_k from `moe_module.gate.top_k`
3. Initialize state if needed
4. Call router directly: `moe_module.gate(flat_input)` for TopKRouter or `moe_module.gate(flat_input, input_ids)` for HashRouter
5. Extract logits and selected_experts from router output `(logits, weights, indices)`
6. For each expert: `F.linear(flat_input, moe_module.experts.gate_up_proj[expert_idx])` → gate_up → chunk → SiLU gate ⋅ up → `F.linear(result, moe_module.experts.down_proj[expert_idx])` → call `update_pruning_state_single_expert()` → cleanup intermediates
7. Apply `moe_module.experts.limit` for gate/up clamping: `gate.clamp(max=self.limit)` and `up.clamp(min=-self.limit, max=self.limit)`

**`_find_moe_module_in_block()`** — Same as parent (class-name based, already works)

**Router extraction** — Must handle two router types:
- `DeepseekV4TopKRouter.forward(hidden_states)` → returns `(logits, weights, indices)`
- `DeepseekV4HashRouter.forward(hidden_states, input_ids)` → returns `(logits, weights, indices)`
- Use `model.config.mlp_layer_types[layer_idx] == "hash_moe"` to detect hash layers

### 4. `layerwise_observer.py`: Add `input_ids` to `ReplayBatch`

```python
@dataclass
class ReplayBatch:
    inputs: List[torch.Tensor]
    kwargs: Dict[str, Any]
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None  # NEW: needed for V4 hash router
```

Also update `ReplayCache.append()`, `ReplayCache.materialize()`, and `intercept_entry_inputs()` to capture and carry `input_ids` from the model's `input_ids` kwarg.

**CRITICAL:** `intercept_entry_inputs` is in `_capture_first_block_inputs` (line 430). The `input_ids` are passed as kwargs to the model's forward (which passes them through to decoder layers). Capture them from `kwargs.get("input_ids")`.

### 5. Main pipeline change

The `record_activations_layerwise` function or the main entry point needs to instantiate `DeepseekV4MoEObserver` instead of `LayerwiseMoEObserver` when the model is V4. This check uses `_is_v4_model(model)` from `model_util.py`.

Where this goes depends on how the pipeline dispatches. If using `layerwise_prune.py`, the instantiation happens there. For now, implement the class so it can be swapped in.

### 6. Standard (non-layerwise) observer hook

The standard `MoETransformerObserver._hook_factory` in `observer.py` (line 316-476) has two broken branches for V4:
- `fused=True` path: uses `moe.router` (V4 uses `moe.gate`)
- `fused=False` path: uses `enumerate(module.experts)` (V4 experts is not iterable)

**Fix:** Register forward hooks on `module.gate` directly (the router submodule) instead of on the MoE block. This avoids the broken fused/non-fused branch entirely.

Create a `register_v4_standard_hooks()` helper function in `v4_moe_observer.py` that hooks `DeepseekV4TopKRouter/HashRouter` submodules to capture logits.

### 7. Tests

Write `tests/test_v4_moe_observer.py` with:
- `test_update_pruning_state_single_expert()` — Create a simple 2-expert state, feed tokens, verify metrics are computed correctly
- `test_online_stats_partial_update()` — Verify `_partial_update` correctly updates one expert index
- `test_v4_moe_observer_import()` — Verify the class imports and is a subclass of LayerwiseMoEObserver

## Environment

Python: `C:\Users\pauma\miniconda3\envs\py310\python.exe`
Test command: `$env:PYTHONPATH="src"; & "C:\Users\pauma\miniconda3\envs\py310\python.exe" -m pytest tests/test_v4_moe_observer.py tests/test_v4_block_loader.py tests/test_v4_model_registration.py -v`

## Reference Files

Read these before implementing:
- `src/reap/layerwise_observer.py` (full file) — `LayerwiseMoEObserver` class with `_process_moe_activations`, `ReplayBatch`, `intercept_entry_inputs`
- `src/reap/pruning_metrics.py` (full file) — existing `update_pruning_state` function (follow the same pattern)
- `src/reap/metrics.py` (full file) — `OnlineStatsTracker` class with `update()` method
- `src/reap/observer.py` (full file) — `MoETransformerObserver._hook_factory` for the standard observer fix

## Report

Write results to `.superpowers/sdd/task-03-report.md`.
