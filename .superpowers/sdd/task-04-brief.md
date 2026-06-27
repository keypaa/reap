# Task 4: Phase 3 — V4-Specific Pruning

## Context

Tasks 1-3 implemented model registration, block-from-disk loading, and the V4 observer. Now we implement the pruning step that removes low-saliency experts from V4's 3D weight tensors.

## Files to Create
- `src/reap/v4_prune_utils.py` — V4 pruning helpers

## Files to Edit
- `src/reap/prune.py` — Add V4 interception branch before `fused` check

## Detailed Requirements

### 1. `v4_prune_utils.py` — `_prune_v4_layer()`

```python
def _prune_v4_layer(moe, retained_indices, model, layer_idx):
    """Prune V4 experts by indexing 3D weight tensors along dim 0.
    
    V4's DeepseekV4Experts stores weights as 3D nn.Parameter tensors:
    - gate_up_proj: [num_experts, 2*intermediate_dim, hidden_dim]
    - down_proj: [num_experts, hidden_dim, intermediate_dim]
    
    Pruning: keep retained_indices along dim 0, update num_experts, 
    prune router gate, remap hash router tid2eid if applicable.
    """
```

Logic:
1. **Expert weights:** Index `gate_up_proj.data[retained_indices]` and `down_proj.data[retained_indices]` along dim 0, clone, create new `nn.Parameter`
2. **Router gate:** Index `moe.gate.weight.data[retained_indices]`, update `moe.gate.out_features`
3. **e_score_correction_bias:** Only on `TopKRouter` — guard with `hasattr`:
   ```python
   if hasattr(moe.gate, "e_score_correction_bias"):
       moe.gate.e_score_correction_bias = nn.Parameter(
           moe.gate.e_score_correction_bias.data[retained_indices].clone()
       )
   ```
4. **Router num_experts:** update `moe.gate.num_experts` if it exists
5. **Hash router tid2eid remap:** Call `_remap_hash_router_tid2eid()` if hash router
6. **Shared experts:** NEVER prune (`moe.shared_experts` unchanged)

### 2. `v4_prune_utils.py` — `_remap_hash_router_tid2eid()`

```python
def _remap_hash_router_tid2eid(gate, old_to_new):
    """Remap tid2eid lookup table after expert pruning.
    
    tid2eid is a registered buffer [vocab_size, top_k] mapping token IDs
    to expert indices. After pruning, old expert indices must be remapped
    to new (compacted) indices.
    
    old_to_new: list where old_to_new[old_idx] = new_idx for retained,
                -1 for pruned (falls back to expert 0).
    """
```

Logic:
- `old_to_new = torch.tensor(old_to_new, device=gate.tid2eid.device, dtype=torch.long)`
- Clone `tid2eid`, clamp to valid range, remap via `old_to_new[tid2eid.clamp(min=0)]`
- Mask out entries that were -1 in original (unused TIDs)
- Use `gate.tid2eid.data.copy_(remapped)` to preserve buffer registration

### 3. `prune.py` — Add V4 interception

In the `prune()` function (prune.py:43-165), add V4 interception BEFORE the `fused` flag check:

```python
def prune(observer_data, model, prune_args, n_experts_to_prune, pruned_model_dir):
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    
    # ... existing super expert logic ...
    
    for layer in tqdm(observer_data, "Pruning layers..."):
        # ... existing expert selection logic (same as current) ...
        
        retained_expert_indices = [...]
        moe = get_moe(model, layer)
        
        # --- V4 INTERCEPTION ---
        if "DeepseekV4" in model.__class__.__name__:
            from reap.v4_prune_utils import _prune_v4_layer
            _prune_v4_layer(moe, retained_expert_indices, model, layer)
        
        elif not model_attrs["fused"]:
            # ... existing ModuleList path (unchanged) ...
        else:
            # ... existing fused path (unchanged) ...
    
    # --- Config update ---
    if "DeepseekV4" in model.__class__.__name__:
        model.config.n_routed_experts = num_retained
        model.config.num_local_experts = num_retained
    else:
        setattr(model.config, model_attrs["num_experts"], retained_experts)
```

### 4. Pruning model loading

The `prune()` function receives a loaded model. For V4, the model must be loaded on CPU (can't fit on GPU). The loading strategy depends on the pipeline:
- **Layerwise pipeline:** Model was already loaded via block-from-disk; after observation, the pruning step loads the full model on CPU with `from_pretrained(device_map="cpu", torch_dtype=torch.bfloat16)` — this WILL OOM on 180 GB.
- **Alternative:** Keep the original `from_pretrained` reference in memory from the layerwise pipeline; do NOT reload.

**For now**, make the pruning function work with whatever model is passed in. The loading strategy is a Phase 5 concern.

### 5. Tests

Write `tests/test_v4_prune.py`:
- `test_prune_v4_layer()` — Create mock 3D params, prune 50% of experts, verify shapes
- `test_remap_hash_router_tid2eid()` — Create mock tid2eid buffer, remap, verify compacted indices
- `test_e_score_correction_bias_guard()` — Verify guard works with/without bias attribute
- `test_config_update()` — Verify config fields are updated correctly
- `test_shared_experts_unchanged()` — Verify shared_experts left untouched

## Environment

Python: `C:\Users\pauma\miniconda3\envs\py310\python.exe`
Test command: `$env:PYTHONPATH="src"; & "C:\Users\pauma\miniconda3\envs\py310\python.exe" -m pytest tests/test_v4_prune.py tests/test_v4_moe_observer.py tests/test_v4_block_loader.py tests/test_v4_model_registration.py -v`

## Report

Write to `.superpowers/sdd/task-04-report.md` with status, commits, test results, concerns.
