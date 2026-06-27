# Fix C: Phase 2 + Phase 3 + Phase 5.1 — Block-from-disk in observer and pruning

This is the core architectural fix. The pipeline needs to load real weights from disk before forwarding through blocks and before pruning.

## Files to Edit/Create
- `src/reap/v4_moe_observer.py` — Major changes
- `src/reap/layerwise_observer.py` — input_ids passthrough
- `src/reap/layerwise_prune.py` — Pruning with real weights
- `src/reap/pruning_metrics.py` — Device handling fix

## Reference Files
- `src/reap/v4_block_loader.py` — V4BlockDiskLoader API
- `src/reap/v4_prune_utils.py` — _prune_v4_layer
- `src/reap/layerwise_observer.py` — LayerwiseMoEObserver base class

## Issues to Fix

### C1: Observer block loading uses meta tensors (Phase 5.1 C1/C4)

The `DeepseekV4MoEObserver` inherits `_load_block_for_replay()` from `LayerwiseMoEObserver` which does `block.to("cuda")` — this crashes for meta-device blocks because PyTorch can't forward through meta tensors mixed with real CPU/CUDA inputs.

**Fix:** Override `_load_block_for_replay` and `_offload_current_block` in `DeepseekV4MoEObserver`:

```python
class DeepseekV4MoEObserver(LayerwiseMoEObserver):
    def __init__(self, model, hook_config, block_names=None, v4_loader=None):
        super().__init__(model, hook_config, block_names)
        self._v4_loader = v4_loader
        self._loaded_layer_device = None
    
    def _load_block_for_replay(self, block_idx):
        if self.currently_loaded_block_idx == block_idx:
            return safe_get_device(self.blocks[block_idx])
        
        # Unload previous block
        self._offload_current_block()
        
        # Load new block from disk
        if self._v4_loader is not None:
            self._v4_loader.load_layer(self.blocks, block_idx)
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._move_block(self.blocks[block_idx], block_idx, target_device)
        
        self.currently_loaded_block_idx = block_idx
        return safe_get_device(self.blocks[block_idx])
    
    def _offload_current_block(self):
        block_idx = self.currently_loaded_block_idx
        if block_idx < 0:
            return
        if self._v4_loader is not None:
            self._v4_loader.unload_layer(None, clear_shard_cache=False)
        self.currently_loaded_block_idx = -1
        cleanup_memory(synchronize=False)
```

The `v4_loader.load_layer(blocks, block_idx)` should:
1. Read weights from safetensor shards
2. Decompress FP4→BF16
3. Load state_dict into the meta block (via `load_state_dict(..., assign=True)`)
4. Return the loaded block

**Problem:** `V4BlockDiskLoader.load_layer()` currently creates a NEW `DeepseekV4DecoderLayer` from config, not loading into an existing meta block. We need to add a new method `load_into_block(block, block_idx)` that loads weights from disk into an existing meta block's parameters.

Add to `V4BlockDiskLoader`:
```python
def load_into_block(self, block, layer_idx):
    """Load real BF16 weights from disk into an existing meta block."""
    def _load_and_dequant(name):
        raw = self._load_tensor(name)
        scale_name = name.replace(".weight", ".weight_scale_inv")
        try:
            scales = self._load_tensor(scale_name)
        except (KeyError, FileNotFoundError):
            return raw.to(torch.bfloat16) if hasattr(raw, 'to') else raw
        return dequantize_fp4_weight(raw, scales)
    
    state_dict = {}
    for key in ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", ...]:
        # build key list from layer_map for this layer
        ...
    block.load_state_dict(state_dict, strict=False, assign=True)
    return block
```

### C2: input_ids passthrough (Phase 2 C1)

Hash routers need `input_ids` for TID→EID lookup. Currently `_after_forward` only passes `(target_device, attention_mask)`.

**Fix in `layerwise_observer.py`:**

Change the callback signatures:
```python
def _after_forward(
    target_device: torch.device,
    attention_mask: Optional[torch.Tensor],
    block_kwargs: Optional[Dict[str, Any]] = None,  # NEW
) -> None:
```

And in `_forward_block`, pass `block_kwargs` to the callback:
```python
if after_forward is not None:
    after_forward(target_device, attention_mask, block_kwargs)
```

**Fix in `v4_moe_observer.py`:**

In `DeepseekV4MoEObserver._process_moe_activations`:
```python
def _process_moe_activations(self, block_idx, moe_module, input_hidden_states,
                               device, attention_mask=None, input_ids=None):
    ...
```

The `DeepseekV4MoEObserver` should also override `_record_activations_for_block` to create a V4-specific `_after_forward` closure that captures `input_ids` from `replay_kwargs`:

```python
def _record_activations_for_block(self, block_idx, moe_module=None, **kwargs):
    # ... similar to parent but with V4-specific _after_forward
    # that captures input_ids from replay_kwargs
    _input_ids = None
    
    def _after_forward(target_device, attention_mask, block_kwargs=None):
        nonlocal _input_ids
        if block_kwargs and "input_ids" in block_kwargs:
            _input_ids = block_kwargs["input_ids"]
        moe_input = captured_moe_input.get("input")
        self._process_moe_activations(
            block_idx, moe_module, moe_input,
            target_device, attention_mask=attention_mask,
            input_ids=_input_ids,
        )
```

### C3: Fix register_v4_standard_hooks hash router handling (Phase 2 C2/C3)

In `v4_moe_observer.py:register_v4_standard_hooks()`:

```python
def register_v4_standard_hooks(model, hook_config, state):
    handles = []
    
    for name, module in model.named_modules():
        if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            _top_k = module.top_k
            _is_hash = getattr(module, 'is_hash', False)
            
            def make_hook(m, is_hash, top_k):
                def hook(mod, args, output):
                    # output is (logits, weights, indices)
                    router_logits = output[0]
                    if is_hash:
                        # For hash routers, indices come from the router's static lookup
                        indices = output[2]  # NOT torch.topk
                        # input_ids is args[1] for hash routers
                        # But we don't need it for indices since output[2] has them
                    else:
                        indices = output[2]  # Same for top-k
                    
                    # ... metrics computation ...
                return hook
            
            handles.append(module.register_forward_hook(make_hook(module, _is_hash, _top_k)))
    
    return handles
```

**Critical:** The existing code at `v4_moe_observer.py:194` does `indices = torch.topk(router_logits, _top_k, dim=-1)[1]` unconditionally. This is wrong for hash routers. Use `output[2]` from the router's output tuple for BOTH router types.

Also ensure `args[1]` is captured for hash routers if needed (though `output[2]` already contains the correct indices).

### C4: Pruning with real weights (Phase 3 C1 + Phase 5.1 C3)

In `layerwise_prune.py`, the pruning reload path currently creates a fresh meta model. Instead, use V4BlockDiskLoader to load all layers one at a time for pruning:

```python
if _is_v4_model_from_name(model_name):
    from reap.v4_block_loader import V4BlockDiskLoader
    v4_loader = V4BlockDiskLoader(model_name, config=config)
    v4_loader.load_non_backbone_modules(model)
    
    # Load each layer from disk, prune, save
    for i in range(len(layers_to_prune)):
        v4_loader.load_into_block(model.model.layers[i], i)
        # ... prune_model operates on model.model.layers[i].mlp
        # After pruning, the layer's weights are compacted in-place
        v4_loader.unload_layer_to_disk(model.model.layers[i])
    
    model.save_pretrained(pruned_model_dir)
```

BUT this is complex. The `prune_model()` function expects to iterate over observer_data and prune each layer. If the layers are loaded one-at-a-time, `prune_model()` needs to work with partially-loaded layers.

**Simpler approach:** Load ALL layers on CPU (one at a time for memory efficiency) before calling prune:

```python
if _is_v4_model_from_name(model_name):
    from reap.v4_block_loader import V4BlockDiskLoader
    v4_loader = V4BlockDiskLoader(model_name, config=config)
    v4_loader.load_non_backbone_modules(model)
    # Load all layers on CPU (one at a time to stay within 180 GB)
    for i in range(model.config.num_hidden_layers):
        v4_loader.load_into_block(model.model.layers[i], i)
    
    # Now model has real weights on CPU
    prune_model(observer_data, model, prune_args, n_experts_to_prune, pruned_model_dir)
    # save_pretrained will work because weights are real
```

Memory: each layer is ~13 GB BF16, but after loading, the next layer replaces it via `load_state_dict(assign=True)`. Peak: ~13 GB + non-backbone. After prune, all layers are materialized on CPU (~560 GB) — OOPS.

So we need to prune layer-by-layer too. The cleanest approach:

Modify the prune loop to load one layer at a time from disk, prune its MoE, save the compacted weights to a temp dir, then move to the next layer.

Actually, let me think about this differently. The current `prune_model()` does:
1. Iterate over layers
2. For each layer, compute which experts to keep
3. Call `_prune_v4_layer(moe, retained_indices)` which indexes the 3D params
4. After all layers are pruned, `model.save_pretrained(dir)`

For V4, step 3 operates on meta tensors. We need to:
- Before step 3: load real weights for this layer from disk
- Step 3: prune (index 3D params)
- After step 3: save the compacted layer weights to a temp safetensor (or keep in memory)
- Free the layer
- Continue

Best approach: Override the prune loop for V4. Create `_prune_v4_model()` function in `v4_prune_utils.py`:

```python
def prune_v4_model(observer_data, model, v4_loader, n_experts_to_prune, pruned_model_dir, ...):
    for layer_idx in range(model.config.num_hidden_layers):
        # Load layer from disk
        v4_loader.load_into_block(model.model.layers[layer_idx], layer_idx)
        
        # Compute which experts to retain (same logic as prune.py)
        retained_expert_indicies = ...  # from observer_data[layer_idx]
        
        # Prune this layer
        _prune_v4_layer(model.model.layers[layer_idx].mlp, retained_expert_indicies)
        
        # Free layer
        model.model.layers[layer_idx].to("meta")
        gc.collect()
    
    # Save pruned model
    model.save_pretrained(pruned_model_dir)
```

This keeps CPU usage at ~13 GB + non-backbone (~2 GB) + intermediate tensors = < 25 GB.

## Implementation Plan

1. **V4BlockDiskLoader**: Add `load_into_block(block, layer_idx)` method
2. **LayerwiseMoEObserver**: Add `block_kwargs` to `_after_forward` callback
3. **DeepseekV4MoEObserver**: Override `_load_block_for_replay`, `_offload_current_block`, `_record_activations_for_block`
4. **v4_moe_observer.py**: Fix `register_v4_standard_hooks` hash router handling
5. **v4_prune_utils.py**: Add `prune_v4_model()` for layer-by-layer pruning
6. **layerwise_prune.py**: Wire V4-specific pruning function
7. **pruning_metrics.py**: Fix device handling for input_ids

## Tests

Write/update tests:
- `test_v4_block_loader.py`: Test `load_into_block` (mock safetensors)
- `test_v4_moe_observer.py`: Test block-from-disk observer overrides
- Existing tests should pass

## Environment

PYTHONPATH="src"
Python: `C:\Users\pauma\miniconda3\envs\py310\python.exe`

## Commit

Message: "Fix C: block-from-disk integration in observer and pruning, input_ids passthrough"
