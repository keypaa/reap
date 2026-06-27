# Block-From-Disk Implementation Plan: REAP Pruning for DeepSeek-V4-Flash

**Date:** 2026-06-27
**Target:** DeepSeek-V4-Flash (284B) on RTX PRO 6000 96GB
**Base spec:** `2026-06-27-deepseek-v4-reap-support-design.md`
**Key constraint:** 96 GB VRAM, 160 GB model, no full-model CPU load

---

## Phase 0: V4 Model Registration

The `MODEL_ATTRS` dict in `src/reap/model_util.py:7` maps model class names to expert paths. V4 needs an entry, but the `fused` flag is the critical design choice.

### 0.1 Add `DeepseekV4ForCausalLM` to `MODEL_ATTRS`

```python
"DeepseekV4ForCausalLM": {
    "moe_block": "mlp",              # DecoderLayer.mlp = DeepseekV4SparseMoeBlock
    "gate_proj": "gate_proj",         # Reused in merge/assert paths, not prune path
    "up_proj": "up_proj",             # Same
    "down_proj": "down_proj",         # Same
    "experts": "experts",             # moe.experts = DeepseekV4Experts
    "fused": ???,                     # NEITHER True NOR False works — see 0.2
    "router": "gate",                 # moe.gate = DeepseekV4TopKRouter/HashRouter
    "num_experts": "num_local_experts", # Config field (also aliased n_routed_experts)
    "num_experts_per_tok": "num_experts_per_tok",
},
```

**Design choice:** Do NOT set `fused`. Instead, add a V4-specific interception layer in `prune.py` and the observers that checks `model.__class__.__name__` before the `fused` flag. The `fused` concept conflates "3D param storage" with "single-forward-path experts" — V4 has 3D params but does NOT use the fused-router path.

### 0.2 The `fused` Flag Dilemma

| Setting | `prune.py` branch | Why it breaks |
|---------|-------------------|---------------|
| `fused=False` | `ModuleList` path (prune.py:110-134) | `enumerate(moe.experts)` iterates `nn.Parameter` attributes (`gate_up_proj`, `down_proj`, `act_fn`) not experts. Also `all_experts = getattr(moe, "experts")` returns `DeepseekV4Experts`, then `[all_experts[i] for i in ...]` indexes into `nn.Parameter` objects. |
| `fused=True` | Fused path (prune.py:136-145) | `moe.experts.gate_up_proj.data[...]` works for 3D indexing, but `moe.router.weight.data[...]` fails — V4 uses `moe.gate`, not `moe.router`. Also `moe.num_experts` (line 141) should be `moe.experts.num_experts`. |

**Solution:** Add a `models/` module check: `if "DeepseekV4" in model.__class__.__name__:` intercept block in `prune.py` and observers, bypassing both `fused` branches.

### 0.3 Add to `OBSERVER_CONFIG_REGISTRY`

In `src/reap/observer.py:527`:

```python
from reap.observer import MoETransformerObserverConfig

@dataclass
class DeepseekV4MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "DeepseekV4SparseMoeBlock"
    num_experts_attr_name: str = "experts.num_experts"
    top_k_attr_name: str = "gate.top_k"
    fused_experts: bool = False  # Nominal only; V4 observer bypasses this
```

Then register in `OBSERVER_CONFIG_REGISTRY`:

```python
"DeepseekV4ForCausalLM": DeepseekV4MoEObserverHookConfig,
```

### 0.4 `patched_model_map()` — No Change Needed

V4 is natively in transformers 4.57.1+. No patched modeling file required. The entry goes through unchanged.

### 0.5 Block Detection Validation

`DECODER_BLOCK_PATTERNS` in `layerwise_model_utils.py:37` includes `r"\.layers\.\d+$"` which matches V4's `model.layers.N`. No change needed. Test with `find_decoder_blocks(model)`.

**Checklist:**
- [ ] 0.1 Add `MODEL_ATTRS` entry for `DeepseekV4ForCausalLM` with placeholder `fused` value
- [ ] 0.2 Create `_is_v4_model()` helper (class-name check) used by observers + prune.py
- [ ] 0.3 Add `DeepseekV4MoEObserverHookConfig` dataclass
- [ ] 0.4 Register in `OBSERVER_CONFIG_REGISTRY["DeepseekV4ForCausalLM"]`
- [ ] 0.5 Verify `find_decoder_blocks()` returns 43 layer names for Flash
- [ ] 0.6 Write unit test: `test_model_attrs_v4()` asserting correct attribute paths

---

## Phase 1: Block-From-Disk Loader

The core innovation. Instead of loading 160 GB to CPU, load one decoder layer at a time.

### 1.1 Architecture: `V4BlockDiskLoader` Class

```
src/reap/
  v4_block_loader.py     # NEW — block-from-disk loading logic
  v4_moe_observer.py     # NEW — V4-specific observer subclass
  v4_prune_utils.py      # NEW — V4 pruning utilities (expert removal, hash remap)
```

`V4BlockDiskLoader` manages the lifecycle of loading, forwarding, and unloading individual decoder layers:

```python
class V4BlockDiskLoader:
    """
    Load one DeepseekV4DecoderLayer at a time from disk.

    Strategy:
    1. Read model.safetensors.index.json to map tensor names → shard files
    2. Load non-backbone modules (embed, norm, lm_head) once — ~2 GB
    3. For each decoder layer: load weights from correct shard, construct
       the nn.Module on GPU, forward calibration batches, free.
    """
```

### 1.2 Weight Location Resolution

Flash's `model.safetensors.index.json` maps each tensor name to a shard file. Pattern:

```
model.layers.0.mlp.experts.gate_up_proj  →  model-00002-of-00046.safetensors
model.layers.0.mlp.experts.down_proj     →  model-00002-of-00046.safetensors
model.layers.0.mlp.gate.weight           →  model-00002-of-00046.safetensors
...
model.layers.1.mlp.experts.gate_up_proj  →  model-00002-of-00046.safetensors
model.layers.1.mlp.experts.down_proj     →  model-00002-of-00046.safetensors
...
model.layers.2.mlp.experts.gate_up_proj  →  model-00003-of-00046.safetensors
```

Each shard stores ~2 layers (~3.57 GB compressed, ~26 GB decompressed BF16). Decompression (FP4→BF16) happens inside `safetensors.safe_open` when using `from_pretrained`.

**Two approaches for Phase 1:**

#### Option A: Full CPU Load + Iterate Blocks (Simpler)

Load the entire model with `device_map="cpu"`, then iterate blocks by moving one to GPU at a time. This needs ~180 GB CPU RAM (160 GB compressed + overhead). Lightning's RTX PRO 6000 machine has 180 GB RAM — **barely enough but workable**.

**Current `layerwise_prune.py` already does this** (line 282-288). The existing `from_pretrained(device_map="cpu")` path works for V4 because:
- `from_pretrained` handles FP4→BF16 decompression (confirmed working)
- `device_map="cpu"` loads all weights to CPU
- The layerwise observer moves one block to GPU at a time

**However:** 160 GB compressed model on CPU + replay cache (~16 GB) + activation tensors ≈ 180+ GB. This may OOM the 180 GB Lightning machine.

#### Option B: True Block-From-Disk (Recommended for Robustness)

Load only the safetensor index, then read individual layer weights directly:

```python
class V4BlockDiskLoader:
    def __init__(self, model_path: str):
        # Load safetensors index
        with open(f"{model_path}/model.safetensors.index.json") as f:
            self.index = json.load(f)["weight_map"]
        # Cache shard file handles (lazy open)
        self._shard_cache: Dict[str, Any] = {}
        # Build mapping: layer_idx → list of tensor names in that layer
        self.layer_tensors = self._build_layer_tensor_map()

    def load_layer(self, layer_idx: int, device: str) -> nn.Module:
        """Load one decoder layer from disk to GPU."""
        # 1. Read all tensors for this layer from shard(s)
        # 2. Build a state_dict for the layer
        # 3. Create empty DeepseekV4DecoderLayer(config)
        # 4. load_state_dict(layer_state_dict)
        # 5. Move to GPU
        pass

    def unload_layer(self, layer: nn.Module):
        """Free layer from GPU."""
        layer.to("cpu")
        del layer
        cleanup_memory()
```

**Advantage:** CPU RAM stays at ~32 GB (calibration data + replay cache only). 
**Disadvantage:** Requires understanding V4's `DeepseekV4DecoderLayer.__init__` to construct empty modules.

**Recommendation:** Implement Option B as primary, with Option A as fallback. The `from_pretrained(device_map="cpu")` path already exists as a fallback if the direct safetensor loading is too complex.

### 1.3 Safetensor Shard Loading Details

Each shard file contains BF16 weights (decompressed from FP4 during `from_pretrained`). If reading raw safetensors:

```python
import safetensors

def _load_tensor_from_shard(self, tensor_name: str) -> torch.Tensor:
    shard_file = self.index[tensor_name]
    if shard_file not in self._shard_cache:
        self._shard_cache[shard_file] = safetensors.safe_open(
            f"{self.model_path}/{shard_file}",
            framework="pt",
            device="cpu",
        )
    return self._shard_cache[shard_file].get_tensor(tensor_name)
```

**Caveat:** Raw `safetensors.safe_open` reads the BF16-decompressed weights? Or still FP4 packed? Need to test. If the safetensors file stores FP4-packed I8 + F8_E8M0 scale, the `from_pretrained` path's decompression is essential, and raw safetensor reading gives raw packed bytes.

**Mitigation:** If direct safetensor loading gives packed weights, use Option A (full CPU load) for Phase 1, then optimize to block-from-disk in Phase 2.

### 1.4 Non-Backbone Modules

Modules outside decoder layers (embed, norm, lm_head) are loaded once and kept in CPU memory:

```python
# Embed + head: ~1 GB total
model.model.embed_tokens  # nn.Embedding(vocab_size=129280, dim=4096)
model.model.norm          # nn.LayerNorm(4096)
model.lm_head             # nn.Linear(4096, 129280)
```

These are loaded via `from_pretrained(device_map="cpu")` or from the first safetensor shard.

### 1.5 Replay Cache Integration

The existing `ReplayCache` in `layerwise_observer.py` stores hidden states from block N and feeds them to block N+1. With block-from-disk:

1. First pass: load embed + head, capture block-0 inputs into replay cache
2. For each decoder layer N:
   a. Load layer N weights from disk shard onto GPU
   b. Forward all cached replay batches through layer N
   c. Collect metrics via observer hooks
   d. Store layer N outputs back into replay cache (overwrites inputs)
   e. Free layer N from GPU
3. (Optional) Final pass with lm_head for verification

**Checklist:**
- [ ] 1.1 Design `V4BlockDiskLoader` class with `load_layer()` / `unload_layer()`
- [ ] 1.2 Implement `_load_tensor_from_shard()` for raw safetensor reading
- [ ] 1.3 Test: does `safetensors.safe_open` return BF16 or packed FP4 weights?
- [ ] 1.4 If packed, implement Option A fallback (full CPU load via `from_pretrained`)
- [ ] 1.5 Implement `load_non_backbone_modules()` — embed, norm, lm_head
- [ ] 1.6 Implement `_build_layer_tensor_map()` from index.json
- [ ] 1.7 Create empty `DeepseekV4DecoderLayer` from config (no weight loading needed for empty module)
- [ ] 1.8 Integration test: load layer 0, forward 1 batch, verify output shape
- [ ] 1.9 Memory benchmark: peak VRAM during single-layer forward

---

## Phase 2: Observer — DeepseekV4MoEObserver

V4 cannot use either branch of `MoETransformerObserver._hook_factory`. Must create a V4-specific observer subclass.

### 2.1 V4 Hook Factory — Registration

Create `src/reap/v4_moe_observer.py`:

```python
class DeepseekV4MoEObserver(LayerwiseMoEObserver):
    """V4-specific observer overrides for the layerwise pipeline."""

    def _find_moe_module_in_block(self, block_idx: int) -> Optional[nn.Module]:
        # Same as parent — DeepseekV4SparseMoeBlock is detected via class name
        ...

    def _process_moe_activations(
        self, block_idx, moe_module, input_hidden_states, device, attention_mask=None
    ):
        """V4 override: 3D expert params, incremental computation, sparse dispatch."""
        ...
```

### 2.2 Incremental Expert Loop (No `[E, T, D]` Tensor)

Instead of the current `activations = torch.zeros((num_experts, *flat_input.shape))`, compute one expert at a time:

```python
def _process_v4_experts(self, state, flat_input, router_logits,
                        selected_experts, moe_module, num_experts, device,
                        valid_token_mask, renormalize):
    """Compute expert activations incrementally for V4's 3D params."""
    for idx in range(num_experts):
        # Gate-up projection
        gate_up = F.linear(flat_input, moe_module.experts.gate_up_proj[idx])
        gate, up = gate_up.chunk(2, dim=-1)

        # V4-specific activation function with clamping
        # From modeling_deepseek_v4.py: act_fn(gate.clamp(max=self.limit)) * up.clamp(...)
        hidden = F.silu(gate.clamp(max=moe_module.limit)) * \
                 up.clamp(min=-moe_module.limit, max=moe_module.limit)

        # Down projection
        expert_output = F.linear(hidden, moe_module.experts.down_proj[idx])

        # Accumulate into state using single-expert variant
        self._update_state_single_expert(
            state, idx, expert_output,
            router_logits, selected_experts, valid_token_mask, renormalize
        )

        # Free intermediate tensors immediately
        del gate_up, gate, up, hidden, expert_output
```

**Memory:** `flat_input` = `[B*S, D]` ≈ 8 MB (B=4, S=2048, D=4096). Each expert's intermediate = ~8 MB. No `[256, T, D]` tensor = saves 17 GB.

### 2.3 Single-Expert State Update

Add to `src/reap/pruning_metrics.py`:

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
    """Accumulate pruning metrics for ONE expert at a time.

    Instead of building an [E, T, D] tensor, call this per-expert.
    The function computes EAN, REAP, frequency, max_activations for
    the tokens that selected this expert.
    """
    # Get routing weights
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    if renormalize_router_weights:
        topk_weights = torch.gather(routing_weights, 1, selected_experts)
        routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Find tokens that selected this expert
    active_mask = (selected_experts == expert_idx).any(dim=-1)
    if valid_token_mask is not None:
        active_mask = active_mask & valid_token_mask

    if not active_mask.any():
        return

    active_outputs = expert_output[active_mask]
    active_weights = routing_weights[active_mask, expert_idx]

    ean_norm = torch.linalg.norm(active_outputs, dim=-1)
    layer_state["ean_sum"][expert_idx] += ean_norm.sum()
    layer_state["ean_mean"]._partial_update(expert_idx, ean_norm.mean(),
                                            torch.tensor([active_outputs.size(0)]))
    layer_state["weighted_ean_sum"][expert_idx] += (ean_norm * active_weights).sum()
    layer_state["weighted_expert_frequency_sum"][expert_idx] += active_weights.sum()
    layer_state["reap"]._partial_update(expert_idx,
                                        (ean_norm * active_weights).mean(),
                                        torch.tensor([active_outputs.size(0)]))

    max_val = active_outputs.max()
    if max_val > layer_state["max_activations"][expert_idx]:
        layer_state["max_activations"][expert_idx] = max_val
```

Note: `OnlineStatsTracker` needs a `_partial_update(expert_idx, value, count)` method. Current implementation (`metrics.py`) expects full-shape updates. This requires adding a partial-update method.

### 2.4 Router Logits Extraction

**Problem:** Current code in `_process_moe_activations` (layerwise_observer.py:656-659):
```python
if isinstance(result, tuple):
    *_, router_logits = result  # Gets indices (3rd element), not logits!
```

**V4 routers return:** `DeepseekV4TopKRouter.forward(hidden_states)` → `(logits, weights, indices)`.
`DeepseekV4HashRouter.forward(hidden_states, input_ids)` → `(logits, weights, indices)`.

**Fix:** Extract logits at index 0:

```python
def _extract_v4_router_logits(self, router_module, hidden_states,
                               input_ids=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (router_logits, selected_experts) for V4."""
    if hasattr(router_module, 'is_hash') and router_module.is_hash:
        # HashRouter needs input_ids
        logits, weights, indices = router_module(hidden_states, input_ids)
    else:
        logits, weights, indices = router_module(hidden_states)
    _, selected_experts = torch.topk(logits, router_module.top_k, dim=-1)
    return logits, selected_experts
```

### 2.5 Hash Router + `input_ids` in ReplayCache

`DeepseekV4HashRouter.forward(hidden_states, input_ids)` requires `input_ids`. The current `ReplayBatch` dataclass (layerwise_observer.py:52-58) does not carry `input_ids`.

**Changes to `ReplayBatch`:**
```python
@dataclass
class ReplayBatch:
    inputs: List[torch.Tensor]
    kwargs: Dict[str, Any]
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None  # NEW: needed for hash router
```

**Changes to `intercept_entry_inputs`:** Capture `input_ids` from the model forward call's kwargs alongside hidden states.

**Changes to `_process_moe_activations`:** For the first `num_hash_layers` blocks, pass `input_ids` from `ReplayBatch` to `_extract_v4_router_logits`.

**Layer detection:** Use `config.mlp_layer_types[layer_idx] == "hash_moe"` (from model config, 3 layers for Flash).

### 2.6 Standard Observer Path — Separate Fix

The standard (non-layerwise) observer `MoETransformerObserver._hook_factory` in `observer.py` has the same two broken branches. The fix:

- Register a forward hook on `DeepseekV4TopKRouter` / `DeepseekV4HashRouter` directly (submodule of `DeepseekV4SparseMoeBlock`) instead of on the MoE block
- This avoids the single-tensor output issue (V4's `SparseMoeBlock.forward` returns a single tensor, not a tuple)
- The hook captures raw logits before the scoring function applies

**Implementation:** Create a `register_router_hooks` method that hooks `module.gate` (the router submodule) when the MoE block is `DeepseekV4SparseMoeBlock`. This side-steps the `len(output) >= 2` assertion on line 331.

### 2.7 mHC-Aware Replay Cache

V4 hidden states are `[B, S, 4, D]` (hc_mult=4 parallel streams). The replay cache currently stores `[B, S, D]`.

**No change needed for the cache data structure** — `ReplayBatch.inputs` is a `List[torch.Tensor]`, which handles arbitrary shapes. The mHC collapse to `[B, S, D]` happens inside `DeepseekV4DecoderLayer.forward` before the MoE block.

**Memory consideration:** Each cached `[B, S, 4, D]` tensor = 4× larger = 256 MB per batch (B=4, S=2048, D=4096). With 64 batches = 16 GB cache. The `batch_group_size` parameter in `LayerwiseArgs` controls this.

**Checklist:**
- [ ] 2.1 Create `v4_moe_observer.py` with `DeepseekV4MoEObserver` class
- [ ] 2.2 Implement incremental expert loop (no `[E, T, D]` tensor)
- [ ] 2.3 Add `update_pruning_state_single_expert()` to `pruning_metrics.py`
- [ ] 2.4 Add `_partial_update()` to `OnlineStatsTracker` in `metrics.py`
- [ ] 2.5 Implement `_extract_v4_router_logits()` with TopKRouter/HashRouter branching
- [ ] 2.6 Add `input_ids` field to `ReplayBatch` dataclass
- [ ] 2.7 Update `intercept_entry_inputs` to capture `input_ids`
- [ ] 2.8 Update `_process_moe_activations` to pass `input_ids` for hash layers
- [ ] 2.9 Implement standard observer hook for `DeepseekV4TopKRouter`/`DeepseekV4HashRouter`
- [ ] 2.10 Test: forward 4-layer V4-tiny with observer, verify metrics shape
- [ ] 2.11 Benchmark: per-expert `F.linear` time vs full batched `bmm` (expected ~50-100ms per expert)

---

## Phase 3: Pruning — V4-Specific

### 3.1 Guard Against Non-V4 Paths

In `src/reap/prune.py:prune()`, add V4 interception BEFORE the `fused` flag check:

```python
def prune(observer_data, model, prune_args, n_experts_to_prune, pruned_model_dir):
    model_attrs = MODEL_ATTRS[model.__class__.__name__]

    # ... (super expert preservation, pruning method selection) ...

    for layer in tqdm(observer_data, "Pruning layers..."):
        # ... (expert selection logic) ...

        retained_expert_indices = [...]

        moe = get_moe(model, layer)

        # --- V4 INTERCEPTION ---
        if "DeepseekV4" in model.__class__.__name__:
            _prune_v4_layer(moe, retained_expert_indices, model, layer)
        elif not model_attrs["fused"]:
            # ... existing ModuleList path ...
        else:
            # ... existing fused path ...

    # --- Config update ---
    if "DeepseekV4" in model.__class__.__name__:
        model.config.n_routed_experts = len(retained_expert_indices)
        model.config.num_local_experts = len(retained_expert_indices)
    else:
        setattr(model.config, model_attrs["num_experts"], retained_experts)
```

### 3.2 V4 Expert Removal

Create `src/reap/v4_prune_utils.py`:

```python
def _prune_v4_layer(moe, retained_indices, model, layer_idx):
    """Prune V4 experts by indexing 3D weight tensors along dim 0."""
    # Expert weights: 3D parameter tensors [N, dim, dim]
    moe.experts.gate_up_proj = nn.Parameter(
        moe.experts.gate_up_proj.data[retained_indices].clone()
    )
    moe.experts.down_proj = nn.Parameter(
        moe.experts.down_proj.data[retained_indices].clone()
    )
    moe.experts.num_experts = len(retained_indices)

    # Router: access via `gate`, not `router`
    gate = moe.gate
    gate.weight = nn.Parameter(
        gate.weight.data[retained_indices].clone()
    )
    gate.out_features = len(retained_indices)

    # e_score_correction_bias — only on TopKRouter
    if hasattr(gate, "e_score_correction_bias"):
        gate.e_score_correction_bias = nn.Parameter(
            gate.e_score_correction_bias.data[retained_indices].clone()
        )
    if hasattr(gate, "num_experts"):
        gate.num_experts = len(retained_indices)

    # Shared experts — NEVER prune
    # moe.shared_experts is a DeepseekV4MLP — left untouched
```

### 3.3 Hash Router `tid2eid` Remapping

After pruning hash-routed layers (first 3), the `tid2eid` buffer (shape `[vocab_size, top_k]`) contains old expert indices. Must remap:

```python
def _remap_hash_router_tid2eid(gate, old_to_new):
    """Remap tid2eid lookup table after expert pruning.

    old_to_new[old_idx] = new_idx for retained experts, else -1 for pruned.

    tid2eid is a registered buffer, not a regular tensor. Must use
    register_buffer() or .data.copy_() to preserve buffer registration.
    """
    device = gate.tid2eid.device
    old_to_new = torch.tensor(old_to_new, device=device, dtype=torch.long)

    tid2eid = gate.tid2eid.clone()
    mask = tid2eid >= 0  # some entries may be -1 (unused TID)

    # Remap: pruned experts → 0 (best effort fallback)
    remapped = old_to_new[tid2eid.clamp(min=0)]
    remapped[~mask] = 0

    # Use data.copy_ to preserve buffer registration
    gate.tid2eid.data.copy_(remapped)
```

**Layer detection:** Use `model.config.mlp_layer_types[layer_idx]` rather than hardcoding layers 0-2.

### 3.4 `e_score_correction_bias` Guard

`e_score_correction_bias` exists on `TopKRouter` but NOT on `HashRouter`. Always guard:

```python
if hasattr(gate, "e_score_correction_bias"):
    gate.e_score_correction_bias = nn.Parameter(...)
```

### 3.5 Config Update

V4 config uses both `n_routed_experts` and `num_local_experts` (they're aliased). After pruning all layers:

```python
model.config.n_routed_experts = num_retained
model.config.num_local_experts = num_retained
```

### 3.6 Pruning Step: Model Reload

After the observer collects metrics, the pruning step loads the full model. For V4 Flash (160 GB), this must also be done layerwise or via `device_map="auto"` with CPU offloading.

**Current flow** (layerwise_prune.py:357-369): Reload model with `device_map="auto"` for pruning. This expects to fit the model on GPU. For V4 Flash, this fails.

**Solution:** Use `from_pretrained(device_map="cpu")` for the pruning step too. Move the pruned weights to CPU and save from there. The `model.save_pretrained(pruned_model_dir)` already handles CPU saving.

**Alternative:** Implement a `V4PruningLoader` that:
1. Loads the model via `device_map="cpu"`
2. Prunes retained indices in-place (CPU operations)
3. Saves to disk

This avoids the ~160 GB GPU load entirely.

**Checklist:**
- [ ] 3.1 Add V4 interception branch in `prune.py:prune()` before `fused` check
- [ ] 3.2 Implement `_prune_v4_layer()` — 3D tensor indexing
- [ ] 3.3 Implement `_remap_hash_router_tid2eid()` with `data.copy_()`
- [ ] 3.4 Add `hasattr` guards for `e_score_correction_bias`
- [ ] 3.5 Update `model.config.n_routed_experts` and `num_local_experts`
- [ ] 3.6 Implement pruning model loading strategy (CPU or layerwise)
- [ ] 3.7 Test: load V4-tiny, prune 25% experts, save, reload, verify shape
- [ ] 3.8 Verify `shared_experts` unchanged after pruning
- [ ] 3.9 Verify MTP head handling (`_keys_to_ignore_on_load_unexpected`)

---

## Phase 4: Memory Budget Verification

### 4.1 Per-Layer VRAM Budget (Flash, BF16)

| Component | Size | Notes |
|-----------|------|-------|
| Expert weights (256 × gate_up_proj + down_proj) | ~12.8 GB | 256 × (2×2048×4096 + 4096×2048) × 2 B |
| Gate router weight | ~2 MB | 256 × 4096 × 2 B |
| Shared expert MLP | ~134 MB | 4096×8192 + 8192×4096 |
| Attention components | ~260 MB | Q, K, V, O projections |
| mHC params | ~8 MB | Negligible |
| Activations (flat_input) | ~8 MB | 4×2048×4096 × 2 B (per expert: same) |
| Expert output | ~8 MB | Per expert, freed immediately |
| Metrics state | ~2 MB | Per-expert scalars |
| Forward intermediates | ~2 GB | LayerNorm output, attention states, etc. |
| **Total per layer** | **~15.2 GB** | Fits easily in 96 GB |

### 4.2 Peak VRAM with Block-From-Disk

| Component | Size |
|-----------|------|
| Current decoder layer | ~15.2 GB |
| Non-backbone (embed + norm + head) | ~2 GB (on CPU) |
| Replay batch materialized | ~256 MB |
| **Peak GPU** | **~15.5 GB** |

96 GB available → ~80 GB headroom. No issue for Flash.

### 4.3 CPU RAM Budget

| Component | Size |
|-----------|------|
| Calibration data (64 batches × 4 × 2048 × 4096) | ~4 GB |
| Replay cache (64 batches × 256 MB) | ~16 GB |
| Non-backbone modules | ~2 GB |
| Other (tokenizer, Python overhead) | ~2 GB |
| **Total (block-from-disk)** | **~24 GB** |
| Full model CPU load (fallback) | ~160 GB (compressed) |

With block-from-disk: 24 GB fits in any machine. Without: 160+ GB needs the high-RAM instance.

### 4.4 Time Budget

| Phase | Operations | Time |
|-------|-----------|------|
| Per layer: load from shard | Read ~3.57 GB from shard, decompress FP4→BF16 | ~3 s (NVMe) |
| Per layer: forward 64 batches | 64 × (single expert loop: 256 × 3ms) = 64 × 0.77s | ~49 s |
| Per layer: unload | Free + GC | ~1 s |
| Per layer total | | ~53 s |
| **All 43 layers** | 43 × 53 s | **~38 min** |
| Pruning step | Load full model, prune, save | ~15 min |
| **Total** | | **~53 min** |

### 4.5 V4 Pro (1.6T) — Future Consideration

Pro's per-layer is ~52 GB (BF16) weights + ~5-7 GB overhead = ~57-59 GB. This fits the RTX PRO 6000 (96 GB) but requires the incremental expert approach (no `[E, T, D]` tensor which would add ~45 GB). Per-layer time will be higher: 384 experts × 2 matmuls each × D=7168.

**Conclusion:** Flash is safe. Pro needs careful benchmarking but should fit.

**Checklist:**
- [ ] 4.1 Measure per-layer VRAM with `torch.cuda.memory_allocated()`
- [ ] 4.2 Verify peak VRAM < 48 GB (target headroom for safety)
- [ ] 4.3 Measure per-layer time (load + forward + unload)
- [ ] 4.4 Measure replay cache CPU memory for batch_group_size=64
- [ ] 4.5 Verify Option A fallback (full CPU load) fits in 180 GB
- [ ] 4.6 Profile expert loop time: 256 experts × 2 F.linear calls

---

## Phase 5: End-to-End Integration

### 5.1 Entry Point

Add a new entry point or extend the existing `layerwise_prune.py`:

```bash
python -m reap.v4_prune \
    --model_name "deepseek-ai/DeepSeek-V4-Flash" \
    --dataset_name "theblackcat102/evol-codealpaca-v1" \
    --prune_method "reap" \
    --compression_ratio 0.25 \
    --batch_size 4 \
    --block_disk_loader True
```

Or extend `layerwise_prune.py` with a `--model-type v4` flag that selects the V4 observer.

### 5.2 CLI Arguments

Add to `args.py`:

```python
@dataclass
class V4BlockDiskArgs:
    block_disk_loader: bool = field(
        default=True,
        metadata={"help": "Use block-from-disk loading (load one layer at a time from safetensors)"}
    )
    shard_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to safetensor shard directory (defaults to model path)"}
    )
    num_hash_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of hash router layers at the start (auto-detected from config)"}
    )
```

### 5.3 Pipeline Integration

The updated pipeline in `layerwise_prune.py`:

```python
def main():
    # Parse args as before...

    # Model selection based on class name
    if "DeepseekV4" in model_class_name:
        if layerwise_args.block_disk_loader:
            observer_data = record_v4_activations_block_disk(...)
        else:
            observer_data = record_activations_layerwise(...)
    else:
        # Standard layerwise path
        observer_data = record_activations_layerwise(...)

    # Pruning — shared code path that branches on model class
    ...
```

### 5.4 Tokenizer

V4 uses standard `AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Flash")`. No special encoding logic needed. The `encoding/encodigng_dsv4.py` in the repo is for inference-time prompt construction, not needed for REAP calibration.

### 5.5 Evaluation

After pruning, run `model.generate()` to verify coherence. The pruned model is saved as a standard HF model; vLLM/SGLang support is confirmed from the V4 README.

### 5.6 Save Format

The pruned model directory contains:
- `model.safetensors` (pruned weights)
- `config.json` (updated `n_routed_experts`, `num_local_experts`)
- `tokenizer.json`, `tokenizer_config.json`
- `generation_config.json`
- `args.yaml` (for reproducibility)

**Checklist:**
- [ ] 5.1 Create `v4_prune.py` entry point or extend `layerwise_prune.py`
- [ ] 5.2 Add `V4BlockDiskArgs` to `args.py`
- [ ] 5.3 Route to V4 observer when model class is `DeepseekV4ForCausalLM`
- [ ] 5.4 Implement `record_v4_activations_block_disk()` function
- [ ] 5.5 Wire up pruning step (reload model on CPU, prune, save)
- [ ] 5.6 Smoke test: `model.generate()` on pruned model (1 forward pass)
- [ ] 5.7 Run full pipeline end-to-end on Flash with 10% compression
- [ ] 5.8 Compare `model.config` before/after pruning for correctness
- [ ] 5.9 Write integration test with `pytest` for V4 observer + pruning

---

## Risk Register

| # | Risk | Impact | Likelihood | Mitigation | Phase |
|---|------|--------|------------|-----------|-------|
| R1 | Raw `safetensors` reading returns packed FP4 (I8) bytes, not BF16 | Block-from-disk must implement custom FP4→BF16 decompression | **Certain** | Implement `v4_block_loader.py` with I8→FP4 unpack + F8_E8M0 dequant (~50 lines). See Phase 1.2. | 1 |
| R2 | `from_pretrained` full CPU load decompresses to BF16 → ~560 GB | Lightning's 180 GB machine cannot host full model | **Certain** | Option A (full load) is NOT viable. Custom block-from-disk with per-layer decompression is the only option. | 1 |
| R3 | `OnlineStatsTracker._partial_update` needs non-trivial refactor | Incremental expert update blocked | Medium | Replace `OnlineStatsTracker` with direct tensor accumulation for V4-specific path | 2 |
| R4 | Hash router `input_ids` not available in hook closure | Hash layers fail with TypeError | Low | Propagate `input_ids` through `ReplayBatch` (2.6). Test with first 3 layers. | 2 |
| R5 | Pruned model with remapped `tid2eid` produces different outputs | Output quality degrades on first 3 layers | Medium | Only 3/43 hash layers; quantization noise likely dominates. Accept for Phase 1. | 3 |
| R6 | MTP head tensors cause `save_pretrained` warnings | Unnecessary files in output | Low | Already ignored by `_keys_to_ignore_on_load_unexpected`; harmless | 3 |
| R7 | `e_score_correction_bias` access on HashRouter | AttributeError crash | Low | `hasattr` guard (3.4) | 3 |
| R8 | Per-expert `F.linear` loop is too slow (>3 min/layer) | Pipeline takes 2+ hours | Medium | Benchmark early (2.11). If slow, batch experts in groups of 8-16. | 2 |
| R9 | Config field change: `n_routed_experts` vs `num_local_experts` mismatch | Config not updated after pruning | Low | Update both fields (3.5) | 3 |
| R10 | FP4→BF16 decompression fails for specific attention tensors | Load hangs or produces NaN | Low | Test `from_pretrained` on a single layer in Phase 0 | 0 |
| R11 | `input_ids` never reaches `DeepseekV4DecoderLayer.forward` as kwarg | Hash router observation fails silently (TypeError on `None`) | **Critical** | Check V4 HF modeling file before Phase 1. If absent, redesign: hook inside `DeepseekV4Model.forward` or capture `input_ids` at model entry point | 1 |
| R12 | Per-expert `F.linear` kernel launch overhead dominates runtime | 700K+ launches add 30-60s of pure driver overhead | Medium | Benchmark in Phase 2.1; batch experts in groups of 8-16 if overhead >10% of total time | 2 |
| R13 | Pruning step reloads model from disk (duplicate load) | 5-10 min of wasted I/O per run | Medium | Keep CPU-loaded model in memory between observation and pruning; skip the delete+reload pattern | 3 |

---

## File Change Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/reap/model_util.py` | Edit | Add `DeepseekV4ForCausalLM` to `MODEL_ATTRS` |
| `src/reap/observer.py` | Edit | Add `DeepseekV4MoEObserverHookConfig`, register in registry |
| `src/reap/v4_block_loader.py` | **NEW** | Block-from-disk safetensor loader |
| `src/reap/v4_moe_observer.py` | **NEW** | V4-specific observer subclass |
| `src/reap/v4_prune_utils.py` | **NEW** | V4 pruning helpers |
| `src/reap/pruning_metrics.py` | Edit | Add `update_pruning_state_single_expert()` |
| `src/reap/metrics.py` | Edit | Add `OnlineStatsTracker._partial_update()` |
| `src/reap/layerwise_observer.py` | Edit | Add `input_ids` to `ReplayBatch`, hash router branching |
| `src/reap/layerwise_model_utils.py` | Edit (maybe) | Add V4-layer detection if needed |
| `src/reap/prune.py` | Edit | Add V4 interception before `fused` branch |
| `src/reap/args.py` | Edit | Add `V4BlockDiskArgs` if creating separate entry point |
| `src/reap/v4_prune.py` | **NEW** | Separate entry point (or extend `layerwise_prune.py`) |
| `tests/` | Edit | Add V4 integration tests |

---

## Architecture Diagram

```
layerwise_prune.py / v4_prune.py
    │
    ├──→ OBSERVER_CONFIG_REGISTRY["DeepseekV4ForCausalLM"]
    │       └── DeepseekV4MoEObserverHookConfig
    │
    ├──→ V4BlockDiskLoader
    │       ├── model.safetensors.index.json → tensor→shard map
    │       ├── load_layer(N) → state_dict → nn.Module on GPU
    │       └── unload_layer()
    │
    ├──→ DeepseekV4MoEObserver (extends LayerwiseMoEObserver)
    │       ├── _process_moe_activations()  [overridden]
    │       │   └── _process_v4_experts() [incremental, single-expert]
    │       ├── _extract_v4_router_logits() [TopKRouter + HashRouter]
    │       └── ReplayCache [with input_ids field]
    │
    └──→ prune()
            └── _prune_v4_layer() [before fused branch]
                    ├── 3D weight tensor indexing
                    ├── hash router tid2eid remap
                    └── config update
```

---

## Quick Start (for implementer)

```bash
# 1. Verify model loads (Phase 0)
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'deepseek-ai/DeepSeek-V4-Flash',
    device_map='cpu',
    torch_dtype='auto',
    trust_remote_code=True
)
print(model.__class__.__name__)  # DeepseekV4ForCausalLM
print(len(model.model.layers))   # 43
print(type(model.model.layers[0].mlp))  # DeepseekV4SparseMoeBlock
"

# 2. Verify safetensors index (Phase 1)
python -c "
import json
with open('path/to/model.safetensors.index.json') as f:
    index = json.load(f)
# Check tensor→shard mapping for layer 0
layer0_tensors = [k for k in index['weight_map'] if k.startswith('model.layers.0.')]
print(f'Layer 0: {len(layer0_tensors)} tensors')
for t in layer0_tensors:
    print(f'  {t} → {index[\"weight_map\"][t]}')
"

# 3. Run full pipeline (10% compression test)
python -m reap.v4_prune \
    --model_name "deepseek-ai/DeepSeek-V4-Flash" \
    --dataset_name "theblackcat102/evol-codealpaca-v1" \
    --prune_method "reap" \
    --compression_ratio 0.10 \
    --batch_size 2 \
    --batch_group_size 16
```

---

## References

- Design spec: `docs/superpowers/specs/2026-06-27-deepseek-v4-reap-support-design.md`
- V4 modeling: `transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py`
- Flash config: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json`
- Existing DeepSeek V2 support: `src/reap/models/modeling_deepseek.py`
- `fused` expert path (Llama4): `src/reap/observer.py:353` (standard), `layerwise_observer.py:662` (layerwise)
- Non-fused expert path (Qwen3, Mixtral, DeepSeek V2): `observer.py:374`, `layerwise_observer.py:681`
- Pruning branching: `prune.py:110`
- Block detection: `layerwise_model_utils.py:37`
