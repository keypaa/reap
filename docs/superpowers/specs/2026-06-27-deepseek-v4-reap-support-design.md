# DeepSeek V4 — REAP Pipeline Adaptation Design

**Date:** 2026-06-27
**Status:** Draft
**Target:** DeepSeek-V4-Flash (284B) first, then DeepSeek-V4-Pro (1.6T)

## 1. Motivation

DeepSeek released V4-Pro (1.6T total, 49B active) and V4-Flash (284B total, 13B active) on April 24, 2026 under MIT license. Both models introduce architectural innovations that break every assumption in the current REAP pruning pipeline. This document maps all breakages and designs the adaptations needed.

## 2. DeepSeek V4 Architecture — What Changed

### 2.1 Overview

| Aspect | DeepSeek V3/V2 | DeepSeek V4 |
|---------|---------------|-------------|
| Attention | Multi-head Latent Attention (MLA) | Hybrid CSA + HCA (3 layer types) |
| Residual | Standard x + sublayer(x) | mHC: 4 parallel streams + learned collapse |
| MoE experts | ModuleList of FFN modules | 3D param tensor [N, d, d] |
| Router | Learned top-K gate | Learned (TopKRouter) or static hash (HashRouter) |
| Shared expert | Per-model | Per-block (dense MLP always active) |
| Weight format | BF16/FP8 | Mixed FP4 + FP8 (no BF16 weights published) |
| Model type | `DeepseekV2ForCausalLM` | `DeepseekV4ForCausalLM` |

### 2.2 Config Comparison

| Config Field | Flash (284B) | Pro (1.6T) |
|---|---|---|
| `hidden_size` | 4096 | 7168 |
| `num_hidden_layers` | 43 | 61 |
| `num_attention_heads` | 64 | 128 |
| `head_dim` | 512 | 512 |
| `n_routed_experts` | 256 | 384 |
| `n_shared_experts` | 1 | 1 |
| `num_experts_per_tok` | 6 | 6 |
| `moe_intermediate_size` | 2048 | 3072 |
| `num_hash_layers` | 3 | 3 |
| `hc_mult` | 4 | 4 |
| `q_lora_rank` | 1024 | 1536 |
| `o_groups` | 8 | 16 |
| `o_lora_rank` | 1024 | 1024 |
| `topk_method` | `noaux_tc` | `noaux_tc` |
| `scoring_func` | `sqrtsoftplus` | `sqrtsoftplus` |
| `routed_scaling_factor` | 1.5 | 2.5 |
| Weight size (FP4+FP8) | ~160 GB | ~862 GB |
| `num_key_value_heads` | 1 | 1 |

### 2.3 Dimensioning Per-Layer Memory

**Flash (284B) per decoder layer:**
- Attention: q projections (q_a: 4096×1024, q_b: 1024×32768), kv_proj (4096×512), o projections (grouped) ≈ 130M params → 260 MB (BF16)
- MoE gate: 256×4096 ≈ 1M params → 2 MB
- MoE experts: 256 × (gate_up_proj: 2×2048×4096 + down_proj: 4096×2048) = 256 × 25.2M ≈ 6.4B params → **12.8 GB (BF16)**
- Shared expert MLP: 4096×8192 + 8192×4096 ≈ 67M params → 134 MB
- mHC: (2+4)×4 × (4×4096) ≈ 0.4M params → negligible
- Norms, etc: negligible
- **Total weights per layer: ~13 GB (BF16)**

**Pro (1.6T) per decoder layer:**
- MoE experts: 384 × (gate_up_proj: 2×3072×7168 + down_proj: 7168×3072) = 384 × 66.1M ≈ 25.4B params → **50.8 GB (BF16)**
- Other components scale similarly
- **Total weights per layer: ~52 GB (BF16)** — exceeds all current single-GPU VRAM

**Per-layer BF16 memory (Pro):** ~52 GB weights + ~5-7 GB incremental compute overhead = **~57-59 GB total**. Fits comfortably on A100 80GB and RTX PRO 6000 96GB with the incremental-expert approach. The full-activations-tensor approach would add ~45 GB (384 experts × 8192 tokens × 7168 hidden) and overflow — so incremental experts are mandatory for Pro.

### 2.4 Expert Storage (`DeepseekV4Experts`)

```python
class DeepseekV4Experts(nn.Module):
    def __init__(self, config):
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_dim, hidden_dim)
        )  # [N, 2*D_ff, D]
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_dim)
        )  # [N, D, D_ff]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Iterates over hit experts only (sparse dispatch)
        for expert_idx in hit:
            current = F.linear(hidden_states[token_idx],
                               self.gate_up_proj[expert_idx])
            current = self._apply_gate(current)
            current = F.linear(current, self.down_proj[expert_idx])
            final.index_add_(0, token_idx, current * weights[...])
```

**Key:** This is a sparse expert implementation. Not all experts are computed per token — only the top-K selected ones. For REAP, we need ALL experts' activations for saliency computation. This requires a dense loop over all `N` experts, which the current code already does (`activations[idx] = expert(flat_input)`).

**Memory optimization:** Instead of `[N, T, D]` tensor (17 GB for Flash), compute one expert at a time and accumulate into `update_pruning_state` incrementally.

### 2.5 Router: Two Types

**`DeepseekV4TopKRouter`** (layers 3+):
```
forward(hidden_states) → (logits, weights, indices)
  logits = F.linear(flat, weight)          # raw scores
  scores = score_fn(logits)                 # sqrtsoftplus
  indices = topk(scores + e_score_correction_bias, top_k)
  weights = softmax(scores.gather(1, indices))
  return logits, weights * routed_scaling_factor, indices
```

**`DeepseekV4HashRouter`** (layers 0-2):
```
forward(hidden_states, input_ids) → (logits, weights, indices)
  indices = tid2eid[input_ids]              # frozen lookup
  logits = F.linear(flat, weight)
  weights = scores.gather(1, indices)
  return logits, weights * routed_scaling_factor, indices
```

**Implications:**
- Hash routers need `input_ids` passed through the replay cache
- Hash routers have frozen expert selection — expert removal requires remapping `tid2eid` table
- TopK routers are prunable by row-indexing `weight` and `e_score_correction_bias`

### 2.6 mHC Residual (all layers)

```python
# Inside DeepseekV4DecoderLayer.forward:
# hidden_states shape: [B, S, hc_mult, D]  (hc_mult = 4)
post, comb, collapsed = attn_hc(hidden_states)
attn_output = self_attn(input_layernorm(collapsed), **kwargs)
hidden_states = post.unsqueeze(-1) * attn_output.unsqueeze(-2) \
                + torch.matmul(comb.transpose(-1,-2), hidden_states)

post, comb, collapsed = ffn_hc(hidden_states)
mlp_output = mlp(post_attention_layernorm(collapsed))
return post.unsqueeze(-1) * mlp_output.unsqueeze(-2) \
       + torch.matmul(comb.transpose(-1,-2), hidden_states)
```

**Implications:**
- Hidden states are always `[B, S, 4, D]`, never `[B, S, D]`
- The `collapsed` tensor (before attention/MLP) IS `[B, S, D]` — this is where we hook for the MoE input
- Replay cache must store and replay `[B, S, 4, D]` tensors (4× memory)
- The mHC collapse happens inside the decoder layer, so we can't simply tap `layer.input` for metric collection

### 2.7 Block Detection

`model.layers[i]` maps to `DeepseekV4DecoderLayer` — this works with the existing `find_decoder_blocks` patterns since V4 uses `.layers.\d+` naming.

## 3. Pipeline Adaptation Design

### 3.1 Overall Strategy: Block-from-Disk with Incremental Experts

**Why not full CPU load:**
- V4 Flash: ~160 GB FP4+FP8 weights → needs custom CPU-side dequantization
- V4 Pro: ~862 GB — impractical to hold in any single machine's RAM
- Neither model has BF16 weights published

**Why not API-based:**
- Cannot measure expert activations (need model-internal router logits + EAN norms)
- Cannot prune (need weight modification)

**Block-from-disk approach:**
1. Load safetensors index to know which shard contains which block's weights
2. Load only non-backbone modules (embed, norm, lm_head) once — ~2 GB
3. For each decoder layer:
   a. Load its weights from the correct safetensor shard(s)
   b. Dequantize FP8/FP4 → BF16 on GPU
   c. Construct the block's `nn.Module` with loaded weights
   d. Forward calibration batches through it, collect metrics
   e. Free block, move to next
4. Save accumulated metrics to disk
5. Reload full model (or reconstruct pruned model) for pruning step

### 3.2 FP8/FP4 Loading and Dequantization

**Challenge:** The published weights are FP4+FP8 mixed (no BF16 version). PyTorch `from_pretrained` loads them at their native precision and only converts to BF16 when used in FP32/BF16 matmuls.

**Solution for block-from-disk:**
- Read raw safetensor bytes for each tensor
- Parse FP8 (E4M3), FP4 (NVFP4), and associated scale factors from the quantization metadata
- Transfer to GPU, dequantize to BF16
- Alternative: Use HuggingFace's built-in FP8 loader (`quantization_config`) which handles this automatically when loading partial weights

Actually, the simplest path: **load the model once normally with `device_map="cpu"` and let HuggingFace handle FP8/FP4 dequantization**. The `quantization_config` in the model config tells transformers how to handle the mixed-precision weights. When loaded on CPU, they stay compressed. When a tensor is moved to GPU, it gets dequantized.

But this still requires ~160 GB CPU RAM. For the block-from-disk approach, we'd need a custom safetensor loader.

**Compromise approach (recommended):**
- Use HFCache + `device_map="sequential"` or custom weight loading
- Load the model onto CPU with FP8/FP4 compressed (~160 GB for Flash) using a machine with 192-256 GB RAM
- The current layerwise pipeline then moves one block to GPU at a time
- This avoids writing a custom FP8/FP4 dequantizer

**For Pro (1.6T, ~862 GB):**
- Still need block-from-disk approach
- Must parse safetensors and dequantize FP4+FP8 → BF16 per-block
- One Pro block fits in 96 GB after dequantization, so load+dequantize+process+free per block

### 3.3 Metric Collection Changes

#### 3.3.1 Hidden State Shape

The observer needs to hook into the MoE block at the point where `collapsed` hidden states (`[B, S, D]`) enter `DeepseekV4SparseMoeBlock.forward`.

**Hook point:** `DeepseekV4SparseMoeBlock.forward` — input is the collapsed hidden states from `post_attention_layernorm(collapsed)`, which is `[B, S, D]`.

No change needed to the hooking mechanism — just register the forward hook on the MoE module itself, which already receives `[B, S, D]` input.

#### 3.3.2 Expert Activation Computation (Incremental)

**Current code** (pruning_metrics.py / layerwise_observer.py):
```python
activations = torch.zeros((num_experts, *flat_input.shape), device=device)
for idx, expert in enumerate(moe_module.experts):
    activations[idx] = expert(flat_input).to(device)
update_pruning_state(state, activations=activations, ...)
```

**V4 adaptation** (3D weights, incremental):
```python
# For V4: experts is DeepseekV4Experts with 3D params
def process_v4_experts_incremental(state, flat_input, router_logits,
                                    selected_experts, moe_module, device):
    for idx in range(num_experts):
        # Compute this expert's activation
        gate_up = F.linear(flat_input, moe_module.experts.gate_up_proj[idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = act_fn(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        expert_output = F.linear(hidden, moe_module.experts.down_proj[idx])

        # Accumulate into state (single-expert variant)
        _update_pruning_state_single_expert(
            state, idx, expert_output,
            router_logits, selected_experts, valid_token_mask
        )

        del gate_up, gate, up, hidden, expert_output
```

**New function needed:** `update_pruning_state_single_expert()` that handles one expert at a time instead of the full batch.

#### 3.3.3 Router Logits Extraction

**Current code** (observer.py):
```python
*_, router_logits = output  # assumes last element of output tuple
```

**V4 adaptation:**
```python
# V4 TopKRouter returns (logits, weights, indices)
# logits is the raw pre-score-fn logits
logits = module.gate(hidden_states)[0]  # index 0 = raw logits
_, selected_experts = torch.topk(logits, top_k, dim=-1)
```

This already works with the layerwise observer's `extract_router_logits` pattern, just need to handle the triple return instead of tuple-unpacking the last element.

#### 3.3.4 Hash Router Handling

For hash-routed layers (first 3):
- Router selection is deterministic (frozen `tid2eid` lookup)
- Still produces learned `logits` and `weights` for weighting
- Metric collection works the same — we read router logits and compute expert frequencies
- Exception: if `num_hash_layers` > 0 and the hash routers always pick certain experts, it affects pruning decisions (experts never selected by hash routing can still be pruned)

#### 3.3.5 Shared Expert Exclusion

The `shared_experts` (dense MLP) is always active and must never be pruned. Ensure that:
- `num_experts` for metric collection only counts routed experts
- Pruning logic never indexes into `shared_experts`

### 3.4 Pruning Changes

#### 3.4.1 Expert Weight Removal

V4 stores expert weights as 3D parameters. Pruning = indexing the first dimension:

```python
def prune_v4_experts(moe_module, retained_indices):
    # 3D weight tensors: [N, dim, dim]
    moe_module.experts.gate_up_proj = nn.Parameter(
        moe_module.experts.gate_up_proj.data[retained_indices]
    )
    moe_module.experts.down_proj = nn.Parameter(
        moe_module.experts.down_proj.data[retained_indices]
    )
    # Router weights
    moe_module.gate.weight = nn.Parameter(
        moe_module.gate.weight.data[retained_indices]
    )
    # Expert correction bias
    moe_module.gate.e_score_correction_bias = \
        moe_module.gate.e_score_correction_bias.data[retained_indices]

    # Config update
    moe_module.experts.num_experts = len(retained_indices)
    moe_module.gate.num_experts = len(retained_indices)
```

#### 3.4.2 Hash Router Remapping

After pruning, hash routers' `tid2eid` table (shape `[vocab_size, top_k]`) contains old expert indices. Must remap:

```python
def remap_hash_router(gate, old_to_new):
    # old_to_new[old_idx] = new_idx or -1 if pruned
    tid2eid = gate.tid2eid.clone()
    mask = tid2eid >= 0
    remapped = old_to_new[tid2eid.clamp(min=0)]
    # Clamp pruned experts to 0 (best effort; they won't be selected)
    remapped[~mask] = 0
    gate.tid2eid = remapped
```

#### 3.4.3 Shared Expert Preservation

`moe_module.shared_experts` (type `DeepseekV4MLP`) is never modified during pruning.

### 3.5 Replay Cache Changes

The replay cache stores `[B, S, D]` hidden states. For V4, it must store `[B, S, 4, D]` (mHC streams):

```python
# Current cache (1 stream)
replay_inputs: List[torch.Tensor]  # each [B, S, D]

# V4 cache (4 streams)
replay_inputs: List[torch.Tensor]  # each [B, S, 4, D]
```

Memory impact: 4× larger replay cache. Still manageable:
- B=4, S=2048, D=4096, hc_mult=4: 4 × 2048 × 4 × 4096 × 2 bytes = 256 MB per batch
- 64 batches: 16 GB cache

### 3.6 Block Detection and Hooking

**Block detection** — add `DeepseekV4DecoderLayer` to decoder block patterns:
```python
# Already matches: model.layers[i] style
```

**MoE module discovery** — register `DeepseekV4SparseMoeBlock` class in observer config:
```python
OBSERVER_CONFIG_REGISTRY["DeepseekV4ForCausalLM"] = DeepseekV4MoEObserverHookConfig(
    module_class_name_to_hook_regex="DeepseekV4SparseMoeBlock",
    num_experts_attr_name="experts.num_experts",  # or "gate.num_experts"
    top_k_attr_name="gate.top_k",
    fused_experts=False,  # NOT fused — uses 3D params, not a single module
)
```

## 4. Hardware Requirements

### 4.1 V4 Flash (284B)

| Scenario | GPU | VRAM | CPU RAM | $/hr (cloud) |
|---|---|---|---|---|
| Layerwise: current (full CPU load)* | 1× A100 40GB | 35 GB | 192-256 GB | $2-4 |
| Layerwise: block-from-disk | 1× RTX PRO 6000 | 20 GB | 32 GB | $0.72-1.50 |
| Layerwise: block-from-disk | 1× A100 40GB | 20 GB | 32 GB | $1-2 |
| Pruning step only | 1× A100 80GB | ~70 GB | 192 GB | $2-3 |

*\*Full CPU load is simpler but needs a high-RAM instance. FP8/FP4 weights on CPU are ~160 GB, needing 192+ GB machine.*

**Recommended:** Single RTX PRO 6000 Blackwell (96 GB) or A100 40GB with block-from-disk. The RTX PRO 6000 at ~$0.72/hr is the most cost-effective.

### 4.2 V4 Pro (1.6T)

| Scenario | GPU | VRAM | CPU RAM | Notes |
|---|---|---|---|---|
| Block-from-disk (incremental experts) | 1× RTX PRO 6000 | ~59 GB | 64 GB | Fits, no FP8 tricks needed |
| Block-from-disk (incremental experts) | 1× A100 80GB | ~59 GB | 64 GB | Fits comfortably |
| Full activations tensor | 1× H100 80GB | ~100+ GB | 256 GB | Overflows; not recommended |

Pro per-layer is ~52 GB (BF16) + ~5-7 GB overhead = ~59 GB total with incremental experts. Fits a single A100 80GB or RTX PRO 6000 96GB. FP8/FP4 dequantization per-block is still useful for loading efficiency (862 GB model → don't load it all at once) but not needed for VRAM reasons.

### 4.3 Cloud Instance Recommendations

| Provider | Instance | GPU | RAM | $/hr (spot) |
|---|---|---|---|---|
| Lambda Labs | `gpu_1x_h100_pcie` | H100 PCIe 80GB | 256 GB | $1.89 |
| JarvisLabs | RTX PRO 6000 | 96 GB | 64 GB | $0.72 |
| Vast.ai | RTX PRO 6000 | 96 GB | 64-128 GB | $0.50-0.80 |
| RunPod | RTX 6000 Ada | 48 GB | 64 GB | $0.79 |
| AWS | p4d.2xlarge | 2× A100 80GB | 256 GB | $7.85 |

## 5. Implementation Plan

### Phase 1: Model Support (minimal)
- Add `DeepseekV4ForCausalLM` to `MODEL_ATTRS` registry
- Add `DeepseekV4SparseMoeBlock` to `OBSERVER_CONFIG_REGISTRY`
- Verify block detection works with `find_decoder_blocks`
- Write integration test loading the V4 tiny model

### Phase 2: Observer (metric collection)
- Implement incremental expert activation loop
- Handle `DeepseekV4TopKRouter` output format (logits at index 0)
- Handle `DeepseekV4HashRouter` on first 3 layers
- Add `input_ids` passthrough for hash routers
- Verify REAP, EAN, frequency metrics compute correctly

### Phase 3: Layerwise Pipeline
- Adapt replay cache for `[B, S, 4, D]` hidden states
- Ensure mHC collapsed states are correctly captured for MoE input
- Implement block-from-disk weight loading (or use full CPU fallback)
- Add FP8/FP4 → BF16 dequantization handling
- Memory budget: 96 GB GPU / 32 GB CPU target

### Phase 4: Pruning
- Implement 3D expert pruning (weight tensor indexing)
- Implement hash router `tid2eid` remapping
- Preserve `shared_experts` during pruning
- Update config patching (`num_local_experts`)
- E2E test: prune 25% of Flash experts, verify model loads

### Phase 5: Validation
- Run `model.generate()` on pruned model
- Compare output quality vs unpruned at same compression ratio
- Verify no NaN weights after pruning
- Document memory profiling for each phase

## 6. Risk Matrix

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| FP8/FP4 dequantization on custom loader is complex | High | Medium | Use transformers' FP8 loader with `device_map="cpu"` as fallback |
| mHC collapses inside decoder layer — can't easily tap MoE input | Medium | Low | Hook `DeepseekV4SparseMoeBlock.forward` directly (it receives collapsed states) |
| Hash router expert remapping after pruning is lossy | Medium | Medium | Only 3/43 hash layers; remap via `old_to_new` lookup; acceptable for small compression ratios |
| Per-layer expert loop (256 or 384 experts) is slow | Medium | Low | Each expert is just two small matmuls; 256× ~3ms = ~0.8s per batch per layer |
| V4 Pro per-layer incremental compute still needs 57-59 GB | Low | Low | Confirmed fits on A100 80GB and RTX PRO 6000 96GB; raise `batch_group_size` as safety valve |
| Bug in weight indexing leads to silent model corruption | Critical | Low | Always compare pruned model's `generate()` output against unpruned before deployment |

## 7. Open Questions

1. **MTP (Multi-Token Prediction) head:** Present in checkpoint, ignored in modeling file via `_keys_to_ignore_on_load_unexpected`. Does it affect anything? Likely not — it was a training-time auxiliary objective.
2. **`num_local_experts` vs `n_routed_experts` in config:** The model config uses `n_routed_experts`, but the Experts module uses `num_local_experts`. Confirm which is authoritative for pruning.
3. **Score function invertibility:** `sqrtsoftplus` is not commonly used. Does the raw `logits` before scoring carry the information REAP needs, or should we use the scored values? The router returns both.
4. **`routed_scaling_factor`:** Multiply routing weights by 1.5 (Flash) or 2.5 (Pro). Does this affect saliency ordering? Probably not — it's a uniform per-layer scaling.

## 8. Discussion and Audit Findings

### 8.1 Pricing Sources

The prices in §4 were estimated from general knowledge of cloud GPU providers (Lambda, Vast, RunPod, JarvisLabs). They are rough estimates and will vary by provider, region, spot vs on-demand, and commitment length. The user should provide their specific provider(s) and budget for accurate estimates.

### 8.2 The 17 GB Activation Tensor Clarification

The 17 GB figure (`[256, 8192, 4096] × 2 bytes` for Flash) assumes the naive dense approach: compute ALL 256 experts for ALL 8192 tokens and store the result. This is what the current REAP code does (`activations[idx] = expert(flat_input)` in a loop over all experts).

**Key insight:** REAP metrics only need expert outputs for tokens that *actually selected that expert*. V4 already uses sparse dispatch (`DeepseekV4Experts.forward` iterates over `hit` experts only via `index_add_`). By hooking into the existing sparse path, we:
- Compute only the ~192 tokens/expert that were actually routed (6/256 × 8192)
- Store `[~192, 4096]` per expert instead of `[8192, 4096]`
- Reduce activation memory from 17 GB to ~0.5 GB
- Reduce total FLOPs by ~42× (256/6)

So the 17 GB figure is correct for the *current code* applied naively to V4, but a properly adapted V4 observer avoids it entirely.

### 8.3 Per-Layer Time and VRAM Trade-offs

| Collection Strategy | VRAM (Flash) | Time (Flash, 64 batches × 43 layers) | Notes |
|---|---|---|---|
| Full dense `bmm` (all 256 × all 8192 tokens) | 17+ GB | ~5 min total | Current code's approach; wasteful for V4 |
| `F.linear` loop one-at-a-time | ~0.5 GB | ~25 min | Low VRAM, high kernel-launch overhead |
| **V4 sparse hook** (only routed tokens) | **~0.5 GB** | **~3 min** | Uses V4's existing `index_add_` dispatch |
| Accelerate offloading (no code changes) | ~35 GB | ~15 min (with PCIe transfers) | CPU↔GPU transfers add latency |

**Bottom line:** The sparse hook approach is strictly better than both dense and one-at-a-time — it's faster AND uses less memory by leveraging V4's existing dispatch mechanism.

Hardware strategy: a 48 GB GPU (RTX 6000 Ada) can run the full Flash observer in ~3 minutes per layer = ~2 hours total. At ~$0.79/hr on RunPod = ~$1.58 for the full observer run. A bigger GPU doesn't meaningfully speed this up since the bottleneck is the sequential block processing, not GPU compute.

### 8.4 Accelerate Offloading Explanation

`accelerate`'s `device_map="auto"` with CPU offloading works as follows:

1. `from_pretrained(..., device_map="auto")` inspects the model and decides which layers fit on GPU and which must stay on CPU
2. During forward, each parameter is moved to GPU right before its layer executes, then freed right after
3. The underlying mechanism is hooks registered on each module's forward pre/post

For V4 Flash (~160 GB compressed) on a 48 GB GPU:
- Maybe 2-3 decoder layers fit on GPU (each ~13 GB in BF16 after dequant)
- The remaining 40 layers are on CPU
- Each forward pass moves 40 layers × 13 GB = 520 GB over PCIe
- At PCIe 4.0 ×16 bandwidth (~32 GB/s): ~16 seconds per batch of transfer overhead
- With 64 batches: ~17 minutes of pure transfer time per epoch of calibration

**Pros:** Zero code changes, works out of the box
**Cons:** PCIe bandwidth becomes the bottleneck, need ~128 GB CPU RAM for the compressed weights
**Verdict:** Useful as a quick proof of concept, not for production calibration runs

### 8.5 The-Auditor Findings

The following is the full audit report from the-auditor agent, which reviewed all proposed engineering changes for correctness, completeness, and safety.

#### 8.5.1 Change 1: Observer Config & MODEL_ATTRS Registration
**Verdict: [Contradicted] — the `fused` flag is misassigned**

The proposed entry uses `fused_experts=False`. Trace what happens with each setting:

- `fused=False` → observer's loop-based path iterates `for idx, expert in enumerate(module.experts)` (observer.py:379, layerwise_observer.py:697). `DeepseekV4Experts` is **not** iterable — `enumerate()` would iterate over its `nn.Parameter` attributes (`gate_up_proj`, `down_proj`, `act_fn`, etc.), not over individual experts. This silently produces garbage activations or a runtime error.

- `fused=True` → observer's fused path accesses `moe_module.router` (observer.py:354, layerwise_observer.py:664). V4's router is at `moe_module.gate`, not `moe_module.router`. And `router_logits` extraction assumes the output contains `router_scores` at index 1, which V4's `DeepseekV4SparseMoeBlock.forward` doesn't return at all (it returns a single tensor `routed + shared_experts(residual)`).

**Neither setting works.** V4 needs a third observer path (neither fused nor loop-based `ModuleList` iteration), not a toggle between two broken ones.

Also: the design doc's suggestion for `num_experts_attr_name` is `"experts.num_experts"` or `"gate.num_experts"`. Both resolve correctly — `DeepseekV4Experts.num_experts` and `DeepseekV4TopKRouter.num_experts`/`DeepseekV4HashRouter.num_experts` all come from `config.num_local_experts`. The `MODEL_ATTRS` entry needs `"moe_block": "mlp"` for `get_moe()` (model_util.py:122), which is correct for V4's `DecoderLayer.mlp`.

**What's missing:** The `fused` field in `MODEL_ATTRS` influences prune.py's logic too (prune.py:110). With `fused=False` it tries `ModuleList` indexing; with `fused=True` it tries `moe.router`. Both wrong. The `fused` concept conflates "expert weights stored as 3D tensors" with "expert forward path is fused." These are orthogonal for V4.

#### 8.5.2 Change 2: Router Logits Extraction
**Verdict: [Verified] diagnosis, [Contradicted] in both pipeline paths**

**Layerwise path** (`layerwise_observer.py:656-659`):
```python
if isinstance(result, tuple):
    *_, router_logits = result  # <-- gets indices (3rd element), not logits
```
`DeepseekV4TopKRouter.forward` returns `(logits, weights, indices)` — `*_, x` binds `x` to the last element, which is `indices`, not `logits`.

**Standard observer path** (`observer.py:376`):
```python
*_, router_logits = output  # expects last tuple element
```
`DeepseekV4SparseMoeBlock.forward` returns a single tensor (`routed + shared_experts(residual)`). `len(output)` at observer.py:331 would raise `ValueError: Expected output ... to be a tuple of at least length 2`.

**Both paths need fixing.** The proposed Change 2 only describes the fix for the layerwise path. The standard observer path has no proposed fix at all — it cannot use a forward hook on `DeepseekV4SparseMoeBlock` because that module doesn't emit router logits in its forward output. Options:
1. Hook at the router submodule (`module.gate`) instead of the MoE block
2. Create a custom V4 observer subclass that explicitly calls `module.gate(hidden_states)` inside the hook
3. Patch V4's `forward` to return `(output, router_logits)` tuple

**Recommendation:** Option 2 is safest — register a `DeepseekV4MoEObserver` that overrides `_hook_factory` to call the router explicitly, mirroring what `_process_moe_activations` already does in the layerwise path.

#### 8.5.3 Change 3: Expert Activation Collection (Sparse Dispatch)
**Verdict: [Verified] direction, [Unverified] completeness**

The proposal to compute each expert as `F.linear(flat_input, gate_up_proj[idx])` + `F.linear(hidden, down_proj[idx])` is architecturally correct and avoids the `[N, T, D]` activation tensor.

**Missing details:**
1. **Activation function**: V4 uses a clamped SiLU (`act_fn(gate.clamp(max=self.limit)) * up.clamp(min=-self.limit, max=self.limit)` at modeling file line 1029-1030). The proposed pseudo-code (design doc §3.3.2) must include this.
2. **`update_pruning_state_single_expert()`**: Not mentioned in the 8 enumerated changes. Without it, you'd need to build a `[1, T, D]` activation tensor per expert and pass it to the existing `update_pruning_state`, which expects `[num_experts, T, D]`.
3. **Performance**: 256 experts × 2 matmuls each = ~512 matmuls per batch per layer. At 64 batches and 43 layers (Flash), that's ~1.4M matmuls. Should be benchmarked early.

#### 8.5.4 Change 4: mHC-Aware Replay Cache
**Verdict: [Verified] — mostly works by accident**

The replay cache stores whatever `hidden_states` the first decoder block receives and whatever the previous block outputs:
- `DeepseekV4Model.forward` produces `hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)` → `[B, S, 4, D]`.
- `intercept_entry_inputs` captures this as `args[0]` — correct.
- Block output is also `[B, S, 4, D]` — correct.
- `replace_inputs` stores next input as previous output — correct.

**Memory impact**: 256 MB per batch × 64 batches = 16 GB on CPU. The layerwise pipeline's peak CPU memory for the replay cache with `batch_group_size=64` is 16 GB, not negligible.

**Verified issue**: The `_sanitize_cached_block_kwargs` drops `past_key_value`/`past_key_values` (layerwise_observer.py:146), which is correct for V4 since the cache is rebuilt per forward call. However, this means the compressor cache state is **lost** between replay batches — each batch gets a fresh `past_key_values = DynamicCache(...)`. This is [Verified] acceptable since we only need per-expert metrics, not causal consistency across batches.

#### 8.5.5 Change 5: Hash Router Input Passthrough
**Verdict: [Verified] diagnosis, [Unverified] implementation**

`DeepseekV4HashRouter.forward` requires `input_ids` as a second positional arg:
```python
def forward(self, hidden_states, input_ids):
```
The current `extract_router_logits` tries only `router_module(input)` with hidden states, which will fail with `TypeError`. The `except` handler retries with 3D shape and fails identically.

**Fix must**: Carry `input_ids` through the replay cache alongside hidden states, and pass them when calling the router for hash-routed layers. The layerwise observer's `_process_moe_activations` must detect `module.is_hash` and conditionally pass `input_ids`.

**Standard observer path**: Also needs fixing. The hook on the MoE block won't have access to `input_ids` unless stored in a closure or we hook at the router level. **Unaddressed** by the proposed changes.

#### 8.5.6 Change 6: Pruning — 3D Tensor Indexing
**Verdict: [Verified] for weight tensors, [Contradicted] for router handling**

The weight indexing operations are correct. **Problems:**

1. **Router accessed as `gate` not `router`**: Existing prune.py fused path does `moe.router.weight.data[...]`. V4's router is at `moe.gate`. The proposed code correctly uses `moe_module.gate.weight`.
2. **`e_score_correction_bias`**: Exists only on `TopKRouter`, not `HashRouter` — first 3 layers would fail with `AttributeError`. Needs `hasattr` guard.
3. **`_prune_non_fused()` path still runs**: `prune.py:110` checks `if not model_attrs["fused"]` — for V4, if `fused=False`, it enters `ModuleList` path and crashes. If `fused=True`, it enters fused path and tries `moe.router`. Must intercept before this branch with a V4-specific branch.
4. **Config update**: `model_attrs["num_experts"]` must be `"num_local_experts"` for V4.

#### 8.5.7 Change 7: Hash Router Expert Remapping
**Verdict: [Verified] direction, [Contradicted] implementation detail**

1. `tid2eid` is a `buffer`, not a regular attribute. `gate.tid2eid = remapped` overwrites the buffer registration. Must use `gate.register_buffer("tid2eid", remapped, persistent=True)` or `gate.tid2eid.data.copy_(remapped)`.
2. **Pruned expert fallback**: `old_to_new[pruned_expert]` should map to a fallback expert, not index 0 necessarily. Choosing expert 0 as fallback creates a hot expert for all redirected tokens. Use a distributed fallback strategy or accept the imbalance for 3/43 layers.
3. **Layer detection**: Use `config.mlp_layer_types[layer_idx] == "hash_moe"` programmatically rather than assuming layers 0-2.

#### 8.5.8 Change 8: Shared Expert Preservation
**Verdict: [Verified] — trivial to enforce, easily disobeyed**

`moe_module.shared_experts` (type `DeepseekV4MLP`) is a separate attribute not accessed by any current pruning or metric collection path. The `enumerate(module.experts)` bug would NOT accidentally include `shared_experts` — it iterates `module.experts` which is a `DeepseekV4Experts` instance, not `shared_experts`.

**Risk in prune.py**: The non-fused path does `setattr(moe, model_attrs["experts"], retained_experts)`. For V4, if someone adds this without the V4 override, this would replace `moe.experts` (a `DeepseekV4Experts`) with a `ModuleList`, losing `shared_experts` (a sibling attribute). The guard must be structural (class-name-based), not a feature flag.

#### 8.5.9 Summary of Required Fixes Beyond the 8 Proposed

| # | Issue | Severity | Proposed changes cover it? |
|---|-------|----------|---------------------------|
| 1 | Standard observer hook crashes on V4's single-tensor block output | **Critical** | No |
| 2 | `fused` flag doesn't fit V4 — both True and False break differently | **Critical** | No |
| 3 | `enumerate(module.experts)` breaks on `DeepseekV4Experts` | **Critical** | No (observer path) |
| 4 | `extract_router_logits` gets `indices` (3rd element) instead of `logits` (1st) for V4 triple return | **High** | Partially (Change 2) |
| 5 | Hash router `input_ids` not passed in layerwise `extract_router_logits` | **High** | Partially (Change 5) |
| 6 | Hash router `input_ids` not available in standard observer hook | **High** | No |
| 7 | Prune.py enters `ModuleList` or `moe.router` path before V4-specific branch | **High** | No (branching order) |
| 8 | `tid2eid` remapping overwrites registered buffer | **Medium** | No |
| 9 | `e_score_correction_bias` missing on HashRouter layers 0-2 | **Medium** | No |
| 10 | `input_ids` not in `ReplayBatch` dataclass | **Medium** | Partially (Change 5) |
| 11 | FP4+FP8 weight loading strategy not committed | **High** | No |

**Bottom line**: The 8 proposed changes correctly identify the major V4 architecture differences and sketch the right direction for most. However, they only target the layerwise observer + prune paths. The standard (non-layerwise) observer path is entirely unaddressed and will crash on V4 at several points. The `fused` flag binary is fundamentally unsuitable for V4's architecture — V4 needs its own observer subclass that side-steps both the `ModuleList`-iteration and fused-router paths. And the FP4/FP8 weight loading strategy (§3.2) needs to be validated with an actual `from_pretrained` call before any other work is committed.

## 9. References

- DeepSeek V4 Technical Report: https://arxiv.org/abs/2606.19348
- HF modeling file: `transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py`
- Flash config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json
- Pro config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json
- Existing REAP DeepSeek V2 support: `src/reap/models/modeling_deepseek.py`
- REAP paper: https://arxiv.org/abs/2510.13999 — Cerebras, May 2026
- AIMER paper: https://arxiv.org/abs/2603.18492 — Jun 2026
- Cerebras REAP blog: https://www.cerebras.ai/blog/reap
- REAP code: https://github.com/CerebrasResearch/reap
- AIMER code: https://github.com/ZongfangLiu/AIMER
- Full paper review: `docs/research-papers/MoE-pruning-paper-review.md`

---

## Appendix A: HF Repo Deep-Dive Findings

### A.1 Config Files (config.json)

#### Flash (284B) — `hidden_size: 4096`
```json
{
  "architectures": ["DeepseekV4ForCausalLM"],
  "hidden_size": 4096,
  "num_hidden_layers": 43,
  "num_hash_layers": 3,
  "num_nextn_predict_layers": 1,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "num_experts_per_tok": 6,
  "moe_intermediate_size": 2048,
  "head_dim": 512,
  "q_lora_rank": 1024,
  "o_lora_rank": 1024,
  "o_groups": 8,
  "qk_rope_head_dim": 64,
  "num_attention_heads": 64,
  "num_key_value_heads": 1,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "scoring_func": "sqrtsoftplus",
  "routed_scaling_factor": 1.5,
  "topk_method": "noaux_tc",
  "swiglu_limit": 10.0,
  "norm_topk_prob": true,
  "expert_dtype": "fp4",
  "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
  "sliding_window": 128,
  "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536},
  "max_position_embeddings": 1048576,
  "compress_rope_theta": 160000,
  "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0],
  "vocab_size": 129280,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.57.1"
}
```

#### Pro (1.6T) — `hidden_size: 7168`
Differences from Flash: `hidden_size: 7168`, `num_hidden_layers: 61`, `n_routed_experts: 384`, `moe_intermediate_size: 3072`, `num_attention_heads: 128`, `q_lora_rank: 1536`, `o_groups: 16`, `index_topk: 1024`, `routed_scaling_factor: 2.5`, `compress_ratios: [128, 128, 4, 128, 4, 128, ...61 entries..., 0]`.

### A.2 Weight Sharding

| Metric | Flash | Pro |
|--------|-------|-----|
| Total size | 160 GB | 865 GB |
| Shard count | 46 | 64 |
| First shard | 1.06 GB (embed + head) | 1.85 GB (embed + head) |
| Subsequent shards | ~3.57 GB each | 13.9 GB each |
| Storage | Xet (HF LFS) | Xet (HF LFS) |

**Key insight for block-from-disk:** Flash's 3.57 GB shards store ~2 layers each. Pro's 13.9 GB shards span ~1 layer. The 11 MB `model.safetensors.index.json` (LFS pointer) has the tensor→shard mapping.

### A.3 Encoding Format (`encoding/`)

Both models share identical encoding code at `encoding/encoding_dsv4.py` (27.9 kB). It is a self-contained Python encoder/decoder with 4 test cases.

**Special tokens (V4's own format, NOT Jinja):**
- `bos_token = "<｜begin▁of▁sentence｜>"`
- `eos_token = "<｜end▁of▁sentence｜>"`
- `thinking_start = "<think>"`, `thinking_end = "</think>"`
- `user_prefix = "<｜User｜>"`, `assistant_prefix = "<｜Assistant｜>"`
- `dsml_token = "｜DSML｜"` (tool calling)

**Three reasoning modes:** Non-Think (`</think>` immediately), Think High (interleaved `<think>` blocks), Think Max (special system prefix).

**Tool calling format (DSML, XML-like):**
```xml
<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
```

**Impact on REAP:** encoding code is for prompt construction at inference time. REAP observer uses raw tokens + logits, so it's NOT needed for pruning. Tokenizer is loaded via standard `AutoTokenizer.from_pretrained()`.

### A.4 Custom Inference Pipeline (`inference/`)

Six files totaling 76.2 kB:

| File | Size | Purpose |
|------|------|---------|
| `model.py` | 38.6 kB | Complete PyTorch model with all V4 innovations |
| `kernel.py` | 22.2 kB | TileLang JIT CUDA kernels |
| `convert.py` | 7.08 kB | HF safetensors → inference format |
| `generate.py` | 6.3 kB | Interactive/batch generation |
| `config.json` | 991 B | Subset of HF config |
| `README.md` | 951 B | Usage instructions |

#### A.4.1 Key Architecture from model.py

**MoE (`MoE.forward`):**
```python
# Gate dispatches: Gate.forward(x, input_ids) → (weights, indices)
# Only first n_hash_layers use hash router (needs input_ids)
# Iterates local experts, dispatches via torch.where(indices == i)
# all_reduce across TP ranks, then add shared_experts
```

**Gate (`Gate.forward`):**
```python
# hash mode: indices = self.tid2eid[input_ids]  (static int32 lookup table)
# learned mode: indices = (scores + bias).topk(topk)[1]
# scoring funcs: softmax, sigmoid, or sqrtsoftplus
```

**Hyper-Connections (`Block`):**
```python
# hc_mult=4: 4 parallel hidden state copies
# hc_pre: learned Sinkhorn-weighted collapse → 1 stream
# hc_post: learned expand → combine with residual
```

**Attention (`Attention`):** Three layer types by `compress_ratio`:
- `0`: Pure sliding window (layers 0-1 Flash, 0-1 Pro)
- `4`: Sliding window + learned indexer + compressor (most layers)
- `128`: Sliding window + uniform compress (later layers)

#### A.4.2 Custom CUDA Kernels (kernel.py)

All written in **TileLang** (not raw CUDA), JIT compiled at runtime:

| Kernel | What it does |
|--------|-------------|
| `act_quant_kernel` | Per-block FP8 activation quant (128-wide) |
| `fp4_quant_kernel` | Per-block FP4 quant (32-wide), power-of-2 scale |
| `fp8_gemm_kernel` | FP8 x FP8 GEMM with dual per-block scaling |
| `fp4_gemm_kernel` | FP8 act x FP4 weight GEMM (FP4→FP8→GEMM) |
| `sparse_attn_kernel` | Index-gathered sparse attention + online softmax |
| `hc_split_sinkhorn_kernel` | Sinkhorn balancing for HC split weights |

**Impact on REAP:** TileLang is NOT needed for observation. Observer uses standard PyTorch hooks on expert weights (loaded by HF in BF16/FP32 from FP4 via decompression). The custom kernels only accelerate inference.

#### A.4.3 Weight Conversion (convert.py)

Maps HF weight names → inference format. Notable renames:
- `e_score_correction_bias` → `bias`
- `weight_scale_inv` → `scale`
- `q_a_proj` → `wq_a`, `q_b_proj` → `wq_b`
- `gate_proj` → `w1`, `down_proj` → `w2`, `up_proj` → `w3`

Experts stored as INT8 (packed FP4). Converted to FP4 (`float4_e2m1fn_x2`) or optionally cast to FP8 via `cast_e2m1fn_to_e4m3fn()`.

### A.5 Modal GPU Pricing (from modal.com/pricing, 2026-06-27)

| GPU | $/hr | Notes |
|-----|------|-------|
| B200 | $6.25 | Not needed for Flash |
| H200 | $4.54 | Overkill for Flash |
| H100 | $3.95 | Viable, overkill |
| **RTX PRO 6000** | **$3.03** | **96 GB VRAM, best fit** |
| **A100 80GB** | **$2.50** | **Cheapest viable option** |
| L40S | $1.95 | 48 GB, too small |
| L4 | $0.80 | 24 GB, too small |

**Plans:** Starter ($0, $30/mo free credits, 10 GPU concurrency), Team ($250/mo, $100 free credits, 50 GPU concurrency), Enterprise (custom).

**Estimated layerwise observer cost (Flash, single forward pass):**
- 43 layers × ~3 min/layer = ~2.15 hr
- A100 80GB @ $2.50/hr → **$5.38**
- RTX PRO 6000 @ $3.03/hr → **$6.51**
- H100 @ $3.95/hr → **$8.49**

### A.6 generation_config.json (both models)
```json
{
  "bos_token_id": 0, "eos_token_id": 1,
  "do_sample": true, "temperature": 1.0, "top_p": 1.0,
  "transformers_version": "4.46.3"
}
```
Note: `transformers_version` discrepancy (4.46.3 vs 4.57.1 in config.json).

### A.7 Model README — Key Takeaways

Three innovations highlighted:
1. **Hybrid Attention:** CSA + HCA → 27% FLOPs, 10% KV cache vs V3.2
2. **mHC:** Strengthened residuals, 4x parallel streams
3. **Muon Optimizer** (training only)

Post-training: domain-specific expert cultivation → on-policy distillation.
Benchmarks show Flash approaching Pro on reasoning tasks (LiveCodeBench: Flash 91.6 vs Pro 93.5).

### A.8 Implications for REAP Pipeline

| Component | Impact |
|-----------|--------|
| observer.py | `DeepseekV4MoEObserver` subclass needed — bypasses both `fused`/`ModuleList` |
| pruning_metrics.py | Must handle per-expert sparse dispatch, not full `[E,T,D]` tensor |
| model_util.py | Add `DeepseekV4ForCausalLM` entry to `MODEL_ATTRS` |
| prune.py | Guard against `ModuleList` path, 3D tensor indexing |
| layerwise_observer.py | `ReplayCache` needs `input_ids` for hash router layers |
| layerwise_model_utils.py | Handle HC 4x hidden state, detect `DeepseekV4DecoderLayer` |
| eval.py | vLLM/SGLang support confirmed (README shows commands) |
| data.py | No change — standard HF tokenizer |
| encoding/ | Not needed by REAP (tokenizer alone suffices) |

### A.9 Open Questions

1. **Weight loading:** Does `AutoModelForCausalLM.from_pretrained` decompress FP4→BF16/FP32? HF `config.json` says `torch_dtype: bfloat16`. Critical first validation step.
2. **TileLang:** Is it required for HF forward, or only for custom inference? If needed, add `pip install tilelang`.
3. **Pro feasibility:** Each layer's experts ≈ 12B params = ~40 GB in FP8. Exceeds A100 80GB even with layerwise approach. Pro likely needs multi-GPU or 96GB+ GPUs.
4. **Lightning AI pricing:** Needs user input to compare with Modal ($2.50/hr A100, $3.03/hr RTX PRO 6000).
