# Memory Budget: DeepSeek V4 Flash REAP Pruning (Block-from-Disk)

**Date:** 2026-06-27
**Target Hardware:** Lightning AI RTX PRO 6000 (96 GB VRAM, 180 GB RAM)
**Model:** DeepSeek-V4-Flash (284B)
**Pipeline:** Block-from-disk layerwise observer → CPU pruning → eval

---

## 1. Model Configuration

| Field | Value |
|---|---|
| `num_hidden_layers` | 43 (3 hash, 40 MoE top-k) |
| `hidden_size` | 4096 |
| `moe_intermediate_size` | 2048 |
| `n_routed_experts` | 256 |
| `num_experts_per_tok` | 6 |
| `head_dim` | 512 |
| `hc_mult` | 4 |
| Total params | ~284B |
| Weight format (on disk) | Mixed FP4 + FP8 (~160 GB) |
| Weight format (decompressed BF16) | ~560 GB |
| Safetensor shards | 46 × ~3.57 GB each |

---

## 2. Per-Layer BF16 Weight Breakdown

### 2.1 MoE Experts (dominant)

```
gate_up_proj: 256 × 2 × 2048 × 4096 = 4,294,967,296 params → 8.0 GB BF16
down_proj:    256 × 4096 × 2048      = 2,147,483,648 params → 4.0 GB BF16
Total experts: 6,442,450,944 params → 12.0 GB BF16
```

### 2.2 Other Components

| Component | Params | BF16 Size |
|---|---|---|
| MoE gate (router) | 256 × 4096 = 1,048,576 | 2 MB |
| Shared expert MLP | 4096×8192 + 8192×4096 ≈ 67.1M | 128 MB |
| Attention (Q projections) | 4096×1024 + 1024×32768 ≈ 37.7M | 72 MB |
| Attention (KV projections) | 4096×512 ≈ 2.1M | 4 MB |
| Attention (O projections grouped) | ~90-100M | ~180 MB |
| mHC parameters | ~0.4M | ~1 MB |
| Norms, biases, etc. | ~1M | ~2 MB |
| **Total other** | **~200M** | **~0.4 GB** |

### 2.3 Per-Layer Total

| | Size |
|---|---|
| Experts (BF16) | 12.0 GB |
| Other (BF16) | ~0.4 GB |
| **Total BF16 per layer** | **~12.4 GB** |

---

## 3. Observation Phase: Peak Memory

### 3.1 Peak CPU Memory

Block-from-disk approach — NO full model load. Only one layer at a time.

| Component | Size | Notes |
|---|---|---|
| 1 layer BF16 weights | ~12.4 GB | Decompressed from shard, on CPU temporarily |
| Temp decompression buffer | ~3.57 GB | Raw shard data (I8+FP8 packed) |
| Replay cache (64 batches × 256 MB) | 16 GB | [B=4, S=2048, 4, D=4096] × FP16 × 64 |
| Non-backbone modules | ~2 GB | Embed + norm + lm_head |
| Calibration data (64 batches × 4 × 2048) | ~4 GB | Tokenized dataset |
| Python/PyTorch overhead | ~2 GB |
| **Peak CPU** | **~38 GB** |

Margins:
- Required: 38 GB
- Available (Lightning): 180 GB
- **Headroom: 142 GB (79% free)**

### 3.2 Peak GPU Memory (Observation)

Single decoder layer on GPU at a time. Incremental expert loop — NO `[256, T, D]` tensor.

| Component | Size | Calculation |
|---|---|---|
| 1 layer BF16 weights | 12.4 GB | Transferred from CPU |
| Input tokens (flat) | 64 MB | 8192 × 4096 × 2 B |
| Router logits | 8 MB | 8192 × 256 × 4 B (float32) |
| Single expert gate/up activations | 128 MB | 8192 × 4096 × 4 B × 2 (float32 intermediates) |
| Single expert down output | 64 MB | 8192 × 4096 × 2 B |
| Expert metrics accumulation | ~2 MB | Per-expert scalars × 256 |
| Attention intermediates | ~1.5 GB | Flash attn, LayerNorm, residual |
| **Peak GPU** | **~14.2 GB** | |

Margins:
- Required: 14.2 GB
- Available: 96 GB
- **Headroom: 81.8 GB (85% free)**

### 3.3 Time Budget (Observation)

| Operation | Per Layer | All 43 Layers |
|---|---|---|
| Load shard → decompress → GPU xfer | ~5 s | ~3.6 min |
| Forward 64 batches (incremental experts) | ~49 s | ~35 min |
| Unload + GC | ~1 s | ~0.7 min |
| **Total observation** | **~55 s** | **~39.3 min** |

Assumptions:
- NVMe sequential read: ~3 GB/s (shard file ~3.57 GB → ~1.2 s)
- FP4→BF16 decompression: CPU-bound, estimated ~3 s per layer
- PCIe 4.0 ×16 (CPU→GPU): ~25 GB/s → ~0.5 s per layer
- Each expert: 2 × F.linear (gate_up: 2048×4096, down: 4096×2048) × 64 batches × 256 experts = ~32,768 matmuls
- Single small matmul on RTX PRO 6000: ~30 µs → ~1 s per layer compute
- Real bottleneck: kernel launch overhead (256 experts × 2 matmuls × 64 batches = 32,768 launches)
- Estimated real time: ~49 s per layer (dominated by launch overhead + Python loop)

---

## 4. Pruning Phase: Memory

### 4.1 Strategy: Per-Layer CPU Pruning

The pruning step does NOT need the full model in GPU memory. It:
1. Loads one layer at a time from disk (or keeps the last observed layer)
2. Indexes 3D param tensors along dim 0
3. Saves compacted weights to output directory

No forward pass, no gradients, no optimizer.

| Component | Size |
|---|---|
| 1 layer BF16 weights (input) | 12.4 GB |
| 1 layer BF16 weights (output, retained) | 12.4 GB × (1 - compression_ratio) |
| Index arrays | ~1 KB |
| **Peak CPU (pruning)** | **~25 GB** (input + output for 50% compression) |

### 4.2 Alternative: Full Model CPU Load + Prune

Only viable if `device_map="cpu"` decompresses to BF16 without OOM.

| Component | Size |
|---|---|
| Full model decompressed BF16 | ~560 GB |
| Python/PyTorch overhead | ~2 GB |
| **Peak CPU** | **~562 GB** ❌ |

**Verdict:** Full CPU load is NOT viable on 180 GB machine. Per-layer pruning is mandatory.

### 4.3 Time Budget (Pruning)

| Operation | Time |
|---|---|
| Load 1 layer from shard | ~5 s |
| Index 3D params along dim 0 | ~0.1 s |
| Save compacted layer | ~3 s |
| **Per layer** | **~8.1 s** |
| **All 43 layers** | **~5.8 min** |
| Save config + tokenizer | ~0.2 min |
| **Total pruning** | **~6 min** |

---

## 5. Eval Phase: Projected Sizes

Pruned model is saved as standard HF `from_pretrained` compatible.

| Compression | Retained Experts | BF16 Weight Size (approx) | FP4+FP8 Packed Size (approx) |
|---|---|---|---|
| 0% (unpruned) | 256 | 560 GB (full decompress) | 160 GB |
| 10% | 230 | ~500 GB | ~144 GB |
| 25% | 192 | ~420 GB | ~120 GB |
| 50% | 128 | ~280 GB | ~80 GB |

Eval via `from_pretrained(device_map="auto")`:
- At 50% pruning: ~80 GB packed → loads onto GPU with CPU offloading for remaining layers
- RTX PRO 6000 96 GB: fits at 50%+ compression
- At 25% pruning (120 GB packed): needs CPU offloading or a larger GPU

---

## 6. Risk Register Update

New risks identified during memory budget analysis:

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| M1 | `safetensors.safe_open` returns packed FP4 (I8) bytes, not BF16 — raw read gives 160 GB, not 560 GB | Block-from-disk must implement custom FP4→BF16 decompression | **Certain** | Use `from_pretrained` with `device_map="cpu"` + move one layer to GPU at a time, OR implement I8→FP4 unpack + F8_E8M0 dequant for raw shard loading |
| M2 | Full CPU decompress (BF16 = 560 GB) exceeds Lightning's 180 GB | Cannot use simple `from_pretrained(device_map="cpu")` approach | **Certain** | Block-from-disk is mandatory — never load more than 1-2 layers onto CPU simultaneously |
| M3 | Replay cache at 16 GB + layer weights at 12.4 GB = 28.4 GB peak CPU without calibration data | Within limits | Low | Verified: 28.4 + 4 GB calibration + 2 GB overhead = ~35 GB total, well under 180 GB |
| M4 | Per-expert `F.linear` loop kernel launch overhead causes 256× slowdown vs batched matmul | Observation takes 2+ hours instead of ~40 min | Medium | If measured >3 min/layer, batch experts in groups of 16 (16× fewer launches, similar memory) |
| M5 | Pruning step loads model twice (observation + pruning) | 5 min extra I/O | Medium | Keep last observed layer on CPU, prune immediately before loading next layer |
| M6 | `from_pretrained` for eval at <50% compression exceeds 96 GB VRAM | Eval fails on target hardware | Medium | Use CPU offloading (`device_map="auto"`) or higher compression ratio |

---

## 7. Summary

| Phase | CPU RAM | GPU VRAM | Time | Fits RTX PRO 6000? |
|---|---|---|---|---|
| Observation | ~38 GB | ~14.2 GB | ~39 min | ✅ 85% VRAM headroom |
| Pruning | ~25 GB | 0 GB | ~6 min | ✅ CPU-only |
| Eval (50% prune) | ~80 GB | ~80 GB (packed) | Model-dependent | ✅ Marginal — use `device_map="auto"` |
| Eval (25% prune) | ~120 GB | ~120 GB (packed) | Model-dependent | ⚠️ Offloading needed |

**Key numbers:**
- **Peak VRAM (observation):** 14.2 GB of 96 GB → 81.8 GB free
- **Peak CPU RAM (observation):** 38 GB of 180 GB → 142 GB free
- **Peak CPU RAM (pruning):** 25 GB of 180 GB → 155 GB free
- **Total pipeline time:** ~45 minutes (39 min observation + 6 min pruning)
- **Estimated cloud cost (Lightning AI RTX PRO 6000 @ $1.46/hr spot):** ~$1.10
- **Estimated cloud cost (Lightning AI RTX PRO 6000 @ $2.80/hr on-demand):** ~$2.10

**Conclusion:** The pipeline fits comfortably on Lightning AI's RTX PRO 6000 (96 GB VRAM, 180 GB RAM) with >80% headroom in both memory domains. Block-from-disk is essential — full model CPU load is infeasible (560 GB > 180 GB). The only constraint is eval phase at low compression ratios, which can use CPU offloading.
