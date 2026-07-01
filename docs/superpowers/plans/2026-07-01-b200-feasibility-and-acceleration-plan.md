# B200 Feasibility & Acceleration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether renting a B200 GPU (179 GB VRAM at $5-6/hr) enables running REAP on DeepSeek-V4-Flash (284B) with paper-standard calibration config (12,228 samples at 16,384 seq len) — and if not, what the fastest feasible path is.

**Architecture:** Three parallel paths evaluated: (A) keep RTX PRO 6000 layerwise with optimized throughput, (B) B200 with VRAM-resident FP4 weights for disk-less layerwise (~2-3× speedup), (C) multi-GPU cluster with tensor parallelism for true full-model forward (~50-100× speedup). Path selector at end.

**Tech Stack:** PyTorch 2.7+, CUDA 12.4+, transformers 5.12+, FlashAttention-4 (B200), V4BlockDiskLoader, vLLM (for multi-GPU path)

---

## Global Constraints

- DeepSeek-V4-Flash: 284B params, 256 experts, 43 layers
- REAP paper spec for ≥110B: 12,228 samples at 16,384 seq len, no packing
- Our current throughput: 5.6s/batch at 8192 seq len on RTX PRO 6000 (96 GB)
- All changes must work on current fork (keypaa/reap), not upstream
- No renting B200 until plan is verified (hard constraint)

---

## Research Findings (Pre-Plan)

### B200 Capacity Reality

| Item | Size | Fits B200 (179 GB)? |
|---|---|---|
| FP4 weights (disk format) | ~142 GB | ✅ Yes, with 37 GB to spare |
| BF16 weights (decompressed) | ~568 GB | ❌ No — need 4× B200 |
| FP4 + KV cache (16k seq, bs=1) | ~142 + 25 GB = 167 GB | ✅ Marginal |
| FP4 + KV cache (16k seq, bs=8) | ~142 + 40 GB = 182 GB | ❌ OOM |
| FP4 + activations (observation, bs=1) | ~142 + 15 GB = 157 GB | ✅ |

**Key constraint:** B200 alone cannot hold the full model in BF16. Only option is keeping FP4 weights in VRAM and dequantizing on-the-fly — same as layerwise, just without disk I/O.

### Speed Bottleneck Reality

**Layerwise** (our current approach) requires 43 separate forward passes per batch — one per layer. Even if each layer's forward is 2× faster on B200, total time = `batches × 43 × per_layer_time`.

| Approach | Per-layer time | 12,228 batches × 43 layers | 1,000 batches × 43 layers |
|---|---|---|---|
| RTX PRO 6000 (current) | ~5.6s at 8192 | ~817 hrs | ~67 hrs |
| B200 (FP4 VRAM cache, ~3× faster) | ~1.9s at 8192 | ~278 hrs | ~23 hrs |
| B200 (FP4 VRAM cache, 16k seq, ~7.5s) | ~7.5s at 16384 | ~1,097 hrs | ~90 hrs |

**Conclusion: B200 alone does not make 12,228 samples feasible.** The layerwise approach has a fundamental 43× multiplier that B200's ~3× speedup doesn't overcome.

### Where 12,228 Samples IS Feasible

The REAP paper ran on Cerebras CS-3 (wafer-scale) — the full model runs as one forward pass. To replicate this on cloud GPUs, you need **tensor parallelism across multiple GPUs**:

| Setup | Total VRAM | Full model forward? | Est. time for 12,228 batches at 16k |
|---|---|---|---|
| 8× A100 80GB (NVLink) | 640 GB | ✅ (BF16, TP=8) | ~30-45 min |
| 8× H100 80GB (NVLink) | 640 GB | ✅ (BF16, TP=8) | ~20-30 min |
| 2× B200 (NVLink) | 358 GB | ✅ (FP8, TP=2) | ~40-60 min |
| 1× B200 | 179 GB | ❌ (BF16 OOM) | N/A |

### Fix Classification (Which of Our 9 Fixes Survive in Each Path)

| # | Fix | Path A (RTX Layerwise) | Path B (B200 Layerwise) | Path C (Multi-GPU Full-Model) |
|---|---|---|---|---|
| 1 | total_tokens device fix | ✅ Needed | ✅ Needed | ✅ Needed (same pruning_metrics) |
| 2 | FP8 compressor dequant | ✅ Needed | ✅ Needed | ✅ Needed (same weight format) |
| 3 | 4D causal mask | ✅ Needed | ✅ Needed | ✅ Needed (same V4 arch) |
| 4 | Block offload to meta | ✅ Needed | ✅ Needed | ❌ Not needed (no layerwise) |
| 5 | Embed to target_device | ✅ Needed | ✅ Needed | ❌ Not needed |
| 6 | data.py [subset] parse | ✅ Needed | ✅ Needed | ✅ Needed |
| 7 | layerwise chat_template | ✅ Needed | ✅ Needed | ❌ Not needed |
| 8 | BatchEncoding handling | ✅ Needed | ✅ Needed | ❌ Not needed (uses main.py) |
| 9 | block.to(device) after load | ✅ Needed | ✅ Needed | ❌ Not needed |

**Path C (multi-GPU) needs only fixes 1, 2, 3, 6** — the 4 V4-architecture fixes. Everything else is layerwise plumbing.

---

## Path A: RTX PRO 6000 — Optimized Layerwise (Keep Current HW)

**Cost:** $1.46/hr (spot)
**Feasible scale:** ~300-800 samples at 8192 seq len in 6-10 hrs
**Status:** Working (3 layers passed before OOM, memory leak fix pushed)

### Task A1: Verify Memory Leak Fix on GPU

**Files:**
- No changes — verify existing fix
- Run on Lightning RTX PRO 6000

**Research/Verification:**
- [ ] Confirm `block.to_empty(device="meta")` in `v4_moe_observer.py:172` actually frees GPU memory
- [ ] Check: does `to_empty("meta")` work on PyTorch 2.7+ installed on Lightning? (`torch.__version__`)
- [ ] If not: fallback to `block.to("cpu")` which also frees GPU memory

**Verification command:**
```bash
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reaper-calibration[seed-10k]" \
  --batch-size 1 \
  --batches-per-category 32 \
  --model-max-length 8192 \
  --prune-method "reap" \
  --batched-experts \
  --expert-batch-size 16 \
  --run-observer-only True
```

Expected: All 43 layers complete without OOM. Check `nvidia-smi` after each layer — VRAM should stay stable (~50-60 GB), not accumulate.

- [ ] **If OOM still occurs at layer N > 5:** Memory leak is not fully fixed. Check `cleanup_memory()` in `v4_moe_observer.py:179` — may need `torch.cuda.synchronize()` before `gc.collect()`.
- [ ] **If all 43 layers pass:** Measure peak VRAM after each batch. Should be stable.

### Task A2: Determine Max Batch Size at 8192 Seq Len

**Files:**
- No code changes — empirical testing

**Research/Verification:**
- [ ] Try `--batch-size 2 --expert-batch-size 32` with same command
- [ ] Check VRAM peak: should be ~65-70 GB (well within 96 GB)
- [ ] Measure speed: should be ~6-7s/it (slightly slower per iteration but 2× tokens)
- [ ] Try `--batch-size 4` — likely OOM (attention scores at 4×8192²)
- [ ] Record optimal batch-size for throughput (tokens/sec)

**Goal:** Find max `batch-size × expert-batch-size` combo that stays under 85 GB VRAM.

### Task A3: Compute Realistic Sample Budget

**Files:**
- No code changes

**Research/Verification:**
- [ ] Using optimal batch-size from Task A2, compute: max samples in 8 hrs
- [ ] Formula: `(8 × 3600) / (batch_size × per_iteration_time × 43)`
- [ ] Example: bs=2, 7s/it → 8h × 3600 / (2 × 7 × 43) = ~478 samples
- [ ] Report sample count achievable for each of the 4 datasets

---

## Path B: B200 with VRAM-Resident FP4 Weights (Same HW, New Mode)

**Cost:** $5-6/hr (vast.ai)
**Feasible scale:** ~1,000-2,000 samples at 8192 seq len in 8-10 hrs
**Status:** Requires adding `V4BlockVRAMLoader` — ~2-3 days of engineering

### Task B1: Research Transformers & CUDA Compatibility on B200

**Research/Verification — run on B200 rental (just basic check, 1 hr $5-6):**

- [ ] Check `torch.cuda.get_device_capability()` returns (10, 0) on B200
- [ ] Verify `torch.__version__` — needs ≥ 2.4 for Blackwell support
- [ ] Check `CUDA_HOME` version ≥ 12.4
- [ ] Test `torch.float8_e4m3fn` is available: `torch.tensor([1.0], dtype=torch.float8_e4m3fn)`
- [ ] Check FlashAttention-4 availability: `import flash_attn; flash_attn.__version__`

**If any check fails:** Abort Path B — B200 ecosystem not ready.

**Testing commands:**
```bash
python -c "import torch; print(torch.cuda.get_device_capability(), torch.__version__)"
python -c "import torch; t = torch.tensor([1.0], dtype=torch.float8_e4m3fn); print('FP8 OK')"
python -c "import flash_attn; print('FA:', flash_attn.__version__)"
```

### Task B2: Build V4BlockVRAMLoader

**Files:**
- Create: `src/reap/v4_block_vram_loader.py`
- Modify: `src/reap/v4_moe_observer.py` (select loader based on env)

**Interfaces:**
- Consumes: `model_path`, `config`, `device="cuda"` (must be GPU)
- Produces: Same interface as `V4BlockDiskLoader.load_into_block(block, layer_idx, device)`

Requires changing `load_non_backbone_modules` to work from VRAM-resident state dict.

**Architecture:**
`V4BlockVRAMLoader` extends `V4BlockDiskLoader` but overrides `_load_tensor()`:
- On init: load all FP4/FP8 shard tensors into a single large GPU buffer (142 GB)
- `_load_tensor()` reads from buffer instead of disk + safetensors
- This removes ~0.3s of disk I/O per layer

**Key risks:**
- 179 GB VRAM means the buffer + single dequantized layer + activations must all fit
- Buffer: 142 GB (FP4). Dequantized layer: ~3.5 GB (BF16). Activations: ~15-30 GB (bs=2, 8192). Total: ~161-175 GB. Marginal.
- If OOM: reduce to bs=1 or use CPU buffer fallback

- [ ] **Step 1: Research feasibility**

```python
# Estimate total VRAM usage — run on RTX PRO 6000 first to measure per-layer decompressed size
# Already have this data: single layer ~14.4 GB BF16 weights
# With VRAM buffer: 142 GB (FP4) + 14.4 GB (decomp layer, temporary) + 15 GB (activations)
# = 171 GB — within 179 GB but tight
```

- [ ] **Step 2: Design the VRAM buffer**

```python
class V4BlockVRAMLoader(V4BlockDiskLoader):
    """Like V4BlockDiskLoader but keeps all FP4 weights in a GPU buffer."""
    def __init__(self, model_path, config=None, device="cuda"):
        super().__init__(model_path, config)
        self._vram_buffer = {}  # tensor_name -> fp4_tensor on GPU
        self._buffer_device = torch.device(device)
    
    def _load_all_to_vram(self):
        """Load all FP4/FP8 tensors from disk into GPU buffer."""
        all_tensors = set(self.index["weight_map"].keys())
        for shard_file in self._get_unique_shard_files(all_tensors):
            tensors = safetensors.safe_open(shard_file, device=self._buffer_device.type)
            for key in tensors.keys():
                if key in all_tensors:
                    self._vram_buffer[key] = tensors.get_tensor(key)
        # ~142 GB allocated — VRAM should show ~142 GB used
```

- [ ] **Step 3: Write V4BlockVRAMLoader class** (full implementation)

```python
def _load_tensor(self, tensor_name):
    if tensor_name in self._vram_buffer:
        # Already on GPU — return reference
        # For dequantization, the FP4 data stays in VRAM
        return self._vram_buffer[tensor_name]
    return super()._load_tensor(tensor_name)  # fallback to disk
```

- [ ] **Step 4: Modify layerwise observer to use VRAM loader**

In `v4_moe_observer.py:61-66`:
```python
if self._v4_loader is not None:
    block = self._block_at(block_idx)
    layer_idx = self._actual_layer_idx(block_idx)
    if has_meta_tensors(block):
        self._v4_loader.load_into_block(block, layer_idx, target_device)
```

Add environment variable or auto-detect: if VRAM > 160 GB (i.e., B200), use VRAM loader.

- [ ] **Step 5: CPU smoke test (no GPU needed)**

```python
# Create loader and test _load_tensor for a few layers
# Verify same weights as disk loader
```

- [ ] **Step 6: GPU smoke test (on RTX PRO 6000 first)**

Run 1-2 layers with VRAM loader on RTX PRO 6000 (will OOM on full buffer but test partial load).
Verify dequantized weights match disk-loaded weights exactly.

- [ ] **Step 7: B200 full test (rent B200, 1 hr ~$5-6)**

Same command as current but with VRAM loader. Check:
- VRAM at idle (after buffer load): ~142 GB
- VRAM during layer forward: ~170 GB (within 179 GB)
- Timing: should be ~3-4s/it (vs 5.6s on RTX PRO 6000)

### Task B3: Retune Batch Size on B200

**Files:**
- No changes — empirical

**Verification:**
- [ ] Try `--batch-size 2 --expert-batch-size 32` on B200
- [ ] Measure VRAM peak, speed
- [ ] Compute max samples per hour

---

## Path C: Multi-GPU Full-Model Forward (~100× Speedup)

**Cost:** $10-30/hr (8× A100 on runpod/vast)
**Feasible scale:** ~12,228 samples at 16,384 seq len in ~30-60 min
**Status:** Major engineering effort — needs vLLM integration or tensor parallelism

### Why This Is the Only Path to Paper-Standard Calibration

The core issue: even with all FP4 weights in VRAM (Path B), layerwise requires 43× more forward passes than a single full-model forward. Multi-GPU tensor parallelism lets you:

1. Load all BF16 weights across 8 GPUs (71 GB each on A100-80GB)
2. Run ONE forward pass per batch through all 43 layers
3. Register standard MoE hooks (not layerwise replay cache)

One forward pass for 284B on 8× A100: ~0.15-0.3 seconds at 16k seq len.
12,228 batches × 0.3s = ~1 hour.

### Task C1: Determine Best Multi-GPU Platform

**Research — no code:**

- [ ] Compare: runpod (8× A100-80GB, ~$13-20/hr), vast.ai (8× A100, ~$10-15/hr), Lambda Labs (8× A100, ~$15/hr)
- [ ] Check: does runpod have B200 clusters? (likely not yet)
- [ ] Check: does vast.ai offer 2×B200 instances? (NVLink required for TP)
- [ ] Report: cheapest platform with NVLink-connected GPUs

### Task C2: Research vLLM for Full-Model Observation

**Research — no code:**

vLLM already supports tensor parallelism for DeepSeek V4 (it's the standard serving framework for V4). The question: can we use vLLM's forward pass for observation instead of raw PyTorch?

- [ ] Check if vLLM exposes router logits or can be hooked
- [ ] Check if vLLM supports FP4 weights directly (it does for DeepSeek-V3, likely for V4)
- [ ] If vLLM works: observation becomes `vllm.engine.forward()` with hooks → much simpler than our layerwise
- [ ] Return: feasibility assessment

### Task C3: Adapt Standard REAP Pipeline for V4

**Files:**
- Modify: `src/reap/main.py` (remove V4 guard at line 121-125)
- Modify: `src/reap/observer.py` (register V4 MoE hooks)
- Modify: `src/reap/model_util.py` (add V4 to MODEL_ATTRS for standard observer)

If vLLM is usable:
- Use vLLM engine with tensor parallelism
- Register hooks on the vLLM model or intercept at the output
- Collect metrics same way as standard observer

If vLLM is not usable:
- Load model with `device_map="auto"` across 8 GPUs
- Use `accelerate` for distributed inference
- Register standard MoE hooks via `MoETransformerObserver`
- Run: `python -m reap.main ...` (standard pipeline)

**Key insight:** The standard observer (`MoETransformerObserver`) uses forward hooks that trigger on every MoE block automatically during a single forward pass. No layerwise replay needed. This is what the upstream REAP code does.

### Task C4: Full Verification Run on Multi-GPU

**Files:**
- No changes — run existing pipeline if adapted

**Verification:**
- [ ] Load DeepSeek-V4-Flash on 8× A100 with device_map="auto"
- [ ] Run 1 batch: verify all 43 layers' metrics collected
- [ ] Run 1228 batches (10% of 12,228): measure time, verify no OOM
- [ ] Project full 12,228 time from measured throughput

### Task C5: Prune on Multi-GPU

**Files:**
- Modify: `src/reap/prune.py` or `src/reap/v4_prune_utils.py`

Pruning (weight removal) is much simpler than observation — it's just weight tensor reshaping. Can likely run on a single GPU.

- [ ] Verify pruned model weights can be saved
- [ ] Verify pruned model can be loaded for eval

---

## Cost Comparison (All Paths)

### Observation (Single Dataset)

| Path | HW | Est. Samples | Est. Time | Est. Cost | Paper Standard? |
|---|---|---|---|---|---|
| **A** (current) | RTX PRO 6000 | 500 at 8192 | 8 hrs | $11.68 | ❌ (4% coverage) |
| **B** (B200 layerwise) | 1× B200 | 1,000 at 8192 | 10 hrs | $50-60 | ❌ (8% coverage) |
| **C** (multi-GPU) | 8× A100 | 12,228 at 16384 | 1 hr | $13-20 | ✅ Full |
| **C** (multi-GPU, 4 datasets) | 8× A100 | 48,912 total | 4 hrs | $52-80 | ✅ Full |

### Observation + Prune + Eval (4 Datasets)

| Path | Est. Total | Est. Cost |
|---|---|---|
| A (current) | 32 hrs | $46.72 |
| B (B200) | 40+ hrs | $200-240 |
| C (multi-GPU) | 6 hrs (4 obs + 1 prune + 1 eval) | $78-120 |

---

## Risk Assessment

### Path A Risks (RTX PRO 6000)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Memory leak not fully fixed | Medium | Test on Lightning first. If leak persists, add explicit `del` + `torch.cuda.empty_cache()` |
| Layer 0-2 pass but later layers OOM from KV cache growth | Low | No KV cache in observation mode — no past_key_values |
| spot instance preemption mid-run | Medium | Save intermediate results (already done: `save_path` param) |
| 500 samples insufficient for REAP convergence | Medium | This is the tradeoff — accept or switch to Path C |

### Path B Risks (B200)

| Risk | Likelihood | Mitigation |
|---|---|---|
| 179 GB VRAM insufficient even for FP4 buffer + activations | Medium | Test on single batch first. If OOM, reduce buffer to CPU and stream to GPU |
| B200 CUDA 12.4+ not available on vast.ai | Medium | Check before renting. Use `torch.cuda.is_available()` test |
| FlashAttention-4 not available for V4 | Medium | Fall back to eager attention (slower but works) |
| $5-6/hr rental but only 2-3× speedup = poor ROI | High | Pre-compute ROI before renting — if ROI < 2× over RTX, skip |

### Path C Risks (Multi-GPU)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Standard observer hooks don't work with V4's MoE structure | Medium | Add V4-specific hook config to OBSERVER_CONFIG_REGISTRY |
| device_map="auto" on 8 GPUs fails for V4 | Medium | Try accelerate launcher first; test on single batch |
| vLLM doesn't expose router logits (can't compute REAP score) | High | Fall back to raw PyTorch with accelerate |
| Multi-GPU spot cost higher than expected | Low | Use on-demand for the 1-hour run ($20) |
| Pruning code assumes layerwise observation format | High | Need to align observation data format with what prune step expects |

---

## Decision Tree

```
Is 500-800 samples at 8192 enough for your eval?
├── YES → Use Path A (continue with RTX PRO 6000, $11.68/run)
│
└── NO → Is $50-60/dataset acceptable for 2× samples?
    ├── YES → Use Path B (B200 layerwise, need 2-3 days engineering)
    │
    └── NO → Go Path C (multi-GPU full model)
        ├── Requires: 4 V4-specific fixes (already done)
        ├── Requires: Remove V4 guard in main.py + hook registration
        ├── Requires: Multi-GPU platform account
        ├── Cost: ~$20/hr × 6 hrs = $120 total for all 4 datasets
        └── Delivers: Paper-standard 12,228 samples at 16,384 seq len
```

---

## Immediate Next Steps (Whatever Path You Choose)

1. **Fix the memory leak** — commit `91c5a44` (already pushed)
2. **Test Path A throughput on Lightning** — verify all 43 layers pass, measure speed
3. **Decide** — based on whether 500-800 samples is enough

If Path C:
4. Research vLLM hooks or accelerate-based loading for V4 (Task C2)
5. Remove V4 guard in main.py (Task C3)
6. Test on 1 GPU with small batch first
7. Rent 8× A100 for the run

---

## Appendix: What B200 Actually Changes

**B200 vs RTX PRO 6000 specs:**

| Spec | RTX PRO 6000 | B200 | Ratio |
|---|---|---|---|
| VRAM | 96 GB HBM3e | 179 GB HBM3e | 1.86× |
| Memory bandwidth | ~2 TB/s | ~8 TB/s | 4× |
| FP8 TFLOPS | ~2.0 PFLOPS | ~4.5 PFLOPS | 2.25× |
| CUDA compute cap | 9.0 (Hopper) | 10.0 (Blackwell) | gen bump |
| Flash attention | FA3 | FA4 | gen bump |
| Cost/hr (spot) | ~$1.46 | ~$5-6 | 3.4-4.1× |
| VRAM for FP4 weights | ❌ Cannot hold | ✅ Can hold (142 GB) | — |

**Key takeaway:** B200's advantage is primarily VRAM capacity (allowing FP4 buffer), not compute speed. For the attention-dominated layerwise workload, memory bandwidth (4×) matters more than compute, giving ~2-3× real-world speedup over RTX PRO 6000.
