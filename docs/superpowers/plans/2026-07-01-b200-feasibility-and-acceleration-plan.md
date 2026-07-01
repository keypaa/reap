# B200 Feasibility & Acceleration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine the cheapest, fastest path to run REAP on DeepSeek-V4-Flash (284B) with paper-standard calibration (12,228 samples at 16,384 seq len). Evaluate B200 (179 GB, $5-6/hr) vs 8× A100-80GB (640 GB, $12/hr) vs staying on RTX PRO 6000 ($1.46/hr).

**Architecture:** Three paths evaluated: (A) keep RTX PRO 6000 layerwise with optimized throughput, (B) B200 with VRAM-resident FP4 weights for disk-less layerwise (~2-3× speedup), (C) multi-GPU cluster with tensor parallelism for true full-model forward (~50-100× speedup).

**Tech Stack:** PyTorch 2.7+, CUDA 12.4+, transformers 5.12+, FlashAttention-4 (B200), V4BlockDiskLoader, vLLM (for multi-GPU path)

---

## Global Constraints

- DeepSeek-V4-Flash: 284B params, 256 experts, 43 layers
- REAP paper spec for ≥110B: 12,228 samples at 16,384 seq len, no packing
- Our current throughput: 5.6s/batch at 8192 seq len on RTX PRO 6000 (96 GB)
- All changes must work on current fork (keypaa/reap), not upstream
- **No renting anything until plan is fully verified (hard constraint)**

---

# ⚠️ CORRECTION: This Pipeline Does NOT Do FP4 Inference

**Critical fact:** Every number in this document assumes weights are in **BF16 in VRAM**. The FP4 format is only a disk serialization format — it's decompressed to full BF16 before any computation touches it. There is no FP4 inference path here.

The code proves this:
```
v4_block_loader.py:552-565 — load_into_block()
  554: state_dict = self._build_layer_state_dict(layer_idx)
       # ↑ calls dequantize_fp4_weight() → .to(torch.bfloat16)
  558: block.to_empty(device=device)
  560: block.load_state_dict(state_dict, assign=True)
       # ↑ puts BF16 weights in .weight
  564: block.to(device)
       # ↑ BF16 on GPU; FP4 discarded
```

After `load_into_block`, `block.mlp.experts.gate_up_proj.weight` is a BF16 tensor `[256, 2*intermediate, hidden]`. **FP4 never reaches GPU memory.** It's a disk compression format only.

**What this means for REAP:**
- **Calibration is correct** — BF16 compute is BF16 compute. The activation statistics REAP collects are valid regardless of how weights were stored on disk.
- **Calibration timing/VRAM numbers don't tell us about FP4-resident inference** — they measure BF16 weight residency, which is 4× the size of FP4.
- **Pruning 50% of experts reduces VRAM, but not by the FP4 factor** — you go from 256 BF16 experts (14.4 GB/layer) to 128 BF16 experts (7.2 GB/layer). You don't get the additional 4× compression from FP4 because there's no fused dequant kernel.
- **To get real FP4 inference savings**, you'd need a fused dequant matmul kernel (like bitsandbytes `Linear4bit`) that keeps FP4 in VRAM and dequantizes tile-by-tile during the matmul. This doesn't exist in our pipeline.

**What this means for the numbers below:**
- The cost-per-sample table compares **observation throughput** (dominated by attention compute, not weight loading). It is valid for comparing calibration costs across hardware.
- VRAM figures throughout the doc are BF16-residency numbers. Halving the expert count (pruning) halves the weight VRAM, but never reaches FP4 density.
- Post-pruning inference VRAM is a separate analysis requiring a fused dequant kernel. That is not scoped in this plan.

---

# Path Cost Per Sample (Observation Throughput Only)

Don't compare GPU cost/hr. Compare **cost per sample** — that's what actually matters for calibration.

| Path | HW | Samples/hr | Cost/hr | **Cost per 1,000 samples** | Paper Standard? |
|---|---|---|---|---|---|
| **A** | RTX PRO 6000 | ~63 at 8192 | $1.46 | **$23.17** | ❌ |
| **B** | B200 | ~100 at 8192 | $5.50 | **$55.00** | ❌ |
| **C** | **8× A100-80GB** | **~12,228 at 16384** | **$12.00** | **$0.98** | ✅ |

**Path B (B200) is the WORST value:** $55 per 1,000 samples — 56× more expensive than Path C, slower than Path A per dollar.

**Path C (8× A100-80GB) is 56× cheaper per sample than B200 and 24× cheaper than RTX PRO 6000.**

**One hour on 8× A100-80GB delivers more calibration data than 200 hours on RTX PRO 6000.**

For all 4 datasets (48,912 samples):
- Path C: **4 hours, $48** — paper-standard, done in a morning
- Path A: **777 hours, $1,134** — would take a month
- Path B: never finishes

---

# THE B200 TRAP

**Do NOT rent a B200.** Here's why:

1. **179 GB is not enough for full-model BF16** (needs 568 GB). You're stuck in layerwise mode.
2. **B200 in layerwise mode is only 2-3× faster** than RTX PRO 6000 (memory bandwidth improvement, not compute) but costs **4× more per hour**.
3. **Result: B200 processes fewer samples per dollar** than RTX PRO 6000 — it's slower per $, not faster.
4. **Cannot reach paper standard** (12,228 at 16k) — layerwise has a fundamental 43× multiplier.
5. **B200 doesn't fix the FP4 issue** — our pipeline decompresses FP4→BF16 on load regardless of GPU. B200's extra VRAM (179 GB vs 96 GB) doesn't buy FP4 inference because the pipeline never does FP4 inference. The weights would still be BF16 in VRAM.
6. **Even with a fused FP4 kernel, B200 alone can't fit the full model** — 568 GB BF16-equivalent across 179 GB is impossible.

**The ONLY scenario where B200 helps:** You write a fused dequant kernel AND you're OK staying in layerwise mode. Then B200's 179 GB holds all FP4 weights (142 GB) + one dequantized layer + activations. But this is a ~2-3× speedup (no disk I/O) at 4× the cost — terrible ROI compared to 8× A100-80GB at $12/hr which gives full-model forward.

**Conclusion:** B200 is a dead end for this project. Skip it entirely.

---

## Research Findings

### B200 Capacity Reality (For Reference Only — We're Not Using It)

**IMPORTANT:** All figures below assume the **naive dequant** model (BF16 weights in VRAM). FP4 format is disk-only — it is decompressed to BF16 before reaching GPU memory. There is no FP4-resident inference path.

| Item | Size in VRAM (BF16) | Size if fused FP4 existed | Fits B200 (179 GB)? |
|---|---|---|---|
| Full model weights | 568 GB (BF16) | 142 GB (FP4) | ❌ BF16. ✅ FP4 (hypothetical). |
| Single layer weights | ~14.4 GB (BF16) | ~3.6 GB (FP4) | ✅ Obviously |
| Single layer + activations (8192 seq, bs=1) | ~14.4 + 15 GB = 29.4 GB | ~3.6 + 15 GB = 18.6 GB | ✅ |
| Single layer + activations (16384 seq, bs=1) | ~14.4 + 30 GB = 44.4 GB | ~3.6 + 30 GB = 33.6 GB | ✅ |

**Key constraint:** B200 alone cannot hold the full model in BF16. Our current pipeline only loads one layer at a time, so B200's extra VRAM isn't utilized for what matters most (full-model forward).

### Speed Bottleneck Reality

Layerwise requires 43 separate forward passes per batch — one per layer.

| Approach | Per-layer time | 12,228 batches × 43 layers | 1,000 batches × 43 layers |
|---|---|---|---|
| RTX PRO 6000 (current) | ~5.6s at 8192 | ~817 hrs | ~67 hrs |
| B200 (FP4 VRAM cache, ~3× faster) | ~1.9s at 8192 | ~278 hrs | ~23 hrs |
| B200 (16k seq, ~7.5s) | ~7.5s at 16384 | ~1,097 hrs | ~90 hrs |

### Where 12,228 Samples IS Feasible

The REAP paper ran on Cerebras CS-3 (wafer-scale) — the full model runs as one forward pass. To replicate this on cloud GPUs, you need **tensor parallelism across multiple GPUs**:

| Setup | Total VRAM | Full model forward? | Est. time for 12,228 batches at 16k |
|---|---|---|---|
| **8× A100-80GB (NVLink)** | 640 GB | ✅ (BF16, TP=8) | **~30-45 min** |
| 8× H100-80GB (NVLink) | 640 GB | ✅ (BF16, TP=8) | ~20-30 min |
| 2× B200 (NVLink) | 358 GB | ✅ (FP8, TP=2) | ~40-60 min |
| 1× B200 | 179 GB | ❌ (BF16 OOM) | N/A |

### Fix Classification (Which of Our 9 Fixes Survive in Each Path)

| # | Fix | Path A (RTX Layerwise) | Path B (B200 — SKIP) | Path C (Multi-GPU) |
|---|---|---|---|---|
| 1 | total_tokens device fix | ✅ Needed | N/A | ✅ Needed (same pruning_metrics) |
| 2 | FP8 compressor dequant | ✅ Needed | N/A | ✅ Needed (same weight format) |
| 3 | 4D causal mask | ✅ Needed | N/A | ✅ Needed (same V4 arch) |
| 4 | Block offload to meta | ✅ Needed | N/A | ❌ Not needed (no layerwise) |
| 5 | Embed to target_device | ✅ Needed | N/A | ❌ Not needed |
| 6 | data.py [subset] parse | ✅ Needed | N/A | ✅ Needed |
| 7 | layerwise chat_template | ✅ Needed | N/A | ❌ Not needed |
| 8 | BatchEncoding handling | ✅ Needed | N/A | ❌ Not needed (uses main.py) |
| 9 | block.to(device) after load | ✅ Needed | N/A | ❌ Not needed |

**Path C needs only fixes 1, 2, 3, 6** — the 4 V4-architecture fixes. Everything else is layerwise plumbing.

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

## Path B: B200 with VRAM-Resident FP4 Weights — SKIP

**Do not pursue.** See "The B200 Trap" above. $55 per 1,000 samples is the worst value of any option, and it can't reach paper-standard calibration anyway.

---

## Path C: Multi-GPU Full-Model Forward (~100× Speedup)

**Cost:** $12/hr on vast.ai for 8× A100-80GB (640 GB total, NVLink)
**Feasible scale:** 12,228 samples at 16,384 seq len in ~30-60 min
**Total for all 4 datasets:** ~4 hours, ~$48
**Status:** Major engineering effort — needs standard REAP observer adapted for V4 on multi-GPU

### Why This Is the Only Path to Paper-Standard Calibration

The core issue: even with all FP4 weights in VRAM (Path B), layerwise requires 43× more forward passes than a single full-model forward. Multi-GPU tensor parallelism lets you:

1. Load all BF16 weights across 8 GPUs (71 GB each on A100-80GB)
2. Run ONE forward pass per batch through all 43 layers
3. Register standard MoE hooks (not layerwise replay cache)

One forward pass for 284B on 8× A100-80GB: ~0.15-0.3 seconds at 16k seq len.
12,228 batches × 0.3s = ~1 hour.

---

## Cost Summary (All Paths)

### Observation (Single Dataset)

| Path | HW | Samples | Samples/hr | Time | **Cost** | Cost per 1k samples | Paper Standard? |
|---|---|---|---|---|---|---|---|
| **A** | RTX PRO 6000 | 500 at 8192 | 63 | 8 hrs | **$11.68** | $23.17 | ❌ |
| **B (SKIP)** | 1× B200 | 1,000 at 8192 | 100 | 10 hrs | **$55.00** | $55.00 | ❌ |
| **C ⭐** | **8× A100-80GB** | **12,228 at 16384** | **12,228** | **1 hr** | **$12.00** | **$0.98** | **✅ Full** |

### All 4 Datasets (Observation + Prune + Eval)

| Path | Total Time | **Total Cost** | Paper Standard? |
|---|---|---|---|
| A (current) | 32 hrs | **$46.72** | ❌ |
| B (SKIP) | 40+ hrs | **$200-240** | ❌ |
| **C ⭐** | **6 hrs (4 obs + 1 prune + 1 eval)** | **~$72** | **✅ Yes** |

---

## ⚠️ CRITICAL: 4 Pre-Flight Verifications Before Renting Anything

**We cannot rent 8× A100-80GB until ALL of these pass.** No exceptions. Each item below must be checked and confirmed working on our current hardware (RTX PRO 6000 or CPU) before spending a cent on multi-GPU.

### Verification 1: Can the model load across 8 GPUs with device_map="auto"?

**What we need to know:** The standard REAP pipeline (`main.py`) uses `device_map="auto"` which relies on Hugging Face `accelerate` to split the model across GPUs. We need to verify this works for V4.

**Check without renting:**
- [ ] Read `main.py` line 119-125 — understand why V4 is blocked (it just raises RuntimeError for any V4 model)
- [ ] Read `model_util.py` — check if V4 is registered in `MODEL_ATTRS` with correct MoE submodule path
- [ ] Read `observer.py` `OBSERVER_CONFIG_REGISTRY` — check if V4 has an observer config (it doesn't — needs to be added)
- [ ] On Lightning RTX PRO 6000: try loading V4 model with `device_map="auto"` on a single GPU (will OOM on full model but tests the accelerate integration):
  ```python
  from transformers import AutoConfig, AutoModelForCausalLM
  config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V4-Flash", trust_remote_code=True)
  # This should fail gracefully, not with a confusing error
  model = AutoModelForCausalLM.from_pretrained(
      "deepseek-ai/DeepSeek-V4-Flash",
      device_map="auto",
      torch_dtype="auto",
      trust_remote_code=True,
  )
  ```

**Pass criteria:** The error (if any) should be a CUDA OOM, not a code crash. That means accelerate attempted GPU placement and ran out of VRAM — it WOULD work with 8 GPUs. If it crashes with `KeyError`, `AttributeError`, or other code error, Path C is blocked.

### Verification 2: Do standard MoE hooks work on V4's MoE structure?

**What we need to know:** The `MoETransformerObserver` registers hooks on `MODEL_ATTRS[model_class]["moe_path"]` submodules. We need to find the correct path for V4's MoE.

**Check without renting:**
- [ ] On Lightning RTX PRO 6000: load layer 0 (with our existing `V4BlockDiskLoader`), then inspect its MoE structure:
  ```python
  from transformers import AutoConfig
  from reap.v4_block_loader import V4BlockDiskLoader
  import torch
  
  config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V4-Flash", trust_remote_code=True)
  loader = V4BlockDiskLoader("deepseek-ai/DeepSeek-V4-Flash", config=config)
  
  # Create a meta block and load it to CPU
  from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer
  with torch.device("meta"):
      block = DeepseekV4DecoderLayer(config, layer_idx=0)
  loader.load_into_block(block, 0, "cpu")
  
  # Inspect MoE submodule
  moe = block.mlp  # or wherever the MoE is
  print(type(moe))
  print([n for n, _ in moe.named_modules()])
  print(hasattr(moe, 'gate'), hasattr(moe, 'experts'))
  ```
- [ ] Based on output, determine the correct `moe_path` for `MODEL_ATTRS`
- [ ] Verify that forward hooks on this path would capture: `router_logits` (from gate) and expert activations (from experts)

**Pass criteria:** V4's MoE module has `gate` and `experts` submodules that match the hook pattern used by `MoETransformerObserver`. If V4 uses a completely different MoE structure (e.g., `DeepseekV4Experts` is not a standard `nn.ModuleList`), we need custom hook code.

### Verification 3: Does the standard observer data format match what prune.py expects?

**What we need to know:** `prune.py` (or `v4_prune_utils.py`) reads observer data to decide which experts to remove. The standard observer and the layerwise observer may save data in different formats.

**Check without renting:**
- [ ] Read `pruning_metrics.py` `initialize_pruning_state()` — this defines the state dict format
- [ ] Read `layerwise_observer.py` `_record_all_blocks_for_batch_group()` — see what gets saved
- [ ] Read `observer.py` `record_all_blocks()` — see what the standard observer saves
- [ ] Compare: are the state dict keys identical? (total_tokens, expert_frequency, pairwise_expert_frequency, ean_sum, weighted_ean_sum, ean_mean)
- [ ] Read `v4_prune_utils.py` — does the prune step read from observer state and how?
- [ ] Read `prune.py` — does it call `initialize_pruning_state` and expect the same keys?

**Pass criteria:** Both observers produce data with the same keys/format from `initialize_pruning_state`. If they differ, we need a format adapter.

### Verification 4: Can prune.py handle observation results from the standard pipeline?

**What we need to know:** `prune.py` may have code paths specific to the layerwise pipeline (e.g., checking for file naming, block indices, etc.).

**Check without renting:**
- [ ] Read `prune.py` end-to-end: how does it load observer data?
- [ ] Read `main.py` `main()` function: how does it save observer data?
- [ ] Compare file naming conventions
- [ ] Create a mock observer data file (using `initialize_pruning_state` for all 43 layers) and test if `prune.py` can read it
- [ ] If `prune.py` references layerwise-specific paths (e.g., `results/.../layerwise/...`), note what needs to change

**Pass criteria:** A mock observer data file with the correct format can be loaded by the prune pipeline without crashes. If `prune.py` and `main.py` use different file formats, list the exact changes needed.

---

## Path C Implementation (If Pre-Flight Passes)

### Task C1: Determine Best Multi-GPU Platform

**Research — no code:**

- [ ] Compare: runpod (8× A100-80GB, ~$13-20/hr), vast.ai (8× A100-80GB, 256 cores, 2,032 GB RAM, 25 TB SSD, **$12/hr**), Lambda Labs (8× A100-80GB, ~$15/hr)
- [ ] Check: do these have NVLink between GPUs? (Required for tensor parallelism without CPU bottlenecks)
- [ ] Check: does vast.ai's 8× A100-80GB instance have NVLink? (If not, performance drops ~2×)
- [ ] Report: cheapest platform WITH NVLink

### Task C2: Research vLLM for Full-Model Observation

**Research — no code:**

vLLM already supports tensor parallelism for DeepSeek V4 (it's the standard serving framework). The question: can we use vLLM's forward pass for observation instead of raw PyTorch?

- [ ] Check if vLLM exposes router logits or can be hooked
- [ ] Check if vLLM supports FP4 weights directly (it does for DeepSeek-V3, likely for V4 Flash too)
- [ ] If vLLM works: observation becomes `vllm.engine.forward()` with hooks → much simpler than our layerwise
- [ ] If vLLM doesn't work: fall back to raw PyTorch with `accelerate`
- [ ] Return: feasibility assessment

### Task C3: Adapt Standard REAP Pipeline for V4

**Files:**
- Modify: `src/reap/main.py` — remove V4 guard (line 121-125)
- Modify: `src/reap/observer.py` — add V4 observer config to `OBSERVER_CONFIG_REGISTRY`
- Modify: `src/reap/model_util.py` — add V4 entry to `MODEL_ATTRS` with correct `moe_path`
- Possibly create: `src/reap/models/modeling_deepseek_v4_patch.py` (if router logits aren't exposed)

If vLLM is usable:
- Use vLLM engine with tensor parallelism
- Register hooks on the vLLM model or intercept at the output
- Collect metrics same way as standard observer

If vLLM is not usable:
- Load model with `device_map="auto"` across 8 GPUs
- Use `accelerate` for distributed inference
- Register standard MoE hooks via `MoETransformerObserver`
- Run: `python -m reap.main ...` (standard pipeline)

**Key insight:** The standard observer (`MoETransformerObserver`) uses forward hooks that trigger on every MoE block automatically during a single forward pass. No layerwise replay needed. This is what the upstream REAP code does for all non-V4 models.

### Task C4: Full Verification Run on Multi-GPU

**Files:**
- No changes — run existing pipeline if adapted

**Verification (on rented 8× A100-80GB, 1 hr min ~$12):**

- [ ] Run 1 batch at 16384 seq len: verify all 43 layers' metrics collected without OOM
- [ ] Run 10 batches: measure per-batch time
- [ ] Check VRAM distribution across 8 GPUs (should be balanced ~70-75 GB each)
- [ ] Run 1228 batches (10% of 12,228): verify no memory leak, no crash
- [ ] Project full 12,228 time from measured throughput
- [ ] Save observer data and verify format matches `prune.py` expectations

### Task C5: Prune on Single GPU

**Files:**
- Modify: `src/reap/prune.py` or `src/reap/v4_prune_utils.py` (if needed)

Pruning (weight removal) is much simpler than observation — it's just weight tensor reshaping. Can run on a single GPU or even CPU.

- [ ] Load observer data from Path C4
- [ ] Verify prune produces pruned model with 128 experts (50% compression)
- [ ] Save pruned model
- [ ] Verify saved model can be loaded (at least the config shows 128 experts)

---

## Risk Assessment

### Path A Risks (RTX PRO 6000)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Memory leak not fully fixed | Medium | Test on Lightning first. If leak persists, add explicit `del` + `torch.cuda.empty_cache()` |
| spot instance preemption mid-run | Medium | Save intermediate results (already done: `save_path` param) |
| 500 samples insufficient for eval | Medium | Accept the tradeoff or switch to Path C |

### Path C Risks (Multi-GPU)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Standard observer hooks don't work with V4's MoE structure | Medium | Add V4-specific hook config to OBSERVER_CONFIG_REGISTRY (Verification 2) |
| device_map="auto" on 8 GPUs fails for V4 | Medium | Try accelerate launcher first; test on single batch (Verification 1) |
| vLLM doesn't expose router logits (can't compute REAP score) | High | Fall back to raw PyTorch with accelerate |
| Multi-GPU spot cost higher than expected | Low | Use on-demand for the 1-hour run ($12) |
| Pruning code assumes layerwise observation format | High | Need format adapter (Verification 3+4) |

---

## Decision Tree

```
Can you accept ~500 samples at 8192 seq len for eval?
├── YES → Use Path A (RTX PRO 6000, $11.68/run)
│
└── NO → Must go Path C: 8× A100-80GB
    ├── Complete 4 Pre-Flight Verifications (takes 1-2 days)
    ├── Implement Path C code changes (takes 2-3 days)
    ├── Rent 8× A100-80GB for verification run ($12)
    ├── Run all 4 datasets ($48 total)
    ├── Prune + eval (~$12)
    └── Total: ~$72, 3-5 days engineering + 6 hours compute
```

---

## Immediate Next Steps (Whatever Path You Choose)

1. **Fix the memory leak** — commit `91c5a44` (already pushed)
2. **Test Path A throughput on Lightning** — verify all 43 layers pass, measure speed
3. **Run 4 Pre-Flight Verifications** (can be done on current HW, no renting needed)
4. **Decide** based on pre-flight results

---

## Appendix: What B200 Actually Changes (For Reference)

**B200 vs RTX PRO 6000 specs:**

| Spec | RTX PRO 6000 | B200 | Ratio |
|---|---|---|---|
| VRAM | 96 GB HBM3e | 179 GB HBM3e | 1.86× |
| Memory bandwidth | ~2 TB/s | ~8 TB/s | 4× |
| FP8 TFLOPS | ~2.0 PFLOPS | ~4.5 PFLOPS | 2.25× |
| CUDA compute cap | 9.0 (Hopper) | 10.0 (Blackwell) | gen bump |
| Flash attention | FA3 | FA4 | gen bump |
| Cost/hr (spot) | ~$1.46 | ~$5-6 | 3.4-4.1× |
| **Cost per 1,000 samples** | **$23.17** | **$55.00** | **0.42× value** |

**All figures assume naive BF16 dequant.** Neither GPU does FP4 inference in our pipeline. B200's extra VRAM (179 GB) would only matter if we either:
- (a) Loaded the full model in BF16 (impossible — needs 568 GB), or
- (b) Had a fused FP4→BF16 dequant kernel (doesn't exist in our pipeline)

B200 is NOT worth it for this workload. The 4× memory bandwidth is neutralized by the 4× higher cost and the fundamental 43× layerwise multiplier.
