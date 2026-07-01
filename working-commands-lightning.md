# Cloud Launch Guide: DeepSeek V4 Flash REAP Pipeline

**Last updated:** 2026-06-28
**Target hardware:** RTX PRO 6000 (96 GB VRAM, $1.46/hr spot)
**Model:** deepseek-ai/DeepSeek-V4-Flash (284B, 256 experts, 43 layers, ~160 GB on disk)
**Pipeline:** layerwise observation → prune 50% → eval

---

## Overview

Five stages, each gates the next. Stop and assess at each stage boundary.

| Stage | What | Est. Cost | Stop if |
|-------|------|-----------|---------|
| 0: CPU Setup | Clone, build, verify, V2-Lite smoke test | $0 | Tests fail |
| 1: GPU Env | Launch instance, GPU torch, download ~160 GB | $0 (storage maybe $4-5) | CUDA not available, download fails |
| 2: 1-Layer Smoke | Load 1 V4 layer, observe, prune — measure timing | ~$0.05 | Forward pass fails, VRAM > 90 GB |
| 3: Full Obs | 4 runs × 43 layers | TBD (Stage 2 tells us) | VRAM leak, wrong output path |
| 4: Prune + Eval | Prune 4 models, run eval | TBD | Observer data doesn't match |

---

## Prerequisites

- Lightning AI account with billing set up
- `lightning` CLI installed
- HF token with read access (`hf auth login` before starting)
- Your fork of `keypaa/reap` on GitHub (not upstream CerebrasResearch/reap)
- SSH key added to Lightning AI

---

## Stage 0: CPU Setup (Free Tier, ~30 min)

**Cost: $0** — All on Lightning AI free CPU tier or your local machine.

### 0.1 — Clone and build

```bash
git clone https://github.com/keypaa/reap.git
cd reap
git submodule update --init --recursive
bash scripts/build.sh --v4
source .venv/bin/activate
python -c "from reap.layerwise_prune import main; print('OK')"
python -c "
from reap.v4_block_loader import V4BlockDiskLoader
from reap.v4_moe_observer import DeepseekV4MoEObserver
from reap.v4_prune_utils import prune_v4_model
print('All V4 components OK')
"
```

**What to check:**
- `build.sh --v4` completes without errors (skips CUDA deps deepspeed/vllm)
- All 3 V4 imports work
- If `build.sh` fails: try `pip install git+https://github.com/huggingface/transformers.git` then manually `pip install -e ".[dev]" --no-deps`

### 0.2 — Unit tests

```bash
cd reap && PYTHONPATH="src" python -m pytest tests/test_v4_*.py -v
```

**What to check:** All V4 tests pass (block loader, observer, pipeline dispatch, prune, batched experts). 16 test_observer.py failures are pre-existing (Qwen3 compatibility with transformers 5.12.1) — ignore them, they don't affect V4.

**If tests fail:** Stop. Diagnose. The V4 tests must all pass before proceeding.

### 0.3 — DeepSeek-V2-Lite-Chat smoke test (optional, non-V4 validation)

Only needed if you changed non-V4 pipeline code. If you didn't, skip this.

```bash
python scripts/patch_deepseek.py
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V2-Lite-Chat" \
  --dataset-name "theblackcat102/evol-codealpaca-v1" \
  --batch-size 1 \
  --batches-per-category 2 \
  --model-max-length 256 \
  --prune-method "reap" \
  --n-experts-to-prune 2 \
  --do-eval False \
  --run-observer-only True
```

**What to check:** 27 layers processed without error.

**If it fails:** Likely `DynamicCache.get_usable_length` error — check transformers version. V2 patched file was written for 4.55.0 but `--v4` installs 5.12.1. If the error happens, the fix is in `scripts/patch_deepseek.py` (uses `get_seq_length()` fallback).

---

## Stage 1: GPU Environment (RTX PRO 6000, ~20-30 min)

**Cost: $0 for setup** (GPU time not billed until you run). Download ~160 GB may take 10-20 min.

### 1.1 — Launch GPU instance

```bash
lightning create --spot --accelerator gpu --gpu-type rtx-pro-6000 --name reap-v4
```

**What to expect:** Instance appears in Lightning AI dashboard in ~1-2 minutes. Spot means it can be preempted — save work regularly.

**If launch fails:**
- Check billing setup
- Check spot availability (RTX PRO 6000 may not have spot capacity; try without `--spot`)
- Try a different GPU type (A100-80GB at $2.19/hr would also work but costs more)

### 1.2 — Clone and GPU torch

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/keypaa/reap.git
cd reap
git submodule update --init --recursive
bash scripts/build.sh --v4
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

**Verify CUDA:**
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB")
```
Expected: `CUDA: True, VRAM: 96`

### 1.3 — Download V4 Flash weights (~160 GB)

```bash
hf auth login  # if not already logged in
hf download deepseek-ai/DeepSeek-V4-Flash
```

**What to expect:** Downloads to HF cache. Takes 10-20 min depending on bandwidth. ~160 GB total (FP4+FP8 shards, quantized).

**Verify:**
```python
from reap.v4_block_loader import V4BlockDiskLoader
loader = V4BlockDiskLoader("deepseek-ai/DeepSeek-V4-Flash")
print(f"Layer map has {len(loader.layer_map)} layers")
```
Expected: 43 layers.

**If download fails:**
- Check disk space (`df -h`) — need at least 200 GB free
- Try with `--resume-download` flag
- If bandwidth is slow, you can use `--local-dir` to avoid repeated downloads

### 1.4 — Verify V4 components import with CUDA

```bash
source .venv/bin/activate
python -c "
from reap.v4_block_loader import V4BlockDiskLoader
from reap.v4_moe_observer import DeepseekV4MoEObserver
from reap.v4_prune_utils import prune_v4_model
from reap.layerwise_prune import main
import torch
print(f'All imports OK, device count: {torch.cuda.device_count()}')
"
```

---

## Stage 2: 1-Layer Smoke Test (~$0.05, ~2 min)

**Cost: ~$0.05** (2 min on RTX PRO 6000 at $1.46/hr). If this fails, you wasted a nickel.

This is the FIRST time you touch real GPU compute. Do NOT skip this — it measures real per-layer timing and catches GPU-specific issues before a 43-layer run.

### 2.1 — Smoke test script

The script is in the repo at `scripts/test_v4_one_layer.py` (version-controlled, survives spot preemption).

### 2.2 — Run it

```bash
PYTHONPATH="src" python scripts/test_v4_one_layer.py
```

**What to check:**
- Output shape is `[1, seq_len, 4096]` or similar
- Peak VRAM is well under 90 GB (should be ~14-15 GB)
- The timing output — THIS is your real per-layer cost

**If it fails:**
- "No tensors found for block": safetensors key mismatch — check `v4_block_loader.py` prefix logic
- "Cannot copy out of meta tensor": missing `to_empty()` call in weight loading
- OOM (killed): Something is wrong — with 1 layer and 1 token it should use ~15 GB max. Check for memory leak.

**Decision point:**
- Timing < 10s per layer → full observation will be ~7 min per run (fast!)
- Timing 30-60s per layer → full observation ~22-43 min per run
- Timing > 120s per layer → something is wrong (maybe CPU→GPU transfer bottleneck)

Write down the per-layer timing. You'll use it to estimate Stage 3 costs.

### 2.3 — Actual CPU smoke test results (2026-07-01)

Ran on Lightning AI CPU instance (no GPU) against DeepSeek-V4-Flash layer 0.

**Command:**
```bash
PYTHONPATH="src" python scripts/test_v4_one_layer.py --device cpu --layer 0
```

**Results:**
- Layer loaded successfully: 22 parameters materialized
- Forward + observe: **4.24s** per layer on CPU
- Estimated full 43-layer run: **~3 min** (optimistic, scales with samples × categories)

**Fixes applied to `scripts/test_v4_one_layer.py`:**

| Fix | Line | Why |
|-----|------|-----|
| Added `dtype=torch.bfloat16` to `hidden_3d` randn | 62 | CPU `F.linear` requires matching dtypes (float32 vs bfloat16). CUDA handles mixed-dtype automatically via implicit promotion; CPU does not. |
| Moved `block.to("meta")` after benchmark | 69-81 | Benchmark re-ran `_process_moe_activations` after the block was already moved to meta, producing `Cannot copy out of meta tensor` error. The unload must happen after timing. |

**CPU vs GPU extrapolation:**

| Dataset | Samples | CPU Est. | GPU Target |
|---------|---------|----------|------------|
| seed-10k (3 cats, 10k samples) | ~1,250 batches | **~63 hrs** | ~7-10 min |
| v1-full (10 cats, 23k samples) | ~2,875 batches | **~145 hrs** | ~16-23 min |
| v1-filtered (10 cats, 21k samples) | ~2,625 batches | **~132 hrs** | ~15-21 min |
| structured-outputs (1 cat, 430 samples) | ~54 batches | **~2.7 hrs** | ~30 sec |

**Conclusion:** CPU is impractically slow for full runs. GPU (RTX PRO 6000) is expected to be ~500× faster. Use CPU only for component-level debugging.

### 2.4 — Fixes Applied During GPU Bringup (2026-07-01)

Ran into issues migrating from CPU smoke test to full GPU pipeline. Each fix was committed to `keypaa/reap`.

| # | File | Fix | Why |
|---|------|-----|-----|
| 1 | `v4_block_loader.py:539` | Added `block.to(device)` after `load_state_dict(assign=True)` | `assign=True` replaces CUDA meta tensors with CPU state_dict tensors. Gate weights ended up on CPU, causing `F.linear` device mismatch on GPU. |
| 2 | `data.py:212-222` | Parse `[subset]` from dataset name (e.g. `keypa/reaper-calibration[seed-10k]` → name + subset) | Single-dataset path didn't strip `[subset]` notation, causing `HFValidationError` and registry lookup failure. |
| 3 | `layerwise_prune.py:272` | Set default `chat_template` on tokenizer if missing | DeepSeek V4 tokenizer has no `chat_template`. `ChatDatasetProcessor.apply_chat_template()` fails without one. |
| 4 | `v4_moe_observer.py:119` | Handle `BatchEncoding` in batch type check | Non-vLLM path produces `BatchEncoding` objects (from HF tokenizer), not `dict` or `Tensor`. |
| 5 | `v4_moe_observer.py:87` | Move `embed` to `target_device` | `load_non_backbone_modules` leaves embed on CPU; forward pass on GPU fails with cross-device error. |
| 6 | `v4_moe_observer.py:299` | Keep `total_tokens` on CPU to match state device | `valid_token_mask.sum()` returned CUDA tensor but `self.state[block_idx]["total_tokens"]` was initialized on CPU — `+=` cross-device error. |
| 7 | `v4_block_loader.py:441` | Dequantize FP8 weights in compressor/indexer tensors | `_process_compressor_tensors` loaded `attn.indexer.q_b_proj.weight` raw as `Float8_e4m3fn` without applying its paired `.scale` — forward pass hit `BFloat16 != Float8_e4m3fn` matmul error. |
| 8 | `v4_moe_observer.py:129` | Create 4D causal attention mask instead of 2D | V4 compressor concatenates `block_bias` (4D) to `attention_mask` via `torch.cat(..., dim=-1)` — 2D mask caused `got 2 and 4` RuntimeError in `modeling_deepseek_v4.py:842`. |

---

## Stage 3: Full V4 Flash Observation

**Cost: TBD** — multiply your Stage 2 per-layer time × 43 layers × 4 runs. This is the most expensive stage.

**Goal:** Run observer on all 43 layers with 4 different calibration datasets.

### 3.1 — Important: Reuse your observation output path

After Stage 3 runs, the output directory will be something like:
```
results/DeepSeek-V4-Flash/<dataset-name>/all/layerwise_observer.pt
```

Write down the exact path after the first run. You'll need it in Stage 4.

### 3.2 — Run 1: keypa/reaper-calibration[seed-10k]

```bash
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reaper-calibration[seed-10k]" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --batched-experts \
  --expert-batch-size 16 \
  --run-observer-only True
```

**What to expect:**
- 43 layers processed sequentially
- 3 categories (math, code, agentic), ~10k samples total
- Each category exhausts naturally (~3.3K max / 8 batch-size = ~412 batches)
- 1024 is a safe upper bound — loop stops when samples run out
- VRAM ~37 GB peak (block weights ~14.4 GB + transposed weight buffer ~17 GB + activations ~6 GB) — well within 96 GB
- `--batched-experts` groups expert matmuls into 16-expert batches (~2K kernel launches per forward pass vs ~32K without), trading VRAM for speed

**Monitoring:**
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 30
```
In another terminal:
```bash
tail -f results/DeepSeek-V4-Flash/reaper-calibration/layerwise/observer.log
```

**If it fails mid-run:**
- OOM: Reduce `--batch-size` to 4. If still OOM, reduce to 2.
- Layer X load error: Note which layer. Check if safetensors file for that layer exists.
- Process killed by system (max RAM): Set `--low_cpu_mem_usage True` if available.
- If partial data was saved, `--overwrite-observations True` to restart clean.

### 3.3 — Run 2: keypa/reap-calibration-v1-full

```bash
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reap-calibration-v1-full" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --batched-experts \
  --expert-batch-size 16 \
  --run-observer-only True
```

**What to expect:**
- 10 categories, 23,088 samples total
- Each category exhausts naturally below the 1024 ceiling
- Takes ~2.3× longer than Run 1 due to more categories and samples

### 3.4 — Run 3: keypa/reap-calibration-v1-filtered

```bash
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reap-calibration-v1-filtered" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --batched-experts \
  --expert-batch-size 16 \
  --run-observer-only True
```

**What to expect:**
- Same as Run 2 but 20,980 samples (no refusals/philosophy removed)
- Slightly faster than Run 2 (~10% less data)

### 3.5 — Run 4: 0xSero/structured-outputs-calibration-v1

```bash
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "0xSero/structured-outputs-calibration-v1" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --batched-experts \
  --expert-batch-size 16 \
  --run-observer-only True
```

**What to expect:**
- Single "all" category, 430 samples, no category field
- Will exhaust at ~54 batches (430 / 8 = 53.75)
- Fastest run by far — 430 samples vs 10k-23k for the others

### 3.6 — Verify all observer data

```bash
python -c "
import torch, glob
for f in sorted(glob.glob('results/DeepSeek-V4-Flash/*/all/layerwise_observer.pt')):
    data = torch.load(f)
    print(f'{f.split(\"/\")[3]:40s} layers={len(data)} experts={data[0][\"expert_frequency\"].shape[0]}')
"
```

Expected output:
```
reaper-calibration[seed-10k]          layers=43 experts=256
reap-calibration-v1-full              layers=43 experts=256
reap-calibration-v1-filtered          layers=43 experts=256
structured-outputs-calibration-v1     layers=43 experts=256
```

**If data is wrong:** If any layer has 0 experts or missing keys, re-run that dataset. If all 4 look correct, Stage 3 is done.

**Decision point before Stage 4:** The pruned models will use these observation files to decide which experts to remove. If the observation data is bad, the pruned models will be bad. Verify before proceeding.

---

## Stage 4: Prune + Eval (~24 min prune, eval TBD)

**Cost:** Prune ~$0.58 (24 min ÷ 60 × $1.46/hr for 4 prunes × ~6 min each). Eval cost is TBD — depends on how many benchmarks you run.

### 4.1 — Important: Parameter consistency

Prune commands MUST use the same `--batch-size`, `--batches-per-category`, and `--model-max-length` as the corresponding observation run. The prune step reads cached observer data that was computed with those parameters — if they don't match, the pipeline may fail to find the cached data.

### 4.2 — Prune all 4 models

```bash
# After Run 1 — prune from seed-10k observations
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reaper-calibration[seed-10k]" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --run-observer-only False

# After Run 2 — prune from v1-full observations
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reap-calibration-v1-full" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --run-observer-only False

# After Run 3 — prune from v1-filtered observations
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reap-calibration-v1-filtered" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --run-observer-only False

# After Run 4 — prune from structured-outputs observations
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "0xSero/structured-outputs-calibration-v1" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --run-observer-only False
```

**What to check after each prune:**
- Config shows `n_routed_experts=128` (was 256)
- Config shows `num_local_experts=128`
- Output is saved to `results/DeepSeek-V4-Flash/<dataset-name>/layerwise_0.50_*/`

**If prune fails:**
- "Observer data not found": Check paths. The prune step looks for cached observations from Stage 3 — did you use a different output directory?
- "Mismatched parameters": Error message will say what parameter is wrong — fix and re-run
- OOM during prune: Prune is just weight slicing (~6 min), should use minimal VRAM. If OOM, close other processes

### 4.3 — Evaluate pruned models (optional but recommended)

Upload pruned models to HF first for easier access:
```bash
hf upload <your-repo> results/DeepSeek-V4-Flash/reaper-calibration/layerwise_0.50_*/
```

Then run eval (fill in your actual eval datasets):
```bash
# Eval a pruned model
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "results/DeepSeek-V4-Flash/reaper-calibration/layerwise_0.50_*/" \
  --do-eval True \
  --eval-datasets "user/dataset1,user/dataset2" \
  --eval-batch-size 8 \
  --eval-limit 1000

# Eval original for baseline comparison
PYTHONPATH="src" python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --do-eval True \
  --eval-datasets "user/dataset1,user/dataset2" \
  --eval-batch-size 8 \
  --eval-limit 1000
```

**What to expect:** REAP typically shows <5% degradation at 50% compression.

**If eval fails:** Check that `--do-eval True` loads the pruned model correctly. The V4 block loader should handle FP4→BF16 decompression automatically.

---

## Cost Tracking

| Stage | GPU Minutes | Est. Cost | Notes |
|-------|------------|-----------|-------|
| 0: CPU Setup | 0 | $0 | Free tier |
| 1: GPU Setup | 0 | $0 | Instance not running GPU work yet |
| 2: 1-Layer Test | ~2 min | ~$0.05 | |
| 3a: seed-10k obs | TBD | TBD | Fill from real timing |
| 3b: v1-full obs | TBD | TBD | ~2.3× Run 1 (more categories) |
| 3c: v1-filtered obs | TBD | TBD | ~90% of Run 2 |
| 3d: structured-outputs obs | TBD | TBD | Fast (430 samples) |
| 4a: Prune ×4 | ~24 min | ~$0.58 | |
| 4b: Eval ×4 | TBD | TBD | |
| **Total** | **TBD** | **TBD** | |

To get real GPU time: after Stage 2.2, take `elapsed` seconds, multiply by 43 layers, then by number of runs.

**IMPORTANT:** Lightning AI spot instances can be preempted. If you're in the middle of a 43-layer run and get kicked off, you lose progress (layerwise doesn't save mid-run). Consider:
- Running during low-traffic hours
- Having the resume command ready
- Budgeting for a full-price instance if preemption is frequent

---

## Quick Reference — Stop/Continue Decisions

| Situation | Action |
|-----------|--------|
| V4 unit tests fail | Stop. Fix before GPU. |
| `build.sh --v4` fails on GPU instance | Stop. Try `pip install git+https://github.com/huggingface/transformers.git` |
| CUDA not available after GPU torch install | Stop. Check driver, nvidia-smi. |
| Weight download fails (network) | Retry with `--resume-download`. If disk full, clean up. |
| 1-layer smoke test passes but timing > 120s | Continue, but budget more time. Consider reducing batch-size. |
| 1-layer smoke test OOM | Stop. Something is very wrong (should use ~15 GB). |
| Stage 3 run OOMs at layer 5 | Stop. Reduce `--batch-size` to 4, restart. |
| Stage 3 run OOMs at layer 30 | Monitor VRAM trend. If leak, reduce batch-size and restart. |
| Stage 3 gets preempted mid-run | Restart from scratch. Consider non-spot if preemption is frequent. |
| Stage 4 says "observer data not found" | Stop. Check output paths match between Stage 3 and 4. |
| Pruned model has 256 experts (not 128) | Prune didn't work — check compression-ratio. |
| Eval crashes on model load | Pruned model weights may be corrupted. Re-run prune. |
| Eval shows >10% degradation | Expected for some tasks at 50% compression. Compare all 4 datasets to see which did best. |
