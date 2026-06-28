# Cloud Deployment Plan: DeepSeek V4 Flash on Lightning AI

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task.

**Goal:** Validate the V4 pruning pipeline end-to-end on Lightning AI GPU, from a 1-layer smoke test through full observation + prune + eval.

**Architecture:** Four stages with escalating cost — local CPU smoke test, cloud GPU 1-layer test, full observation, full prune+eval. Each stage gates the next: don't proceed unless the previous stage passes.

**Tech Stack:** Lightning AI RTX PRO 6000 (96 GB VRAM, 180 GB RAM, $1.46/hr spot), transformers 5.9.0, huggingface_hub, PyTorch 2.5+

**Provider comparison:** Lightning AI ($1.46/hr spot, $2.80/hr on-demand) vs Modal ($3.03/hr). Lightning is cheaper per-GPU. The plan assumes Lightning AI.

## Global Constraints

- No full-model `from_pretrained(device_map="cpu")` — 560 GB BF16 OOMs the 180 GB machine
- All V4 pipeline code uses `V4BlockDiskLoader` for layer-at-a-time loading
- Always verify `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` after each stage
- Pruned model + tokenizer + config must produce coherent `generate()` output
- All costs listed use Lightning AI RTX PRO 6000 pricing


## Stage 0: Local Smoke Test (Pre-Flight)

**Cost:** $0 (local CPU)
**Goal:** Verify our pipeline changes (`_after_forward` callback, observer dispatch) don't break existing non-V4 models.

### Task 0.0: Set up environment

- [x] **Clone repo and init submodules**

```bash
git clone https://github.com/keypaa/reap reap
cd reap
git submodule update --init --recursive
```

- [ ] **Install dependencies**

On Linux (recommended):
```bash
bash scripts/build.sh
```

On Windows (deepspeed doesn't build), install without it:
```bash
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install hatchling editables
uv pip install -e "." --no-build-isolation --no-deps
uv pip install torch transformers datasets tqdm
```

- [ ] **Verify installation**

```bash
uv run python -c "from reap.layerwise_prune import main; print('OK')"
```

### Task 0.1: Run DeepSeek-V2-Lite-Chat end-to-end on CPU

**Files:** N/A — testing existing pipeline only

**Interfaces:**
- Consumes: `python -m reap.layerwise_prune` entry point
- Produces: Evidence that non-V4 path still works

- [ ] **Run layerwise prune on DeepSeek-V2-Lite-Chat**

```bash
cd /path/to/reap
source .venv/bin/activate
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

This runs the observer only (no prune, no eval). On CPU with a 16B model, expect ~5-10 minutes depending on hardware.

- [ ] **Check observer output**

```python
import torch
data = torch.load("results/<model>/all/layerwise_observer.pt", weights_only=False)
print(f"Layers observed: {len(data)}")
for k, v in data[0].items():
    print(f"  {k}: {type(v).__name__} shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
```

Expected: metrics for all MoE layers with correct tensor shapes.

### Task 0.2: Verify full tests pass with local Python

- [ ] **Run full V4 test suite**

```powershell
$env:PYTHONPATH="src"; python -m pytest tests/test_v4_*.py -v
```

Expected: 75/75 pass.

- [ ] **Verify error message for V4 on standard pipeline**

```bash
uv run python -m reap.prune --model-name "deepseek-ai/DeepSeek-V4-Flash" ... 2>&1 | grep "use layerwise_prune"
```

Expected: clear error message saying to use `layerwise_prune` instead.


## Stage 1: Lightning AI Environment Setup

**Cost:** $0 (setup time, no GPU allocated until Stage 2)
**Goal:** Deploy the repo to Lightning AI, install dependencies, verify imports.

### Task 1.1: Create Lightning AI machine

- [ ] **Launch RTX PRO 6000 machine**

Via Lightning AI UI or CLI:
```bash
lightning create --accelerator gpu --gpu-type rtx-pro-6000 --name reap-v4-test
```

Or use spot instance if available:
```bash
lightning create --spot --accelerator gpu --gpu-type rtx-pro-6000 --name reap-v4-spot
```

- [ ] **Clone the repo**

```bash
cd /home/lightning  # or typical Lightning workspace
git clone <repo-url> reap
cd reap
```

- [ ] **Install Python 3.10+ virtual environment**

```bash
uv python pin 3.12
uv venv
source .venv/bin/activate
```

- [ ] **Install dependencies**

```bash
bash scripts/build.sh
```

- [ ] **Verify transformers 5.9.0**

```python
import transformers; print(transformers.__version__)
# Should be 5.9.0+
```

- [ ] **Verify V4 model loads**

```python
from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V4-Flash", trust_remote_code=True)
print(f"Layers: {config.num_hidden_layers}, Experts: {config.n_routed_experts}")
# Expected: Layers: 43, Experts: 256
```

- [ ] **Verify V4BlockDiskLoader imports**

```python
from reap.v4_block_loader import V4BlockDiskLoader
from reap.v4_moe_observer import DeepseekV4MoEObserver
from reap.v4_prune_utils import prune_v4_model
print("All V4 components import successfully")
```

- [ ] **Verify CUDA is available**

```python
import torch; print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB")
# Expected: CUDA: True, VRAM: 96
```

- [ ] **Download V4 Flash weights to cache**

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/latest
```

This downloads ~160 GB. May take 20-40 minutes depending on bandwidth. Run in a `tmux` or `screen` session.

After download, verify:
```python
from reap.v4_block_loader import V4BlockDiskLoader
loader = V4BlockDiskLoader("deepseek-ai/DeepSeek-V4-Flash")
print(f"Layer map has {len(loader.layer_map)} layers")
# Expected: 43 layers
```

## Stage 2: 1-Layer V4 Flash Smoke Test

**Cost:** ~$0.01 (≈30 seconds on RTX PRO 6000 at $1.46/hr)
**Goal:** Load one V4 Flash decoder layer from disk, decompress FP4→BF16, run observer, compute metrics, prune, save. If this works, the entire pipeline is validated.

### Task 2.1: Manual single-layer observer test

- [ ] **Create a standalone single-layer test script**

`/tmp/test_v4_one_layer.py`:
```python
"""Test: load 1 V4 Flash layer, run observer, verify metrics."""
import torch
import gc
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from reap.v4_block_loader import V4BlockDiskLoader
from reap.v4_moe_observer import DeepseekV4MoEObserver
from reap.observer import OBSERVER_CONFIG_REGISTRY
from reap.pruning_metrics import initialize_pruning_state

# 1. Load model on meta
model_name = "deepseek-ai/DeepSeek-V4-Flash"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# 2. Create disk loader and load non-backbone
v4_loader = V4BlockDiskLoader(model_name, config=config)
v4_loader.load_non_backbone_modules(model)

# 3. Load layer 0 from disk
block = model.model.layers[0]
v4_loader.load_into_block(block, 0)

# 4. Move to GPU
block = block.to("cuda")
print(f"Layer 0 on GPU, VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 5. Create a tiny batch (4 tokens)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer("Hello, world!", return_tensors="pt")
input_ids = inputs["input_ids"].cuda()

# 6. Forward through block 0
with torch.no_grad():
    hidden_states = model.model.embed_tokens(input_ids)
    output = block(hidden_states)

print(f"Forward pass OK. Output shape: {output[0].shape}")

# 7. Run observer metrics
moe_block = block.mlp
hook_config = OBSERVER_CONFIG_REGISTRY[model.__class__.__name__]()
observer = DeepseekV4MoEObserver(model, hook_config, v4_loader=v4_loader)

state = initialize_pruning_state(moe_block.experts.num_experts)
observer._process_moe_activations(
    0, moe_block, hidden_states.cuda(),
    torch.device("cuda"), attention_mask=None,
)

print(f"Observer metrics computed: {len(state)} keys")
for k, v in state.items():
    if torch.is_tensor(v):
        print(f"  {k}: shape={v.shape}")
    else:
        print(f"  {k}: {type(v).__name__}")

# 8. Record memory
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Final VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 9. Clean up
block.to("meta")
gc.collect()
torch.cuda.empty_cache()
print(f"After cleanup VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("=== 1-Layer smoke test PASSED ===")
```

- [ ] **Run the single-layer test**

```bash
cd /path/to/reap
PYTHONPATH="src" python /tmp/test_v4_one_layer.py 2>&1
```

Expected output:
- "Forward pass OK. Output shape: torch.Size([1, 4, 4096])"
- Layer 0 VRAM usage: ~13-14 GB
- All observer metrics computed (expert_frequency, ean_sum, reap, etc.)
- Peak VRAM: 14-15 GB (well within 96 GB)
- "=== 1-Layer smoke test PASSED ==="

**If this fails**, debug the specific component (block loader, observer, model structure) and fix before proceeding to Stage 3.

- [ ] **Save evidence**

```bash
mkdir -p .omo/evidence/2026-06-27-one-layer-smoke
cp /tmp/test_v4_one_layer.py .omo/evidence/2026-06-27-one-layer-smoke/
# Record VRAM and test output
```

### Task 2.2: Single-layer pruning test

- [ ] **Test pruning one layer**

Extend the script from 2.1 to also prune layer 0 after observation:

```python
# 10. Prune the MoE of layer 0
from reap.v4_prune_utils import _prune_v4_layer
retained = list(range(128))  # Keep 128 of 256 experts
_prune_v4_layer(moe_block, retained)
print(f"After prune: gate_up_proj shape = {moe_block.experts.gate_up_proj.shape}")
# Expected: torch.Size([128, 4096, 4096])

# 11. Move to meta (simulates the prune_v4_model flow)
block.to("meta")
gc.collect()
```

Expected: weight tensors correctly indexed, 128 experts remaining, no data corruption.

### Task 2.3: Verify pruned model save and reload

- [ ] **Save and reload pruned model**

```python
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    reloaded = AutoModelForCausalLM.from_pretrained(
        tmpdir, device_map="auto", trust_remote_code=True
    )
    print(f"Reloaded pruned model: {reloaded.__class__.__name__}")
    # Verify 128 experts
    print(f"Experts after reload: {reloaded.config.n_routed_experts}")
```

Expected: model saves and reloads successfully with 128 experts.

## Stage 3: Full V4 Flash Observation

**Cost:** ~$0.95 (39 min at $1.46/hr)
**Goal:** Run the full 43-layer observation pass. This is the longest step.

### Task 3.1: Download V4 Flash weights (if not done in Stage 1)

- [ ] **Verify weights are cached**

```bash
ls ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/latest/model.safetensors.index.json
```

### Task 3.2: Run full V4 observation

- [ ] **Run layerwise observation**

```bash
cd /path/to/reap
source .venv/bin/activate

python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "theblackcat102/evol-codealpaca-v1" \
  --batch-size 4 \
  --batches-per-category 64 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --run-observer-only True \
  --low-cpu-mem-usage True
```

**Key parameters:**
- `batch-size 4` — fits VRAM comfortably (14 GB peak per layer)
- `batches-per-category 64` — sufficient for REAP metrics (12,288 tokens total: 64 × 4 × 16,384 × ~0.75 fill)
- `model-max-length 16384` — V4's native context (though training was 8K)
- `low-cpu-mem-usage True` — avoids `from_pretrained(device_map="cpu")` OOM
- `run-observer-only True` — stops after observation, no prune yet

**Expected runtime:** ~39 min (43 layers × ~50-55 seconds per layer)

**Monitoring:** Watch VRAM drift across layers:
```bash
# In a separate terminal, sample every 30 seconds:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 30
```

Expected: VRAM stable at 14-16 GB throughout, no OOM across 43 layers.

- [ ] **Save intermediate metrics snapshot**

After observation completes, the observer saves to:
```
results/DeepSeek-V4-Flash/evol-codealpaca-v1/all/layerwise_observer.pt
```

Verify:
```python
import torch
data = torch.load("results/DeepSeek-V4-Flash/evol-codealpaca-v1/all/layerwise_observer.pt")
print(f"Layers observed: {len(data)}")
print(f"Expert count: {data[0]['expert_frequency'].shape[0]}")
print(f"Sample layer 0: reap={data[0]['reap'][:5]}")
```

Expected: 43 MoE layers with 256 experts each, non-zero REAP scores.

- [ ] **Save evidence**

```bash
mkdir -p .omo/evidence/2026-06-27-full-observation
cp results/DeepSeek-V4-Flash/evol-codealpaca-v1/all/layerwise_observer.pt \
   .omo/evidence/2026-06-27-full-observation/
nvtop --output .omo/evidence/2026-06-27-full-observation/gpu-log.csv
```


## Stage 4: Full V4 Flash Prune + Eval

**Cost:** ~$0.15 (6 min at $1.46/hr) for prune, plus eval costs
**Goal:** Prune the model to a target compression ratio and evaluate.

### Task 4.1: Run pruning

- [ ] **Run prune (not observer-only)**

From the same results directory, run prune (assumes observation from Stage 3 exists):

```bash
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "theblackcat102/evol-codealpaca-v1" \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --run-observer-only False \
  --overwrite-pruned-model True
```

This reuses the cached observer data from Stage 3 and runs only the prune step.

**Key parameter:** `--compression-ratio 0.5` — prune 50% of experts (128 of 256).

**Expected runtime:** ~6 min (43 layers × ~5-8 seconds per layer for load → prune → unload cycle).

- [ ] **Verify pruned model**

```bash
ls results/DeepSeek-V4-Flash/evol-codealpaca-v1/layerwise_0.50_*/pytorch_model*.bin
```

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("results/DeepSeek-V4-Flash/evol-codealpaca-v1/layerwise_0.50_*/")
print(f"Retained experts: {config.n_routed_experts}")
print(f"Num local experts: {config.num_local_experts}")
```

Expected: `n_routed_experts = 128`, `num_local_experts = 128`.

- [ ] **Save evidence**

```bash
mkdir -p .omo/evidence/2026-06-27-pruned-model
# Record model size
du -sh results/DeepSeek-V4-Flash/evol-codealpaca-v1/layerwise_*.50*/
```

### Task 4.2: Run evaluation

- [ ] **Configure eval datasets**

Edit eval args to point to the user's evaluation datasets:
```bash
python -m reap.layerwise_prune \
  --model-name "results/DeepSeek-V4-Flash/evol-codealpaca-v1/layerwise_0.50_*/" \
  --do-eval True \
  --eval-datasets "user/dataset1,user/dataset2" \
  --eval-batch-size 8 \
  --eval-limit 1000
```

- [ ] **Compare against original model**

If possible, run the same eval tasks on the unpruned model for a baseline:
```bash
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --do-eval True \
  --eval-datasets "user/dataset1,user/dataset2" \
  --eval-batch-size 8 \
  --eval-limit 1000
```

Compare metrics between pruned and unpruned. Expected: <5% degradation at 50% compression (typical for REAP).


## Cost Summary

| Stage | Duration | GPU Type | Cost |
|-------|----------|----------|------|
| 0: Local Smoke | ~10 min | CPU | $0 |
| 1: Environment | ~60 min (+160 GB download) | CPU | $0 |
| 2: 1-Layer Test | ~1 min | RTX PRO 6000 | ~$0.02 |
| 3: Full Observation | ~39 min | RTX PRO 6000 | ~$0.95 |
| 4a: Prune | ~6 min | RTX PRO 6000 | ~$0.15 |
| 4b: Eval | ~15-30 min | RTX PRO 6000 | ~$0.36-0.73 |
| **Total** | **~2 hrs GPU** | | **~$1.50-1.85** |

At on-demand pricing ($2.80/hr): ~$3.50 total.

## Rollback / Recovery

If any stage fails:
- **Stage 1 fail** (environment): Open Lightning AI issue or try Modal as fallback
- **Stage 2 fail** (1-layer): Debug the specific component — check `nvidia-smi` for CUDA errors, verify weights load correctly
- **Stage 3 fail** (full observation): Usually memory — check if VRAM drifts up across layers, may need `gc.collect()` tuning
- **Stage 4 fail** (prune): Save checkpoint before prune, verify observer data integrity

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Lightning AI RTX PRO 6000 not available | Delay | Fall back to Modal RTX PRO 6000 ($3.03/hr) or L40S ($2.14/hr) |
| 160 GB download times out | Delay | Use `--resume-download` flag; download overnight |
| CUDA OOM after 40+ layers | Lost observation time | Implement periodic `torch.cuda.empty_cache()`; reduce batch size to 2 |
| Pruned model can't be loaded | Wasted prune | Test reload immediately after save (Task 2.3) |
| Eval metrics worse than expected | Unsatisfactory result | Try different compression ratios (0.25, 0.33, 0.5); preserve super-experts via `--preserve-outliers` |
