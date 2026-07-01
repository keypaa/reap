# Cloud Deployment Plan: DeepSeek V4 Flash on Lightning AI

**Goal:** Validate the V4 pruning pipeline end-to-end on Lightning AI GPU, from a 1-layer smoke test through full observation + prune + eval.

**Architecture:** Five stages — a free-tier CPU setup + validation, then GPU stages with escalating cost. Each stage gates the next.

**Tech Stack:** Lightning AI, transformers 5.13.0.dev0 (git main for V4), huggingface_hub 1.21.0, PyTorch 2.5+

**Provider:** Lightning AI. Use free CPU tier for setup, paid GPU only for execution.

## Two Install Modes

| Mode | Command | Use Case | Transformers |
|------|---------|----------|-------------|
| **Standard** | `bash scripts/build.sh` | Non-V4 models (Qwen3, Llama4, Mixtral, DeepSeek-V2) | 4.55.0 (pinned) |
| **V4** | `bash scripts/build.sh --v4` | DeepSeek V4 Flash / Pro | 5.13.0.dev0 (git main) |

The `--v4` flag skips CUDA-only deps (deepspeed, vllm) and installs transformers from git main + torch CPU. Switch to GPU torch when moving to a paid GPU instance.

## Global Constraints

- No full-model `from_pretrained(device_map="cpu")` — 560 GB BF16 OOMs the 180 GB machine
- All V4 pipeline code uses `V4BlockDiskLoader` for layer-at-a-time loading
- Pruned model + tokenizer + config must produce coherent `generate()` output
- All GPU costs use Lightning AI RTX PRO 6000 spot pricing ($1.46/hr)

## Stage 0: Free-Tier CPU Setup & Validation

**Cost:** $0
**Goal:** Set up environment, verify code imports, run tests.

### Task 0.0: Clone and install

- [x] **Clone repo and init submodules**
```bash
git clone https://github.com/keypaa/reap.git
cd reap
git submodule update --init --recursive
```

- [x] **Install dependencies (V4 mode)**
```bash
bash scripts/build.sh --v4
```

This creates a venv, installs the package with `--no-deps` (skips deepspeed/vllm CUDA build), then installs transformers from git main and torch CPU.

- [x] **Verify installation**
```bash
source .venv/bin/activate
python -c "from reap.layerwise_prune import main; print('OK')"
```

- [x] **Verify V4 components import**
```bash
python -c "
from reap.v4_block_loader import V4BlockDiskLoader
from reap.v4_moe_observer import DeepseekV4MoEObserver
from reap.v4_prune_utils import prune_v4_model
print('All V4 components import successfully')
"
```

### Task 0.1: Run non-V4 smoke test (DeepSeek-V2-Lite-Chat on CPU)

**Goal:** Verify pipeline changes (`_after_forward` callback, observer dispatch) don't break existing non-V4 models.

**Prerequisite:** Download patched model files.
```bash
pip install huggingface_hub
python scripts/patch_deepseek.py
```

This downloads DeepSeek-V2-Lite-Chat weights (~30 GB) to `artifacts/models/` and patches the modeling file.

- [x] **Run layerwise observer on CPU**
```bash
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

Expected: 27 layers processed, observer data saved to `artifacts/DeepSeek-V2-Lite-Chat/.../layerwise/observations_1024_cosine.pt`.

**Known issue:** First run may hit `DynamicCache.get_usable_length` error if transformers != 4.55.0 (the V2 patched model file was written for 4.55.0). The `build.sh --v4` installs transformers from git main (5.13.0.dev0) where this method exists — no conflict. If using standard `build.sh` (4.55.0), the modeling file is fixed to use `get_seq_length()` instead.

- [x] **Verify observer data**
```python
import torch
data = torch.load("artifacts/DeepSeek-V2-Lite-Chat/evol-codealpaca-v1/layerwise/observations_1024_cosine.pt")
print(f"Layers observed: {len(data)}")
```

Expected: metrics for all 27 MoE layers.

### Task 0.2: Run unit tests

- [x] **Run V4 test suite**
```bash
cd reap && PYTHONPATH="src" python -m pytest tests/test_v4_*.py -v
```

Expected: 75/75 pass (already verified locally).

- [x] **Verify V4 guard on standard pipeline** (GPU instance only — `prune.py` imports `vllm` which needs CUDA)

The guard is in `main.py`: if `_is_v4_model(model_name)`, it raises `RuntimeError("use layerwise_prune")`. Verified by source inspection.

## Stage 1: GPU Environment (RTX PRO 6000)

**Cost:** Storage transfer may apply (~$4-5 if moving between cloud regions); GPU time $0 until activated.
**Goal:** Set up paid GPU instance, install GPU torch, download V4 weights.

### Task 1.1: Launch GPU machine

- [ ] **Launch RTX PRO 6000 instance**
```bash
lightning create --spot --accelerator gpu --gpu-type rtx-pro-6000 --name reap-v4
```

### Task 1.2: Clone and set up

- [ ] **Clone repo**
```bash
cd /teamspace/studios/this_studio
git clone https://github.com/keypaa/reap.git
cd reap
git submodule update --init --recursive
```

- [ ] **Install (V4 mode)**
```bash
bash scripts/build.sh --v4
source .venv/bin/activate
```

- [ ] **Install GPU torch (overwrites CPU torch)**
```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

- [ ] **Verify CUDA**
```python
import torch; print(f"CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB")
# Expected: CUDA: True, VRAM: 96
```

### Task 1.3: Download V4 Flash weights

- [ ] **Download to cache (~160 GB, 10-20 min)**
```bash
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/latest
```

- [ ] **Verify download**
```python
from reap.v4_block_loader import V4BlockDiskLoader
loader = V4BlockDiskLoader("deepseek-ai/DeepSeek-V4-Flash")
print(f"Layer map has {len(loader.layer_map)} layers")
# Expected: 43 layers
```

## Stage 2: 1-Layer V4 Flash Smoke Test

**Cost:** ~$0.05 (≈2 min on RTX PRO 6000 spot at $1.46/hr)
**Goal:** Load one V4 Flash decoder layer from disk, decompress FP4→BF16, run observer, compute metrics, prune, save.

### Task 2.1: Single-layer observer test

- [ ] **Create and run standalone test script**

```python
"""test_v4_one_layer.py — load 1 V4 layer from disk, run observer."""
import torch, gc
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from reap.v4_block_loader import V4BlockDiskLoader
from reap.v4_moe_observer import DeepseekV4MoEObserver
from reap.observer import OBSERVER_CONFIG_REGISTRY
from reap.pruning_metrics import initialize_pruning_state

model_name = "deepseek-ai/DeepSeek-V4-Flash"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

v4_loader = V4BlockDiskLoader(model_name, config=config)
v4_loader.load_non_backbone_modules(model)

block = model.model.layers[0]
v4_loader.load_into_block(block, 0)
block = block.to("cuda")
print(f"Layer 0 on GPU, VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer("Hello, world!", return_tensors="pt")
input_ids = inputs["input_ids"].cuda()

with torch.no_grad():
    hidden_states = model.model.embed_tokens(input_ids)
    output = block(hidden_states)
print(f"Forward pass OK. Output shape: {output[0].shape}")

moe_block = block.mlp
hook_config = OBSERVER_CONFIG_REGISTRY[model.__class__.__name__]()
state = initialize_pruning_state(moe_block.experts.num_experts)
observer = DeepseekV4MoEObserver(model, hook_config, v4_loader=v4_loader)
observer._process_moe_activations(
    0, moe_block, hidden_states.cuda(), torch.device("cuda"), attention_mask=None,
)
print(f"Observer metrics computed: {len(state)} keys")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

block.to("meta"); gc.collect(); torch.cuda.empty_cache()
print("=== 1-Layer smoke test PASSED ===")
```

- [ ] **Run it**
```bash
PYTHONPATH="src" python /tmp/test_v4_one_layer.py
```

Expected: forward pass OK, peak VRAM ~14-15 GB, all metrics computed.

### Task 2.2: Single-layer pruning test

- [ ] **Extend the script to prune layer 0 after observation**
```python
from reap.v4_prune_utils import _prune_v4_layer
retained = list(range(128))  # Keep 128 of 256
_prune_v4_layer(moe_block, retained)
print(f"After prune: gate_up_proj shape = {moe_block.experts.gate_up_proj.shape}")
# Expected: torch.Size([128, 4096, 4096])
```

### Task 2.3: Verify pruned model save and reload

- [ ] **Save and reload pruned model**
```python
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    reloaded = AutoModelForCausalLM.from_pretrained(tmpdir, device_map="auto", trust_remote_code=True)
    print(f"Experts after reload: {reloaded.config.n_routed_experts}")
```
Expected: 128 experts.

## Stage 3: Full V4 Flash Observation

**Cost:** TBD — measure per-layer time in Stage 2 first, then multiply by 43 layers × 3 runs
**Goal:** Run all 43 layers with the full calibration dataset.

### Task 3.1: Verify weights cached

- [ ] **Check cache**
```bash
ls ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/latest/model.safetensors.index.json
```

### Task 3.2: Run observation

- [ ] **Run layerwise observation (4 separate runs — one per dataset)**

```bash
# Run 1: your dataset (10k subset, 3 categories)
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reaper-calibration[seed-10k]" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --run-observer-only True

# Run 2: Sero's full (10 categories, includes refusals)
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reap-calibration-v1-full" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --run-observer-only True

# Run 3: Sero's filtered (10 categories, no refusals)
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reap-calibration-v1-filtered" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --run-observer-only True

# Run 4: Structured outputs (430 samples, single "all" category)
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "0xSero/structured-outputs-calibration-v1" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --run-observer-only True
```

**Parameters:** batch-size 8 (~14 GB VRAM/layer), batches-per-category 1024 (safe upper bound — loop draws without replacement per category, stops when samples exhaust), model-max-length 16384 (V4 native context), run-observer-only True.

**Cost & runtime:** Unknown. No real V4 Flash benchmarks on this GPU exist. The per-layer time depends on category count (more categories = more forward passes per layer). Estimate 10-60 min per run but only measurement will tell.

- [ ] **Monitor VRAM**
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 30
```
Expected: stable ~14-16 GB.

- [ ] **Verify observer data**
```python
import torch
data = torch.load("results/DeepSeek-V4-Flash/evol-codealpaca-v1/all/layerwise_observer.pt")
print(f"Layers: {len(data)}, Experts: {data[0]['expert_frequency'].shape[0]}")
```

## Stage 4: Full V4 Flash Prune + Eval

**Cost:** Prune ~$0.44 (18 min × 3 at $1.46/hr). Eval TBD.
**Goal:** Prune 50% of experts and evaluate.

### Task 4.1: Run pruning

- [ ] **Prune each dataset (reuses cached observer data from matching observation run)**

```bash
# After Run 1
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reaper-calibration[seed-10k]" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --run-observer-only False

# After Run 2
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reap-calibration-v1-full" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --run-observer-only False

# After Run 3
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reap-calibration-v1-filtered" \
  --batch-size 8 \
  --batches-per-category 1024 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --run-observer-only False

# After Run 4
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "0xSero/structured-outputs-calibration-v1" \
  --batch-size 8 \
  --batches-per-category 500 \
  --model-max-length 16384 \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --run-observer-only False
```

**Expected:** ~6 min, config shows `n_routed_experts=128`, `num_local_experts=128`.

### Task 4.2: Run evaluation

- [ ] **Eval pruned model**
```bash
python -m reap.layerwise_prune \
  --model-name "results/DeepSeek-V4-Flash/evol-codealpaca-v1/layerwise_0.50_*/" \
  --do-eval True \
  --eval-datasets "user/dataset1,user/dataset2" \
  --eval-batch-size 8 \
  --eval-limit 1000
```

- [ ] **Eval original for baseline**
```bash
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --do-eval True \
  --eval-datasets "user/dataset1,user/dataset2" \
  --eval-batch-size 8 \
  --eval-limit 1000
```

Compare: expected <5% degradation at 50% compression (typical for REAP).

## Cost Summary

| Stage | Notes | GPU Time | Cost ($1.46/hr) |
|-------|-------|----------|-----------------|
| 0: CPU Setup & Test | Free tier | — | $0 |
| 1: GPU Setup + 160 GB download | Varies | — | $0 |
| 2: 1-Layer Test | ~1 min | ~2 min | ~$0.05 |
| 3a: Observation Run 1 (seed-10k, 3 cats) | Unknown — needs measurement | TBD | TBD |
| 3b: Observation Run 2 (v1-full, 10 cats) | Unknown — needs measurement | TBD | TBD |
| 3c: Observation Run 3 (v1-filtered, 10 cats) | Unknown — needs measurement | TBD | TBD |
| 3d: Observation Run 4 (structured-outputs, 1 cat) | 430 samples, fast | TBD | TBD |
| 4a: Prune (×4) | CPU-only, cheap | ~24 min | ~$0.58 |
| 4b: Eval (×3 models) | Unknown — depends on eval size | TBD | TBD |
| **Total** | **Will fill after Stage 2 benchmark** | **TBD** | **TBD** |

\* Storage transfer between cloud regions may add ~$4-5.

## Environment Split: CPU Free Tier vs GPU Paid Instance

| Aspect | Free CPU Tier (Stage 0) | Paid GPU Instance (Stages 1-4) |
|--------|------------------------|--------------------------------|
| **Purpose** | Setup, code fix, unit tests | V4 observation, prune, eval |
| **torch** | CPU-only (`--index-url .../cpu`) | CUDA (`--index-url .../cu128`) |
| **vllm** | Not installed | Installed on GPU |
| **deepspeed** | Not installed | Installed on GPU |
| **transformers** | 5.13.0.dev0 (git main) | 5.13.0.dev0 (git main) |
| **V4 weights** | Not downloaded | ~160 GB downloaded |
| **Cost** | $0 | TBD (Stage 2 benchmark needed) |

## Available Datasets

Four calibration datasets are registered in the pipeline:

| Dataset | Samples | Category | Fields | Format |
|---------|---------|----------|--------|--------|
| `keypa/reaper-calibration[seed-10k]` | 10,000 | `math`, `code`, `agentic` | `messages`, `category` | Chat (role/content pairs) |
| `keypa/reaper-calibration[specialist-300k]` | 300,000 | `math`, `code`, `agentic` | `messages`, `category` | Chat |
| `keypa/reaper-calibration[production-800k]` | 800,000 | `math`, `code`, `agentic` | `messages`, `category` | Chat |
| `keypa/reap-calibration-v1-full` | 23,088 | 10 domains (includes refusals) | `messages`, `category` | Flat text → user message |
| `keypa/reap-calibration-v1-filtered` | 20,980 | 10 domains (no refusals) | `messages`, `category` | Flat text → user message |
| `0xSero/structured-outputs-calibration-v1` | 430 | None | `text` | Flat text with role labels |

**Composite spec examples:**
```bash
# Mix sizes and domains
--dataset-name "keypa/reaper-calibration[seed-10k]:200,keypa/reap-calibration-v1-full:200"

# All three datasets combined
--dataset-name "keypa/reaper-calibration[seed-10k]:100,keypa/reap-calibration-v1-full:100,0xSero/structured-outputs-calibration-v1:50"
```

## Known Issues & Workarounds

| Issue | Symptom | Fix |
|-------|---------|-----|
| `DynamicCache.get_usable_length` not found | `AttributeError` on DeepSeek-V2-Lite-Chat forward | Use `build.sh` (pins 4.55.0) or use `--v4` (5.13.0.dev0 has it). Source file fixed with `get_seq_length()` fallback |
| deepspeed/vllm won't build on CPU | Build error on free tier | Use `bash scripts/build.sh --v4` to skip CUDA deps |
| `AutoTokenizer.from_pretrained` fails | `HFValidationError` on local path | Run `python scripts/patch_deepseek.py` first to populate `artifacts/models/` |
| `LossKwargs` import error | `ImportError` on ERNIE-4.5 | Only affects ERNIE patched model; not relevant for V4 |
| `vllm` not found on CPU | `ModuleNotFoundError` for `prune.py` | Expected on CPU tier — `prune.py` needs CUDA. Use `layerwise_prune.py` for V4 on GPU |
| `prune.py --help` fails without vllm | Can't verify V4 guard output | Guard verified by source code inspection in `main.py` |

## Rollback / Recovery

- **Stage 0 fail** (setup): Check transformers version compatibility; ensure submodules initialized
- **Stage 1 fail** (GPU env): Verify CUDA torch installed correctly; check `nvidia-smi`
- **Stage 2 fail** (1-layer): Debug component in isolation — check FP4 decompression, observer hooks, prune indexing
- **Stage 3 fail** (observation): Usually memory — reduce batch-size to 2; add periodic `torch.cuda.empty_cache()`
- **Stage 4 fail** (prune): Verify observer data integrity before pruning; save intermediate checkpoints
