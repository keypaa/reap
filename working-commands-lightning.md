# Working Commands — REAP Layerwise on Lightning (DeepSeek-V4-Flash)

**Date:** 2026-06-26  
**Hardware:** L4 24 GB VRAM, 32 GB RAM, ~306 GB disk  
**Model:** deepseek-ai/DeepSeek-V4-Flash (284B/13B activated, ~145 GB on disk)

---

## 1. Environment Setup

```bash
# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Init submodules (needed for third-party eval deps)
ls third-party/*/
git submodule update --init --recursive

# Install package + deps
uv pip install --python .venv/bin/python -e ".[dev]"

# Verify
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count()); from transformers import AutoConfig; print('transformers OK')"
# Expected: torch 2.7.1+cu126, CUDA available: True, CUDA devices: 1, transformers OK
```

## 2. Upgrade Transformers (required for deepseek_v4)

```bash
pip install git+https://github.com/huggingface/transformers.git "huggingface_hub>=0.34.0"
```

## 3. Download Model Weights

```bash
python -c "
from huggingface_hub import snapshot_download
print('Downloading DeepSeek-V4-Flash weights (~145 GB)...')
snapshot_download('deepseek-ai/DeepSeek-V4-Flash')
print('Download complete')
"
```
Takes ~7-10 min at ~250 MB/s.

## 4. Patch Source Files (meta-tensor loading + safetensors key matching)

DeepSeek-V4-Flash has two quirks that require patches:

### Quirk A: safetensors keys drop the `model.` prefix
PyTorch modules are `model.layers.0.*` but safetensors keys are `layers.0.*`. Same for non-backbone modules: `model.embed_tokens` → `embed`, `model.norm` → `norm`, `lm_head` → `head`.

### Quirk B: Some block params are not in safetensors (quantized weights)
`load_state_dict(strict=False, assign=True)` leaves unmatched params on meta device, and `block.to(device)` crashes. Need `to_empty()` first.

### Required patches

Copy these 4 files from local Windows to Lightning instance:

```bash
scp src/reap/layerwise_model_utils.py s_XXX@ssh.lightning.ai:/teamspace/studios/this_studio/reap/src/reap/
scp src/reap/layerwise_observer.py s_XXX@ssh.lightning.ai:/teamspace/studios/this_studio/reap/src/reap/
scp src/reap/model_util.py s_XXX@ssh.lightning.ai:/teamspace/studios/this_studio/reap/src/reap/
```

**What each patch does:**

| File | Change |
|------|--------|
| `layerwise_model_utils.py:108-170` | `load_block_weights_from_safetensors` — try progressively shorter prefixes (`model.layers.0` → `layers.0`) to match safetensors keys; use `to_empty()` + re-assign instead of `.to(device)` for meta-safe device placement |
| `layerwise_observer.py:296-340` | `_load_non_backbone_weights` — same prefix stripping for non-backbone modules; fallback to model-specific `NON_BACKBONE_KEY_MAP` when stripping doesn't match (e.g., `embed_tokens` → `embed`) |
| `model_util.py:120-126` | Added `NON_BACKBONE_KEY_MAP` dict mapping safetensor prefixes to PyTorch module names for `DeepseekV4ForCausalLM` |

## 5. Run 1 — keypa seed-10k (~45 min)

```bash
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "keypa/reaper-calibration" \
  --dataset-config-name "seed-10k" \
  --split "train" \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --batch-size 4 \
  --batch-group-size 80 \
  --batches-per-category 128 \
  --model-max-length 4096 \
  --output-file-name "keypa-seed10k-v4flash.pt" \
  --overwrite-observations True \
  --low_cpu_mem_usage True \
  --save_intermediate True \
  --do-eval false
```

**Expected flow:**
```
Added 128 samples from category: all
Total calibration samples: 128
Loading model skeleton for deepseek-ai/DeepSeek-V4-Flash on meta device...
Model loaded: DeepseekV4ForCausalLM
Recording activations using layerwise processing...
Found transformer blocks container: model.layers with 43 blocks
Processing 43 blocks across 2 batch groups of up to 80 batches
Block model.layers.0 weights loaded from disk
Seeding replay cache from the first decoder block  ← ~2 min
Processing block 1/43: model.layers.0               ← ~3 min per block
...
```
    
## 6. Run 2 — Sero 44K (~2-4 hours)

```bash
python -m reap.layerwise_prune \
  --model-name "deepseek-ai/DeepSeek-V4-Flash" \
  --dataset-name "0xSero/reap-calibration-data-v1" \
  --split "train" \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --batch-size 4 \
  --batch-group-size 80 \
  --batches-per-category 3500 \
  --model-max-length 4096 \
  --output-file-name "sero-full44k-v4flash.pt" \
  --overwrite-observations True \
  --low_cpu_mem_usage True \
  --save_intermediate True \
  --do-eval false
```

---

## 7. Add Observer Config for DeepSeek-V4

Model loaded with meta-tensor mode, but hit a second error:
```
ValueError: No observer configuration for model 'DeepseekV4ForCausalLM'.
```

The `OBSERVER_CONFIG_REGISTRY` in `src/reap/observer.py` has entries for `DeepseekV2ForCausalLM` but not V4.

### Investigation — Find MoE Module Attributes

The observer uses `num_experts_attr_name` and `top_k_attr_name` to read expert count and top-k values from the MoE module via `getattr(module, attr_name)`. These must be actual attributes on the module object.

**Step 1: Confirm config values exist**
```bash
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V4-Flash', trust_remote_code=True)
print('n_routed_experts:', config.n_routed_experts)
print('num_experts_per_tok:', config.num_experts_per_tok)
print('num_local_experts (alias):', config.num_local_experts)
"
```
```
n_routed_experts: 256
num_experts_per_tok: 6
num_local_experts (alias): 256
```

**Step 2: List MoE module class and int attributes**
```bash
python -c "
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
config = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V4-Flash', trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
for name, mod in model.named_modules():
    if 'SparseMoe' in type(mod).__name__:
        print(f'Module: {name} -> {type(mod).__name__}')
        for attr in sorted(dir(mod)):
            if not attr.startswith('_'): 
                try:
                    val = getattr(mod, attr)
                    if isinstance(val, int) or isinstance(val, bool):
                        print(f'  self.{attr} = {val}')
                except: pass
        break
"
```
```
Module: model.layers.0.mlp -> DeepseekV4SparseMoeBlock
  call_super_init: False
  dump_patches: False
  is_hash: True    # <-- layer 0 uses hash routing, not learned routing
  training: True
```
**Finding:** `DeepseekV4SparseMoeBlock` has no `num_experts`, `n_routed_experts`, or `top_k` directly. It has a `gate` submodule.

**Step 3: Inspect the gate submodule**
```bash
python -c "
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
config = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V4-Flash', trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
for name, mod in model.named_modules():
    if 'SparseMoe' in type(mod).__name__:
        gate = getattr(mod, 'gate', None)
        if gate is not None:
            print(f'  gate type: {type(gate).__name__}')
            for attr in sorted(dir(gate)):
                if not attr.startswith('_'):
                    try:
                        val = getattr(gate, attr)
                        if isinstance(val, int):
                            print(f'    gate.{attr} = {val}')
                    except: pass
        break
"
```
```
gate type: DeepseekV4HashRouter
  gate.num_experts = 256
  gate.top_k = 6
  gate.call_super_init = False
  gate.dump_patches = False
  gate.hidden_dim = 4096
  gate.training = True
```
**Finding:** The `gate` submodule has both `num_experts` (256) and `top_k` (6).

**Step 4: Verify layer 3 (non-hash MoE) has same structure**
```bash
python -c "
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
config = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V4-Flash', trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
for name, mod in model.named_modules():
    if 'layers.3.mlp' in name and 'SparseMoe' in type(mod).__name__:
        gate = mod.gate
        print(f'Layer 3 gate type: {type(gate).__name__}')
        print(f'  gate.num_experts = {gate.num_experts}')
        print(f'  gate.top_k = {gate.top_k}')
        break
"
```
Expected output:
```
Layer 3 gate type: DeepseekV4TopKRouter
  gate.num_experts = 256
  gate.top_k = 6
```

**Conclusion:** Both hash-router and learned-router layers use `gate.num_experts` and `gate.top_k`. The observer's `reduce(getattr, ...)` path handles the dot notation — so `"gate.num_experts"` chains `getattr(mod, "gate")` then `getattr(gate, "num_experts")`.

### Patch `src/reap/observer.py`

Add before the `OBSERVER_CONFIG_REGISTRY` dict (after the Glm44MoEObserverHookConfig block):

```python
@dataclass
class DeepseekV4MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "DeepseekV4SparseMoeBlock"
    num_experts_attr_name: str = "gate.num_experts"
    top_k_attr_name: str = "gate.top_k"
    fused_experts: bool = False
```

And add to the registry:
```python
OBSERVER_CONFIG_REGISTRY = {
    ...
    "DeepseekV4ForCausalLM": DeepseekV4MoEObserverHookConfig,
}
```

### Apply the patch

Locally, on your Windows machine:
```bash
# Edit src/reap/observer.py to add the new config class
# Then scp it to the instance
scp src/reap/observer.py s_01kvnf22jcnc02d8814freks7k@ssh.lightning.ai:/teamspace/studios/this_studio/reap/src/reap/observer.py
```

Or directly edit on the instance using a text editor (vim/nano).

## Troubleshooting

| Problem | Fix |
|---------|------|
| `ValueError: The checkpoint ... has model type deepseek_v4 but Transformers does not recognize this architecture` | Run `pip install git+https://github.com/huggingface/transformers.git "huggingface_hub>=0.34.0"` |
| `third-party/evalplus does not appear to be a Python project` | Run `git submodule update --init --recursive` before install |
| `scripts/build.sh: $'\r': command not found` | Don't use build.sh — use `uv pip install` directly |
| Process killed by OOM (RAM full) | Must use `--low_cpu_mem_usage True` + `--batch-group-size 80` |
| `No observer configuration for model 'DeepseekV4ForCausalLM'` | Add `DeepseekV4ForCausalLM` → `DeepseekV4MoEObserverHookConfig` entry to `OBSERVER_CONFIG_REGISTRY` in `observer.py` |
| `No tensors found for block 'model.layers.0' in weight index` | Apply `layerwise_model_utils.py` patch (prefix stripping) |
| `Cannot copy out of meta tensor; no data!` | Apply `layerwise_model_utils.py` patch (use `to_empty()` instead of `.to(device)`) |
| `No weights found for non-backbone module: model.embed_tokens` | Update `NON_BACKBONE_KEY_MAP` in `model_util.py` |

