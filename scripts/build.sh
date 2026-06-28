#!/bin/bash
set -euo pipefail

V4_MODE=false
if [[ "${1:-}" == "--v4" ]]; then
    V4_MODE=true
fi

git submodule init
git submodule update
uv venv .venv --seed --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install setuptools wheel

if [ "$V4_MODE" = true ]; then
    # V4 mode: skip CUDA deps, install transformers from git (needs >=5.9.0)
    uv pip install --editable . --no-deps -vv
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install git+https://github.com/huggingface/transformers.git "huggingface_hub>=0.34.0"
    uv pip install accelerate datasets matplotlib seaborn tqdm numpy scipy
else
    # Full install with CUDA deps (deepspeed, vllm, etc.)
    VLLM_USE_PRECOMPILED=1 uv pip install --editable . -vv --torch-backend auto
fi

# For Ernie4-5, uncomment the below:
# .venv/bin/python scripts/patch_ernie4_5.py

# for Llama4 add this to vllm.model_executor.models.registry:_TEXT_GENERATION_MODELS in alphabetical order:
# "Llama4ForCausalLM": ("llama4", "Llama4ForCausalLM"),