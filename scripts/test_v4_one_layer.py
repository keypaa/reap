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
print("=== PASSED ===")

import time
start = time.time()
observer._process_moe_activations(
    0, moe_block, hidden_states.cuda(), torch.device("cuda"), attention_mask=None,
)
block.to("meta"); gc.collect(); torch.cuda.empty_cache()
elapsed = time.time() - start
print(f"Single forward+observe: {elapsed:.2f}s")
print(f"Estimated 43 layers (no data loading): {elapsed * 43 / 60:.1f} min")
