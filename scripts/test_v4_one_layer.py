"""test_v4_one_layer.py — load 1 V4 layer from disk, run observer on CPU.

Usage:
  python scripts/test_v4_one_layer.py
  python scripts/test_v4_one_layer.py --device cuda

Requires the first shard of DeepSeek-V4-Flash downloaded locally.
Download with: huggingface-cli download deepseek-ai/DeepSeek-V4-Flash
"""
import argparse
import gc
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reap.v4_block_loader import V4BlockDiskLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading config for {args.model}...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    print("Creating block disk loader...")
    v4_loader = V4BlockDiskLoader(args.model, config=config)
    v4_loader.load_non_backbone_modules(model)

    layer_idx = args.layer
    print(f"Loading layer {layer_idx} from disk...")
    block = model.model.layers[layer_idx]
    v4_loader.load_into_block(block, layer_idx, device)
    print(f"Layer {layer_idx} loaded to {device}")

    # Check all params are filled (not meta)
    for name, p in block.named_parameters():
        if p.is_meta:
            raise RuntimeError(f"Parameter {name} is still on meta device!")
    print(f"All {sum(1 for _ in block.named_parameters())} parameters materialized")

    # Quick sanity: sum of weights should be finite
    total = sum(p.sum().item() for p in block.parameters() if p.numel() <= 1_000_000)
    print(f"Sum of first-1M-param weights: {total:.2f}")

    # Test observer integration
    from reap.v4_moe_observer import DeepseekV4MoEObserver
    from reap.observer import OBSERVER_CONFIG_REGISTRY
    from reap.pruning_metrics import initialize_pruning_state

    moe_block = block.mlp
    hook_config = OBSERVER_CONFIG_REGISTRY[model.__class__.__name__]()
    state = initialize_pruning_state(moe_block.experts.num_experts)
    observer = DeepseekV4MoEObserver(model, hook_config, v4_loader=v4_loader)
    hidden_3d = torch.randn(1, 16, config.hidden_size, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
    observer._process_moe_activations(
        layer_idx, moe_block, hidden_3d, device, attention_mask=None, input_ids=input_ids,
    )
    print(f"Observer metrics computed")

    # Benchmark (observer only, excluding disk I/O)
    start = time.time()
    observer._process_moe_activations(
        layer_idx, moe_block, hidden_3d, device, attention_mask=None, input_ids=input_ids,
    )
    elapsed = time.time() - start
    print(f"Single forward+observe: {elapsed:.2f}s")
    print(f"Estimated 43 layers: {elapsed * 43 / 60:.1f} min")

    # Unload
    block.to("meta")
    gc.collect()
    print("=== PASSED ===")


if __name__ == "__main__":
    main()
