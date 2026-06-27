# FP4/FP8 Weight Decompression Validation

Validates whether `AutoModelForCausalLM.from_pretrained` automatically decompresses
DeepSeek V4's FP4+FP8 mixed-precision weights to BF16 on load.

## Why This Matters

V4 Flash (284B) and Pro (1.6T) publish weights in mixed FP4+FP8 format — no BF16
weights are available. The REAP observer and pruning pipeline need BF16 weights for:

1. **Observer:** Forward hooks compute expert activations via `F.linear` — needs standard
   float tensors, not quantized
2. **Pruning:** Weight tensor indexing (`gate_up_proj[retained_indices]`) must operate
   on dequantized weights

If `from_pretrained` handles decompression automatically, we use standard
`transformers` loading and the existing layerwise pipeline works with minimal changes.
If not, we need a custom FP4/FP8→BF16 dequantizer.

## How to Run

```bash
# Set your HF token (gated model)
export HF_TOKEN=hf_...

# Run on Modal (A100 80GB, ~$0.42 for a 10min run)
cd validation
modal run app.py --model-name deepseek-ai/DeepSeek-V4-Flash
```

## What It Checks

1. Raw safetensor dtype (reads a single weight shard directly)
2. `from_pretrained` dtype on CPU load
3. dtype after moving a layer to GPU
4. Quantization config metadata
5. Small forward pass to verify model integrity
6. Memory usage metrics
