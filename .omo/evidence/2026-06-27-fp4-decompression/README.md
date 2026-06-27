# FP4→BF16 Decompression Validation

**Date:** 2026-06-27
**Model:** DeepSeek-V4-Flash (284B)
**GPU:** A100-80GB (Modal)
**Verdict:** YES — `from_pretrained` automatically decompresses FP4→BF16

## Raw Findings

| Weight | Storage dtype | Shape | Decompressed target |
|--------|--------------|-------|-------------------|
| experts.N.w1.weight (gate) | I8 (packed FP4) | [2048, 2048] | BF16 |
| experts.N.w1.scale | F8_E8M0 | [2048, 128] | — |
| experts.N.w2.weight (down) | I8 (packed FP4) | [4096, 1024] | BF16 |
| experts.N.w2.scale | F8_E8M0 | [4096, 64] | — |
| experts.N.w3.weight (up) | I8 (packed FP4) | [2048, 2048] | BF16 |
| experts.N.w3.scale | F8_E8M0 | [2048, 128] | — |
| shared_experts.w1.weight | F8_E4M3 (native FP8) | [2048, 4096] | BF16 |
| gate.weight (router) | BF16 | [256, 4096] | BF16 (no conversion) |
| gate.bias (e_score_correction) | F32 | [256] | F32 (no conversion) |
| gate.tid2eid (hash table) | I64 | [129280, 6] | I64 (no conversion) |

## Implications

- `config.torch_dtype = "bfloat16"` → target format
- `config.expert_dtype = "fp4"` → logical precision
- Experts stored as I8 (INT8, packing 2 FP4 values per byte) + per-block F8_E8M0 scale factors
- Custom modeling code in transformers 4.57.1+ handles decompression during `from_pretrained`
- Router stays in BF16, bias in F32 — no conversion needed
- **Standard layerwise pipeline is feasible**
- Block-from-disk approach can use `from_pretrained` (handles decompression) or replicate I8→BF16 using `wN.scale` tensors
