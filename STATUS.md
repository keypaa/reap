# DeepSeek-V4-Flash Integration Status

## Goal
Run the full 43-layer layerwise pruning observer pipeline for DeepSeek-V4-Flash on a CPU-only Lightning.ai instance (31GB RAM). The pipeline loads each transformer block's weights from disk (safetensors shards), runs a forward pass to capture MoE router/expert activations, computes REAP pruning metrics, then unloads the block and moves to the next.

## What Works — Blocks 0 and 1 (hash routers)
Integration test `python scripts/test_v4_full_integration.py --max-layers 2 --num-batches 1 --seq-len 16` **passes**:
- Block 0: 16 tokens processed, 96 expert hits (16 tokens × top_k=6), completed in ~62s total (including model creation + shard loading)
- Block 1: same metrics, completed successfully
- Both use hash routers (blocks 0-2 on V4 Flash are `DeepseekV4HashRouter`)

## What Doesn't — Block 2 and beyond
`--max-layers 3` (blocks 0, 1, 2) or `--specific-layers "2"` hang/crash:
- Blocks 0-1 complete (~7.5s each forward pass)
- Block 2: `_build_layer_state_dict(2)` hangs indefinitely (test timed out after 600s)
- Also fails with `RuntimeError: Tensor on device meta is not on the expected device cpu!` when run as the first block (`--specific-layers "2"`)

## Facts Established

### Shard 00004 loads fine in isolation
```python
path = hf_hub_download("deepseek-ai/DeepSeek-V4-Flash", "model-00004-of-00046.safetensors")
st = safetensors.safe_open(path, framework="pt")
# Opens in 2.4s, 1576 keys — works perfectly
```

### Available cached shards
Only 5 of 46 shards are cached on the instance:
- model-00001-of-00046 (1011 MB, blob: 51765866...)
- model-00002-of-00046 (unknown size, blob: f0404818...)
- model-00003-of-00046 (unknown size, blob: df5f80b9...)
- model-00004-of-00046 (3.4 GB, blob: 948250b4...)
- model-00045-of-00046 (blob: 9a0fd242...)

300 GB free disk space on the instance.

### Layer assignments
| Layer | Shard | Tensors |
|-------|-------|---------|
| 0 | model-00002-of-00046 | 1565 |
| 1 | model-00003-of-00046 | 1565 |
| 2 | model-00004-of-00046 | 1576 |
| 3 | model-00005-of-00046 | 1569 |
| 4+ | model-00006+ | ~1576 each |

Shards 00005+ are NOT cached — they'd auto-download via `hf_hub_download` on first access. Block 2 needs shard 00004 (cached).

### Two failure modes

**Failure mode A (blocks 0→1→2 sequential):** hang during `_build_layer_state_dict(2)`. No error, no crash, no OOM — just silence. The process was observed via `ps` and was no longer running after ~8 minutes (the SSH session timeout).

**Failure mode B (block 2 alone with `--specific-layers "2"`):** immediate crash:
```
RuntimeError: Tensor on device meta is not on the expected device cpu!
```
This happens during `_capture_first_block_inputs` when `_load_block_for_replay(0)` loads block 2 into `self.blocks[0]`, then tries to run the forward pass. The error originates from a torch custom op fake_impl, suggesting some parameter is still on meta device.

## Root Cause Hypotheses

### Hypothesis 1: Memory exhaustion during dequantization
Layer 2's shard (00004) is 3.4 GB, vs ~1 GB for layers 0-1 (shards 00002-00003). `_build_layer_state_dict` loads ALL 128 experts' w1/w2/w3 FP4 weights and dequantizes them simultaneously. For FP4, each expert has ~3.4 GB / 128 ≈ 27 MB of raw int8 data per expert, but dequantized to bf16 doubles the size. Accumulating all 128 experts in memory before stacking could require several GB, and combined with the already-loaded blocks 0-1 data (replay cache, pruning state, etc.), this may exceed 31 GB.

**Evidence**: The process hangs silently rather than crashing — a classic sign of swap thrashing on the Lightning CPU instance.

### Hypothesis 2: `_build_layer_state_dict` does excessive work
The method processes ALL 128 experts' weights even when the layer only needs a forward pass. Each expert requires:
1. Loading int8 weight + fp4 scale from safetensors (mmap I/O)
2. Dequantizing: `dequantize_fp4_weight` → nibble unpack + LUT lookup + reshape + scale multiply
3. Accumulating in lists → `torch.stack` → `torch.cat`

This is O(n_experts × d_model × d_ff) CPU work. If the shard is on a slow filesystem (NFS or FUSE), the I/O could dominate.

### Hypothesis 3: Block 2 has different quantization format
Layer 2 might use FP8 quantization (not FP4) for its experts. The dequantize path would differ, and the batch processing might be slower. Or vice versa — the FP4 LUT dequant is slower than FP8.

### Hypothesis 4: Meta tensor leak in `--specific-layers` mode
When run alone (`--specific-layers "2"`), `_load_block_for_replay(0)` builds the state dict for layer 2, calls `load_state_dict(assign=True)`, but some submodule (possibly `self_attn.rotary_emb` or the compressor/indexer) remains on meta device because `load_non_backbone_modules` was not called before the observer was created. Wait — `load_non_backbone_modules(model)` IS called (line 39 of test script), which materializes embed, norm, lm_head, hc_head, and rotary_emb. But the error says a tensor is on meta device. This suggests either:
- The rotary_emb materialization fails silently for compressed rotary
- Some other submodule (compressor, indexer) has meta tensors that aren't overwritten by `_build_layer_state_dict`

## Suggested Next Steps

1. **Profile `_build_layer_state_dict(2)`** — run it standalone with timing to see if it's slow or truly hung:
   ```python
   import time; t0 = time.time()
   sd = loader._build_layer_state_dict(2)
   print(f"Took {time.time()-t0:.1f}s")
   ```

2. **Check memory** — add `psutil.virtual_memory()` logging in `_build_layer_state_dict` to detect OOM pressure:
   ```python
   import psutil; print(f"RAM: {psutil.virtual_memory().percent}% used")
   ```

3. **Check expert format** — compare whether layer 2 experts use FP4 vs FP8 differently from layers 0-1:
   ```python
   names = loader.layer_map.get(2, [])
   for n in names[:5]: print(n, loader.index["weight_map"].get(n))
   ```

4. **Check meta tensor leak** — after `load_into_block(block, 2)`, enumerate all parameters and check `.is_meta`:
   ```python
   for n, p in block.named_parameters():
       if p.is_meta: print(f"META: {n}")
   ```

5. **Download full model** — run `hf download deepseek-ai/DeepSeek-V4-Flash` to cache all 46 shards on the instance (estimated: ~160 GB total, 300 GB free).

6. **Reduce expert batch size** — the `expert_batch_size` parameter in `DeepseekV4MoEObserver.__init__` is 0 (process all at once). Set it to 32 or 64 to reduce peak memory.

## Key Code Locations

- `src/reap/v4_block_loader.py` — `_build_layer_state_dict()` at line 465, `dequantize_fp4_weight()` at line 30, `_process_shared_experts()` at line 410
- `src/reap/v4_moe_observer.py` — `_capture_first_block_inputs()` at line 70, `_load_block_for_replay()` at line 55
- `src/reap/layerwise_observer.py` — `_forward_block()` at line 729, `_record_all_blocks_for_batch_group()` at line 885
- `scripts/test_v4_full_integration.py` — integration test entry point

## Bugs Already Fixed This Session

1. **HC rename map** (`v4_block_loader.py:100-105`): was `self_attn.attn_hc.fn`, should be `attn_hc.fn` — caused `load_state_dict(assign=True)` to silently drop HC keys, leaving them on meta
2. **Missing `position_embeddings`** (`v4_moe_observer.py:131`): block forward was called without `position_embeddings` dict (requires `main`/`compress` from rotary_emb)
3. **FP8 dequant dtype** (`v4_block_loader.py:82`): `dequantize_fp8_weight` returned f32 (bf16 × f32), attention weights need bf16
4. **ModuleList overflow** (`layerwise_model_utils.py:278`): `extract_model_components` returned all 43 `model.layers` blocks when only a subset was requested
