# Fix B: Phase 1/4 — Import guard, dequantize validation, shard cache leak

## Files to Edit
- `src/reap/v4_block_loader.py`
- `src/reap/layerwise_prune.py`
- `tests/test_v4_block_loader.py`

## Issues to Fix

### B1: Import guard for transformers V4 module

`v4_block_loader.py:9-13` imports `DeepseekV4Config`, `DeepseekV4DecoderLayer`, `DeepseekV4RMSNorm` from transformers. These only exist in transformers >= 5.9.0.

**Fix:** Add try/except import guard:

```python
try:
    from transformers import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4DecoderLayer,
        DeepseekV4RMSNorm,
    )
except (ImportError, KeyError):
    DeepseekV4Config = None
    DeepseekV4DecoderLayer = None
    DeepseekV4RMSNorm = None
```

Then in `__init__` of `V4BlockDiskLoader`, check:
```python
if DeepseekV4Config is None:
    raise ImportError(
        "DeepSeek V4 support requires transformers >= 5.9.0. "
        "Install with: pip install transformers>=5.9.0"
    )
```

### B2: Dequantize dimension non-divisible validation

`dequantize_fp4_weight()` uses integer division `rows // scale_rows` which silently truncates. If dims aren't divisible by block_size (32), the reshape crashes.

**Fix:** Add validation at the start:
```python
block_m_target = 32  # F8_E8M0 blocks every 32 columns
if cols % block_m_target != 0:
    raise ValueError(
        f"Quantized tensor columns ({cols}) must be divisible by block size ({block_m_target})"
    )
```

Also add the check for rows:
```python
if scale_rows * block_m != rows:
    raise ValueError(
        f"Shape mismatch: {rows} rows with {scale_rows} scale rows "
        f"and {block_m} rows per block"
    )
```

Add a test for this validation.

### B3: Shard cache leak

`_shard_cache` accumulates open safetensor handles that each mmap ~3.57 GB. After 43 layers, 46 shards × 3.57 GB = 164 GB virtual memory.

**Fix:** In `unload_layer()`, add shard cache clearing:
```python
def unload_layer(self, layer, clear_shard_cache=False):
    layer.to("cpu")
    del layer
    gc.collect()
    if clear_shard_cache:
        self.close()
```

Also modify `close()`:
```python
def close(self):
    self._shard_cache.clear()
    gc.collect()
```

And in `layerwise_prune.py`, call `v4_loader.close()` after the observation loop completes (before pruning reload).

### B4: Non-divisible scale dimensions

Add to `dequantize_fp4_weight`:
```python
if quantized.dim() < 2 or scales.dim() < 2:
    raise ValueError(
        f"Quantized tensor dims ({quantized.dim()}) and scales dims ({scales.dim()}) must be >= 2"
    )
```

## Tests

- `test_fp4_dequantize_non_divisible_raises` — verify expected error for non-divisible dims
- Existing tests should still pass

## Environment

PYTHONPATH="src"
Python: `C:\Users\pauma\miniconda3\envs\py310\python.exe`

Test command:
```powershell
$env:PYTHONPATH="src"; & "C:\Users\pauma\miniconda3\envs\py310\python.exe" -m pytest tests/test_v4_block_loader.py -v
```

## Commit

Message: "Fix B: import guard, dequantize validation, shard cache cleanup"
