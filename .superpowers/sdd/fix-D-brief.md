# Fix D: Phase 5.1 C2 — HF model name resolution

## Files to Edit
- `src/reap/v4_block_loader.py`
- `src/reap/layerwise_prune.py`

## Issue

`V4BlockDiskLoader(model_name)` does `self.model_path = Path(model_name)`. For HF model IDs like `deepseek-ai/DeepSeek-V4-Flash`, this creates a relative path that doesn't exist. `model.safetensors.index.json` would fail with `FileNotFoundError`.

## Fix

In `V4BlockDiskLoader.__init__`, resolve HF model names to local cache paths:

```python
import os
from pathlib import Path
from huggingface_hub import snapshot_download

class V4BlockDiskLoader:
    def __init__(self, model_path, config=None):
        self.model_path = self._resolve_path(model_path)
        # ... rest of init
    
    @staticmethod
    def _resolve_path(path):
        p = Path(path)
        if p.exists():
            return p
        
        # Check if it's an HF model ID and resolve to cache
        try:
            from huggingface_hub import snapshot_download
            cached = snapshot_download(str(path))
            return Path(cached)
        except (ImportError, Exception) as e:
            raise FileNotFoundError(
                f"Model path '{path}' does not exist locally and could not be "
                f"resolved via HuggingFace Hub. "
                f"Download it first with:\n"
                f"  huggingface-cli download {path}"
            )
```

But `snapshot_download` downloads the ENTIRE model (~160 GB). This is NOT what we want during pipeline setup — we just need to find the local cache path.

**Better approach:** Use `transformers` utilities to find cached paths:

```python
from transformers.utils.hub import cached_file

try:
    index_file = cached_file(str(path), "model.safetensors.index.json")
    return Path(index_file).parent
except Exception:
    raise FileNotFoundError(...)
```

Or even simpler — since the model should be cached before running the pipeline:

```python
@staticmethod
def _resolve_path(path):
    p = Path(path)
    if p.exists():
        return p
    
    # Check transformers cache
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
        # config._name_or_path may have the cache path
    except Exception:
        pass
    
    # Try huggingface_hub snapshot_download (without downloading if cached)
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    
    cache_dir = Path(HUGGINGFACE_HUB_CACHE) / f"models--{str(path).replace('/', '--')}"
    if cache_dir.exists() and (cache_dir / "snapshots").exists():
        snapshots = list((cache_dir / "snapshots").iterdir())
        if snapshots:
            return snapshots[0]
    
    raise FileNotFoundError(
        f"Model path '{path}' not found locally. "
        f"Download with: huggingface-cli download {path}"
    )
```

Also wrap the `V4BlockDiskLoader` constructor in `layerwise_prune.py` so the error is handled gracefully.

## Tests
- `test_v4_block_loader.py`: Test `_resolve_path` with local paths
- Mock the HF cache for path resolution tests

## Environment

PYTHONPATH="src"
Python: `C:\Users\pauma\miniconda3\envs\py310\python.exe`

## Commit

Message: "Fix D: resolve HF model names to local cache in V4BlockDiskLoader"
