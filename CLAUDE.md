# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
bash scripts/build.sh          # creates .venv, inits submodules, installs everything
```

Uses `uv` + hatchling (pyproject.toml). Python 3.12 required. Editable install with torch + vllm pinned. Three third-party eval tools (evalplus, LiveCodeBench, helm) live as git submodules under `third-party/`.

Docker alternative:
```bash
docker compose up --build -d && docker compose exec app bash
```

## Tests

```bash
uv run pytest tests/                   # all tests
uv run pytest tests/test_pruning_metrics.py  # single file
uv run pytest tests/test_pruning_metrics.py::test_name  # single test (if test functions have names)
```

7 test files covering arg parsing, dataset loading, observer, pruning metrics, and e2e pruning.

## Lint

```bash
uv run ruff check src/ tests/
```

Minimal config: only SIM (simplify) rules enabled, line-length 88. No formatter, no mypy, no pre-commit hooks.

## Architecture

Two parallel pipelines share infrastructure: **merge** (cluster + combine expert weights) and **prune** (select low-saliency experts + remove them).

### Data Flow

```
args.py (7 dataclasses, HfArgumentParser)
  ↓
data.py (dataset loading, tokenization, composite dataset specs)
  ↓
observer.py → pruning_metrics.py (forward hooks capture router logits + expert activations, accumulate REAP scores)
  ↓
┌─ merge path: cluster.py → permute.py → merge.py
└─ prune path: prune.py (removes lowest-saliency experts)
  ↓
eval.py (vLLM serving + lm-eval, evalplus, LiveCodeBench)
```

### Key Modules

| Module | Role |
|--------|------|
| `args.py` | All config as dataclasses. 7 classes: ReapArgs, ModelArgs, DatasetArgs, ObserverArgs, PruneArgs, ClusterArgs, EvalArgs, LayerwiseArgs |
| `model_util.py` | `MODEL_ATTRS` registry mapping model class names → MoE submodule paths. `patched_model_map()` swaps in custom modeling files for models that don't expose router logits |
| `observer.py` | `MoETransformerObserver` — registers forward hooks on MoE blocks, captures activations. Per-model configs in `OBSERVER_CONFIG_REGISTRY` for Qwen3, Llama4, Mixtral, DeepSeek, Ernie, GLM4 |
| `pruning_metrics.py` | `initialize_pruning_state()` + `update_pruning_state()` — per-token accumulation of expert_frequency, REAP scores, activation norms |
| `metrics.py` | Distance functions (angular, cosine, CKA, JSD) + `OnlineStatsTracker` for memory-efficient online statistics |
| `cluster.py` | Agglomerative, kmeans, MC-SMoE, frequency-penalized clustering algorithms |
| `merge.py` | `MoEExpertMerger` + `MergeMethod` enum — merges expert weights within clusters |
| `data.py` | `DATASET_REGISTRY` maps HF dataset names → processor classes. Supports composite multi-dataset specs (`name1:N1,name2:N2`) via `parse_composite_dataset_spec()` |

### Layerwise Pipeline (for large models)

Three files implement memory-efficient calibration by processing one transformer block at a time:
- `layerwise_prune.py` — entry point, same CLI interface as `prune.py`
- `layerwise_observer.py` — `LayerwiseMoEObserver` with `ReplayCache` for passing hidden states between blocks
- `layerwise_model_utils.py` — `find_decoder_blocks()`, `extract_model_components()`, block move/cleanup utilities

### Entry Points

All three are `python -m` scripts (not console_scripts):
- `python -m reap.main` — merge pipeline
- `python -m reap.prune` — prune pipeline
- `python -m reap.layerwise_prune` — memory-efficient prune pipeline (recommended for models >100B params)

### Supported MoE Models

Qwen3-MoE, Llama4, Mixtral, DeepSeek-V2/V3.2/V4, ERNIE-4.5, GLM4. Each may have a patched modeling file under `src/reap/models/` if the upstream HF implementation doesn't expose router logits.

## CLI Argument Pattern

All entry points use `HfArgumentParser` with dataclass arguments:
```bash
python -m reap.layerwise_prune \
  --model-name "Qwen/Qwen3-30B-A3B" \
  --dataset-name "theblackcat102/evol-codealpaca-v1" \
  --prune-method "reap" \
  --compression-ratio 0.5 \
  --batch-size 4
```

Arguments map to dataclass fields via `--kebab-case` (e.g., `compression_ratio` → `--compression-ratio`). Some use `--snake_case` legacy format (e.g., `--output_file_name`, `--do-eval`).

## Adding a New Model

1. Add entry to `MODEL_ATTRS` in `model_util.py` mapping model class → MoE submodule path
2. If router logits aren't exposed, create a patched modeling file under `src/reap/models/`
3. Register in `patched_model_map()` in `model_util.py`
4. Add observer config to `OBSERVER_CONFIG_REGISTRY` in `observer.py`
