import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import safetensors
import safetensors.torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

RESULTS: dict[str, Any] = {}


def log(label: str, value: object, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}• {label}: {value}")
    RESULTS[label] = value


def validate_config(model_name: str) -> AutoConfig:
    print(f"\n{'='*60}")
    print(f"Loading config: {model_name}")
    print(f"{'='*60}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    log("Model type", config.architectures)
    log("Hidden size", config.hidden_size)
    log("Num layers", config.num_hidden_layers)
    log("Routed experts", config.n_routed_experts)
    log("Experts per tok", config.num_experts_per_tok)
    log("Torch dtype", config.torch_dtype)

    qconfig = getattr(config, "quantization_config", None)
    log("Quantization config", qconfig)

    expert_dtype = getattr(config, "expert_dtype", None)
    log("Expert dtype (config field)", expert_dtype)

    return config


def inspect_raw_safetensor(model_name: str) -> str | None:
    """Step 1: Inspect raw safetensor metadata (works with minimal RAM).
    Returns path to a shard containing expert weights, or None."""
    print(f"\n{'='*60}")
    print("Step 1: Raw safetensor inspection")
    print(f"{'='*60}")

    index_path = _resolve_index(model_name)
    with open(index_path) as f:
        index = json.load(f)

    shards = index.get("weight_map", {})

    # Find first shard that actually contains expert weights
    shard_to_expert_keys: dict[str, list[str]] = {}
    for key, shard_name in shards.items():
        if "experts.gate_up_proj" in key or "experts.down_proj" in key or "mlp.experts" in key:
            shard_to_expert_keys.setdefault(shard_name, []).append(key)

    log("Total shards with expert weights", len(shard_to_expert_keys))
    if not shard_to_expert_keys:
        log("Expert shards", "NONE FOUND")
        return None

    # Pick the first shard that has expert weights
    expert_shard = min(shard_to_expert_keys.keys(), key=lambda s: int(s.split("-")[-1].split(".")[0]))
    log("Selected expert shard", expert_shard)
    log("Expert keys in that shard", shard_to_expert_keys[expert_shard][:3])

    shard_path = _resolve_shard(model_name, expert_shard)
    if shard_path is None:
        log("Shard path", "NOT FOUND — cannot inspect raw tensors")
        return None

    with safetensors.safe_open(shard_path, framework="pt") as f:
        for ek in shard_to_expert_keys[expert_shard]:
            info = f.get_tensor_info(ek)
            log(f"  {ek.split('.')[-1]} dtype (safetensor metadata)", str(info.get("dtype")), indent=1)
            log(f"  shape", list(f.get_slice(ek).get_shape()), indent=1)
            break

    log("Shard available for weight loading", shard_path)
    return shard_path


def load_expert_shard_and_check_dtype(shard_path: str) -> None:
    """Step 2a: Load one shard with safetensors.torch.load_file and check resulting dtype."""
    print(f"\n{'='*60}")
    print("Step 2a: Load expert shard with safetensors.torch")
    print(f"{'='*60}")

    state = safetensors.torch.load_file(shard_path, device="cpu")
    expert_keys = [k for k in state if "experts.gate_up_proj" in k or "experts.down_proj" in k]
    if not expert_keys:
        log("Expert keys in loaded shard", "NONE")
        return

    for k in expert_keys[:4]:
        t = state[k]
        log(f"  {k} dtype", str(t.dtype), indent=1)
        log(f"  shape", list(t.shape), indent=1)

    log("Result", "Weights loaded successfully — safetensors dtype determined")
    del state


def validate_transformers_load_offload(model_name: str) -> Any:
    """Step 2b: Load via from_pretrained with accelerate disk offloading (may OOM)."""
    print(f"\n{'='*60}")
    print("Step 2b: from_pretrained with disk offloading")
    print(f"{'='*60}")
    log("Status", "ATTEMPTING — model is ~160GB, may OOM on CPU RAM")

    if not torch.cuda.is_available():
        log("GPU available", False)
        log("Outcome", "SKIPPED — no GPU, full model load requires 160GB+ CPU RAM")
        return

    offload_dir = tempfile.mkdtemp(prefix="offload_")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory={0: "90GB"},
            offload_folder=offload_dir,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        load_time = time.time() - t0
        log("Load time (s)", f"{load_time:.1f}")
        log("Outcome", "SUCCESS — model loaded with offloading")

        _inspect_model_dtypes(model)
    except Exception as e:
        log("Load error", str(e))
        log("Outcome", f"FAILED — {e}")
        model = None

    return model


def _resolve_index(model_name: str) -> str:
    if Path(model_name).exists():
        candidates = list(Path(model_name).glob("model.safetensors.index.json"))
        if candidates:
            return str(candidates[0])
        single = Path(model_name) / "model.safetensors"
        if single.exists():
            return str(single)
        raise FileNotFoundError(f"No safetensors index in {model_name}")

    import requests
    url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors.index.json"
    resp = requests.get(url)

    if resp.status_code != 200:
        url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise FileNotFoundError(f"Cannot fetch safetensors index from {model_name}")
    return url


def _resolve_shard(model_name: str, shard_name: str) -> str | None:
    if Path(model_name).exists():
        candidate = Path(model_name) / shard_name
        return str(candidate) if candidate.exists() else None

    import requests
    url = f"https://huggingface.co/{model_name}/resolve/main/{shard_name}"
    resp = requests.head(url)
    if resp.status_code == 200:
        return url
    return None


def _download_shard(model_name: str, shard_path: str) -> str:
    """Download a shard to local temp file. Returns local path."""
    if Path(shard_path).exists():
        return shard_path

    import requests
    dest = os.path.join(tempfile.mkdtemp(prefix="shard_"), os.path.basename(shard_path))
    print(f"Downloading {shard_path} -> {dest} ...")
    resp = requests.get(shard_path, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192 * 1024):
            f.write(chunk)
    size_gb = os.path.getsize(dest) / 1e9
    log("Shard download size (GB)", f"{size_gb:.2f}")
    return dest


def _inspect_model_dtypes(model) -> None:
    log("Model device", model.device.type)
    log("Model dtype", str(model.dtype))

    first_moe = None
    for name, mod in model.named_modules():
        if "mlp" in name and hasattr(mod, "experts"):
            first_moe = (name, mod)
            break

    if first_moe is None:
        log("MoE block", "NOT FOUND")
        return

    name, moe = first_moe
    log("First MoE block", name)

    expert_weight = moe.experts.gate_up_proj
    down_weight = moe.experts.down_proj

    log("  gate_up_proj dtype", str(expert_weight.dtype), indent=1)
    log("  gate_up_proj shape", list(expert_weight.shape), indent=1)
    log("  gate_up_proj device", expert_weight.device.type, indent=1)
    log("  down_proj dtype", str(down_weight.dtype), indent=1)

    router = moe.gate
    router_weight = router.weight
    log("  router.weight dtype", str(router_weight.dtype), indent=1)
    log("  router.weight device", router_weight.device.type, indent=1)

    if hasattr(router, "e_score_correction_bias"):
        bias = router.e_score_correction_bias
        log("  e_score_correction_bias dtype", str(bias.dtype), indent=1)
        log("  e_score_correction_bias device", bias.device.type, indent=1)

    shared = moe.shared_experts
    if hasattr(shared, "gate_proj"):
        log("  shared_experts.gate_proj dtype", str(shared.gate_proj.dtype), indent=1)


def validate_gpu_and_forward(model, tokenizer, model_name: str) -> None:
    """Step 3-4: GPU move + forward pass on layer 3."""
    if model is None:
        log("GPU/Forward", "SKIPPED — model not loaded")
        return

    print(f"\n{'='*60}")
    print("Step 3: Move one layer to GPU, re-check dtype")
    print(f"{'='*60}")

    if not torch.cuda.is_available():
        log("GPU available", False)
        return

    log("GPU available", True)
    log("GPU device", torch.cuda.get_device_name(0))

    target_layer = model.model.layers[3]
    target_layer = target_layer.to("cuda")

    moe = target_layer.mlp
    expert_weight = moe.experts.gate_up_proj
    log("gate_up_proj dtype (on GPU)", str(expert_weight.dtype))
    log("gate_up_proj device", expert_weight.device.type)

    down_weight = moe.experts.down_proj
    log("down_proj dtype (on GPU)", str(down_weight.dtype))

    router_weight = moe.gate.weight
    log("router.weight dtype (on GPU)", str(router_weight.dtype))

    # Forward pass
    print(f"\n{'='*60}")
    print("Step 4: Forward pass on layer 3")
    print(f"{'='*60}")

    tokens = tokenizer("def hello():", return_tensors="pt")
    tokens = {k: v.to("cuda") for k, v in tokens.items()}

    with torch.no_grad():
        hidden = model.model.embed_tokens(tokens["input_ids"])
        for i in range(3):
            hidden = model.model.layers[i](hidden, output_attentions=False)[0]
        output = target_layer(hidden, output_attentions=False)

    log("Layer output shape", list(output[0].shape))
    log("Layer output dtype", str(output[0].dtype))
    log("Layer output contains NaN", bool(torch.isnan(output[0]).any()))
    log("Layer output contains Inf", bool(torch.isinf(output[0]).any()))

    if not torch.isnan(output[0]).any() and not torch.isinf(output[0]).any():
        log("Forward pass", "PASS")
    else:
        log("Forward pass", "FAIL (NaN/Inf detected)")


def conclusion() -> None:
    """Derive the gating answer from collected evidence."""
    print(f"\n{'='*60}")
    print("GATING QUESTION: Does from_pretrained decompress FP4→BF16?")
    print(f"{'='*60}")

    config_torch_dtype = RESULTS.get("Torch dtype")
    config_expert_dtype = RESULTS.get("Expert dtype (config field)")
    safetensor_dtype = None
    for k, v in RESULTS.items():
        if "dtype (safetensor metadata)" in k:
            safetensor_dtype = v
        if "dtype (safetensor" in k:
            safetensor_dtype = v

    loaded_dtype = None
    for k, v in RESULTS.items():
        if k.endswith("dtype") and "gate_up" in k and "on GPU" not in k and "shape" not in k and "safetensor" not in k:
            loaded_dtype = v

    log("config.torch_dtype", config_torch_dtype)
    log("config.expert_dtype", config_expert_dtype)
    log("safetensor metadata dtype", safetensor_dtype)
    log("loaded weight dtype (after from_pretrained)", loaded_dtype)

    # Answer
    if config_torch_dtype == "bfloat16" and config_expert_dtype == "fp4":
        log("FP4→BF16 decompression", "YES (config confirms: torch_dtype=bfloat16, expert_dtype=fp4)")
    if loaded_dtype == "torch.bfloat16":
        log("FP4→BF16 decompression confirmed", "YES (weights are torch.bfloat16 after from_pretrained)")
    elif loaded_dtype == "torch.uint8" or loaded_dtype == "torch.float8_e4m3fn":
        log("FP4→BF16 decompression confirmed", "NO (weights stay in compressed format)")

    log("Standard layerwise pipeline feasible", "YES (decompression is automatic)" if "YES" in str(RESULTS.get("FP4→BF16 decompression", "")) else "NEEDS CUSTOM DEQUANTIZER")


def run_all(model_name: str) -> dict[str, Any]:
    global RESULTS
    RESULTS = {}
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = validate_config(model_name)

    # Step 1: safetensor metadata inspection + find expert shard
    expert_shard_path = inspect_raw_safetensor(model_name)

    # Step 2a: download one expert-containing shard, load weights, check dtype
    model = None
    tokenizer = None
    if expert_shard_path:
        local_path = _download_shard(model_name, expert_shard_path)
        load_expert_shard_and_check_dtype(local_path)

        # Step 2b: try from_pretrained with accelerate disk offloading
        model = validate_transformers_load_offload(model_name)

    # Tokenizer (lightweight, always works)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Steps 3-4: GPU + forward pass
    validate_gpu_and_forward(model, tokenizer, model_name)

    # Step 5: Config details
    print(f"\n{'='*60}")
    print("Step 5: Config details")
    print(f"{'='*60}")
    log("tokenizer vocab size", tokenizer.vocab_size)
    log("config.quantization_config", json.dumps(
        getattr(config, "quantization_config", None), indent=2, default=str
    ))

    # Conclusion
    conclusion()

    return RESULTS
