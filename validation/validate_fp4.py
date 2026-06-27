import json
import os
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


def inspect_raw_safetensor(model_name: str) -> None:
    print(f"\n{'='*60}")
    print("Step 1: Raw safetensor inspection")
    print(f"{'='*60}")

    index_path = _resolve_index(model_name)
    with open(index_path) as f:
        index = json.load(f)

    shards = index.get("weight_map", {})
    shared_expert_key = [k for k in shards if "shared_experts" in k and "gate_up" in k]
    if not shared_expert_key:
        shared_expert_key = [k for k in shards if "shared_experts" in k]
    log("Shared expert weight key", shared_expert_key[:2] if shared_expert_key else "not found")

    first_shard = next(iter(shards.values()))
    log("First shard file", first_shard)

    shard_path = _resolve_shard(model_name, first_shard)
    if shard_path is None:
        log("Shard path", "NOT FOUND — cannot inspect raw tensors")
        return

    with safetensors.safe_open(shard_path, framework="pt") as f:
        keys = f.keys()
        expert_keys = [k for k in keys if "experts.gate_up_proj" in k]
        log("Expert weight keys in first shard", expert_keys[:3])
        if expert_keys:
            tensor_meta = f.get_slice(expert_keys[0])
            log("Expert tensor dtype (safetensor metadata)", str(f.get_tensor_info(expert_keys[0]).get("dtype")))
            log("Expert tensor shape", tensor_meta.get_shape())


def validate_transformers_load(model_name: str, device: str) -> None:
    print(f"\n{'='*60}")
    print(f"Step 2: from_pretrained (device_map={device!r})")
    print(f"{'='*60}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    log("Load time (s)", f"{load_time:.1f}")

    _inspect_model_dtypes(model)
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

    #fallback: try without .index suffix
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


def validate_gpu_move(model) -> None:
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

    mem = torch.cuda.memory_summary()
    log("GPU memory summary", mem)

    return target_layer


def validate_forward_pass(layer, tokenizer, model_name: str, device: str) -> None:
    print(f"\n{'='*60}")
    print("Step 4: Forward pass on a single layer")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    moved = model.model.layers[3].to("cuda")

    tokens = tokenizer("def hello():", return_tensors="pt")
    tokens = {k: v.to("cuda") for k, v in tokens.items()}

    with torch.no_grad():
        # Run through layers 0-3
        hidden = model.model.embed_tokens(tokens["input_ids"])
        for i in range(3):
            hidden = model.model.layers[i](hidden, output_attentions=False)[0]
        output = moved(hidden, output_attentions=False)

    log("Layer output shape", list(output[0].shape))
    log("Layer output dtype", str(output[0].dtype))
    log("Layer output contains NaN", bool(torch.isnan(output[0]).any()))
    log("Layer output contains Inf", bool(torch.isinf(output[0]).any()))

    if not torch.isnan(output[0]).any() and not torch.isinf(output[0]).any():
        log("Forward pass", "PASS")
    else:
        log("Forward pass", "FAIL (NaN/Inf detected)")


def run_all(model_name: str) -> dict[str, Any]:
    global RESULTS
    RESULTS = {}
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = validate_config(model_name)

    # Step 1: raw safetensor dtype
    inspect_raw_safetensor(model_name)

    # Step 2: load with from_pretrained on CPU
    model = validate_transformers_load(model_name, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Step 3: GPU move
    layer = validate_gpu_move(model)

    # Step 4: forward pass
    if layer is not None:
        validate_forward_pass(layer, tokenizer, model_name, "cpu")

    # Step 5: download config.json
    print(f"\n{'='*60}")
    print("Step 5: Config details")
    print(f"{'='*60}")
    log("tokenizer vocab size", tokenizer.vocab_size)
    log("config.quantization_config", json.dumps(
        getattr(config, "quantization_config", None), indent=2, default=str
    ))

    return RESULTS
