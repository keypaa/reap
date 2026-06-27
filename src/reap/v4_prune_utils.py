from __future__ import annotations

import gc
import logging

import torch
import torch.nn as nn
from tqdm import tqdm


def _prune_v4_layer(moe, retained_indices):
    """Prune V4 experts by indexing 3D weight tensors along dim 0.

    V4's DeepseekV4Experts stores weights as 3D nn.Parameter tensors:
    - gate_up_proj: [num_experts, 2*intermediate_dim, hidden_dim]
    - down_proj: [num_experts, hidden_dim, intermediate_dim]

    Pruning: keep retained_indices along dim 0, update num_experts,
    prune router gate, remap hash router tid2eid if applicable.
    """
    retained_indices = sorted(retained_indices)
    num_retained = len(retained_indices)

    # Save original count before mutation (needed for hash router old_to_new)
    original_num_experts = moe.experts.num_experts

    # 1. Expert weights — replace 3D nn.Parameter tensors
    moe.experts.gate_up_proj = nn.Parameter(
        moe.experts.gate_up_proj.data[retained_indices].clone()
    )
    moe.experts.down_proj = nn.Parameter(
        moe.experts.down_proj.data[retained_indices].clone()
    )
    moe.experts.num_experts = num_retained

    # 2. Router gate weights
    moe.gate.weight.data = moe.gate.weight.data[retained_indices]
    if hasattr(moe.gate, "num_experts"):
        moe.gate.num_experts = num_retained
    if hasattr(moe.gate, "out_features"):
        moe.gate.out_features = num_retained

    # 3. e_score_correction_bias — TopKRouter only
    if hasattr(moe.gate, "e_score_correction_bias"):
        moe.gate.e_score_correction_bias.data = (
            moe.gate.e_score_correction_bias.data[retained_indices].clone()
        )

    # 4. Hash router tid2eid remapping
    if hasattr(moe.gate, "tid2eid"):
        old_to_new = [-1] * original_num_experts
        for new_idx, old_idx in enumerate(retained_indices):
            old_to_new[old_idx] = new_idx
        _remap_hash_router_tid2eid(moe.gate, old_to_new)

    # 5. Shared experts — NEVER prune


def _remap_hash_router_tid2eid(gate, old_to_new):
    """Remap tid2eid lookup table after expert pruning.

    tid2eid is a registered buffer [vocab_size, top_k] mapping token IDs
    to expert indices. After pruning, old expert indices must be remapped
    to new (compacted) indices.

    old_to_new: list where old_to_new[old_idx] = new_idx for retained,
                -1 for pruned (falls back to expert 0).
    """
    old_to_new_tensor = torch.tensor(
        old_to_new, device=gate.tid2eid.device, dtype=torch.long
    )
    tid2eid = gate.tid2eid.data.clone()
    was_valid = tid2eid >= 0
    safe_idx = tid2eid.clamp(min=0)
    remapped = old_to_new_tensor[safe_idx]
    remapped = remapped.clamp(min=0)
    remapped[~was_valid] = -1
    gate.tid2eid.data.copy_(remapped)


def prune_v4_model(observer_data, model, v4_loader, prune_args, n_experts_to_prune, pruned_model_dir):
    """Prune a V4 model layer by layer, loading real weights from disk.

    Each layer is loaded from disk, pruned in-place, then moved back to meta
    to keep peak CPU memory at ~layer_size + non_backbone.
    """
    from reap.model_util import get_super_expert_indices

    logger = logging.getLogger(__name__)

    for layer in observer_data:
        if "expert_proba" not in observer_data[layer]:
            observer_data[layer]["expert_proba"] = (
                observer_data[layer]["expert_frequency"]
                / observer_data[layer]["total_tokens"]
            )

    if prune_args.perserve_super_experts or prune_args.perserve_outliers:
        super_expert_idx = get_super_expert_indices(
            observer_data, include_last_layers=prune_args.perserve_outliers
        )
        metrics = [
            "expert_proba", "ean_sum", "ean_mean",
            "weighted_expert_frequency_sum", "weighted_ean_sum",
            "reap", "reap_l2", "weighted_ean_sum_l2",
        ]
        for layer in observer_data:
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                for metric in metrics:
                    if metric in observer_data[layer]:
                        observer_data[layer][metric][super_experts_in_layer] = float("inf")

    num_layers = model.config.num_hidden_layers
    num_retained = model.config.n_routed_experts - n_experts_to_prune

    for layer_idx in tqdm(range(num_layers), desc="Pruning V4 layers..."):
        if layer_idx not in observer_data:
            continue

        v4_loader.load_into_block(model.model.layers[layer_idx], layer_idx)

        num_experts = observer_data[layer_idx]["expert_frequency"].shape[0]
        prune_method = prune_args.prune_method
        if prune_method == "frequency":
            prune_method = "expert_frequency"
        saliency_data = observer_data[layer_idx].get(prune_method)
        if saliency_data is None:
            raise ValueError(
                f"Prune method {prune_args.prune_method} not found in observer data "
                f"for layer {layer_idx}. Available keys: "
                f"{list(observer_data[layer_idx].keys())}"
            )

        _, experts_to_prune = torch.topk(
            saliency_data, n_experts_to_prune, largest=False
        )
        retained_indices = [i for i in range(num_experts) if i not in experts_to_prune]

        moe_block = model.model.layers[layer_idx].mlp
        _prune_v4_layer(moe_block, retained_indices)

        model.model.layers[layer_idx].to("meta")
        gc.collect()

    model.config.n_routed_experts = num_retained
    model.config.num_local_experts = num_retained

    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(pruned_model_dir))
    logger.info("Pruned V4 model saved to %s", pruned_model_dir)
