from __future__ import annotations

import torch
import torch.nn as nn


def _prune_v4_layer(moe, retained_indices, model, layer_idx):
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
