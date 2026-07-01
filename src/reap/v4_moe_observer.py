"""
DeepSeek V4-specific MoE observer with incremental expert loop.

V4 stores expert weights as 3D nn.Parameter tensors (e.g., gate_up_proj: [N, 2*D, D]),
not as ModuleList of per-expert modules. This observer handles:
1. Indexing into 3D params via expert_idx
2. Processing one expert at a time (no [E, T, D] activation tensor)
3. TopKRouter (learned, 40/43 layers) and HashRouter (static hash, first 3 layers)
"""

from __future__ import annotations

import gc
import logging
import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from reap.layerwise_observer import (
    LayerwiseMoEObserver,
    ReplayBatch,
    _FirstblockInputCaptured,
)
from reap.layerwise_model_utils import (
    cleanup_memory,
    has_meta_tensors,
    move_to_device,
    safe_get_device,
)
from reap.pruning_metrics import update_pruning_state_single_expert

logger = logging.getLogger(__name__)


class DeepseekV4MoEObserver(LayerwiseMoEObserver):
    """V4-specific observer with incremental expert loop for 3D parameters.

    Overrides _process_moe_activations to iterate over experts by indexing
    into the 3D weight tensors directly, avoiding the [E, T, D] activation
    tensor and the broken enumerate(moe_module.experts) pattern.

    Also overrides block loading to load real weights from disk into meta blocks,
    and activation recording to pass input_ids for hash router support.
    """

    def __init__(self, model, hook_config, block_names=None, v4_loader=None, expert_batch_size=0):
        super().__init__(model, hook_config, block_names)
        self._v4_loader = v4_loader
        self._expert_batch_size = expert_batch_size

    def _load_block_for_replay(self, block_idx):
        if self.currently_loaded_block_idx == block_idx:
            return safe_get_device(self.blocks[block_idx])

        self._offload_current_block()

        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._v4_loader is not None:
            block = self._block_at(block_idx)
            layer_idx = self._actual_layer_idx(block_idx)
            if has_meta_tensors(block):
                self._v4_loader.load_into_block(block, layer_idx, target_device)

        final_device = self._move_block(self._block_at(block_idx), block_idx, target_device)
        self.currently_loaded_block_idx = block_idx
        return final_device

    def _capture_first_block_inputs(self, data_batches):
        """V4 override: embed tokens directly and call block 0 instead of the
        full model forward, because remaining blocks (1-42) are on meta device
        and would hang or error during a full forward pass."""
        if not self.blocks:
            raise ValueError("Layerwise replay requires at least one transformer block")

        self.replay_cache.clear()

        # Load block 0 from disk (everything: attention, norms, MoE)
        device_str = self._load_block_for_replay(0)
        target_device = torch.device(device_str)
        entry_block = self.blocks[0]

        captured_batches = []
        embed = self.model.model.embed_tokens.to(target_device)

        def intercept_entry_inputs(_, args, kwargs):
            replay_kwargs = {}
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            replay_inputs = [args[0].detach().cpu()]
            _input_ids = kwargs.get("input_ids")
            for key, value in kwargs.items():
                if key in ("hidden_states", "attention_mask", "position_ids", "input_ids"):
                    continue
                replay_kwargs[key] = move_to_device(value, "cpu")
            captured_batches.append(
                ReplayBatch(
                    inputs=replay_inputs,
                    kwargs=self._sanitize_cached_block_kwargs(replay_kwargs),
                    attention_mask=attention_mask.detach().cpu() if torch.is_tensor(attention_mask) else None,
                    position_ids=position_ids.detach().cpu() if torch.is_tensor(position_ids) else None,
                    input_ids=_input_ids.detach().cpu() if torch.is_tensor(_input_ids) else None,
                )
            )
            raise _FirstblockInputCaptured

        hook_handle = entry_block.register_forward_pre_hook(
            intercept_entry_inputs, with_kwargs=True
        )
        logger.info("Seeding replay cache (V4 mode: direct block 0 call)")

        try:
            for batch in data_batches:
                if isinstance(batch, torch.Tensor):
                    input_ids = batch.unsqueeze(0) if batch.dim() == 1 else batch
                elif isinstance(batch, (dict, transformers.tokenization_utils_base.BatchEncoding)):
                    input_ids = batch.get("input_ids", batch.get("input_ids", None))
                    if input_ids is None:
                        raise ValueError(f"Batch dict missing input_ids: {batch.keys()}")
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch)}")

                hidden = embed(input_ids.to(target_device))  # [B, S, hidden]
                hc_mult = getattr(self.model.config, "hc_mult", 1)
                position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=target_device).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids, device=target_device)
                # Compute position_embeddings from the 3D hidden (before HC expansion)
                # as the model does in DeepseekV4Model.forward
                position_embeddings = {
                    "main": self.model.model.rotary_emb(hidden, position_ids=position_ids, layer_type="main"),
                    "compress": self.model.model.rotary_emb(hidden, position_ids=position_ids, layer_type="compress"),
                }
                # Expand to 4D [B, S, hc_mult, hidden] as the block expects
                hidden = hidden.unsqueeze(2).expand(-1, -1, hc_mult, -1)

                try:
                    entry_block(
                        hidden,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        input_ids=input_ids,
                    )
                except _FirstblockInputCaptured:
                    continue
        finally:
            hook_handle.remove()

        for batch in captured_batches:
            self.replay_cache.append(
                inputs=batch.inputs,
                kwargs=batch.kwargs,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
                input_ids=batch.input_ids,
            )
        logger.info("Prepared replay cache for %s batches", len(self.replay_cache))

        if not self.replay_cache:
            raise ValueError("Replay cache did not capture any first-block inputs")

        # Offload block 0 so the block loop starts clean
        self._offload_current_block()

    def _offload_current_block(self):
        block_idx = self.currently_loaded_block_idx
        if block_idx < 0:
            return
        self.currently_loaded_block_idx = -1
        if self._v4_loader is not None:
            self._v4_loader.close()
        cleanup_memory(synchronize=False)

    @torch.inference_mode()
    def _record_activations_for_block(self, block_idx, moe_module=None):
        """V4-specific override that passes input_ids for hash routing."""
        if moe_module is None:
            moe_module = self._find_moe_module_in_block(block_idx)
            if moe_module is None:
                return self._forward_block(block_idx)

        captured_moe_input = {}
        moe_hook_handle = None
        _input_ids = None

        def _capture_moe_input_hook(module, args, output):
            captured_moe_input["input"] = args[0].detach()
            return output

        def _before_forward():
            captured_moe_input.clear()
            nonlocal _input_ids
            _input_ids = None

        def _after_forward(target_device, attention_mask, block_kwargs=None):
            nonlocal _input_ids
            if block_kwargs and "input_ids" in block_kwargs:
                _input_ids = block_kwargs["input_ids"]
            moe_input = captured_moe_input.get("input")
            if moe_input is None:
                raise RuntimeError(f"Failed to capture MoE input for block {block_idx}")

            self._process_moe_activations(
                block_idx, moe_module, moe_input,
                target_device, attention_mask=attention_mask,
                input_ids=_input_ids,
            )
            del moe_input
            captured_moe_input.clear()

        moe_hook_handle = moe_module.register_forward_hook(_capture_moe_input_hook)

        try:
            return self._forward_block(
                block_idx,
                before_forward=_before_forward,
                after_forward=_after_forward,
            )
        finally:
            if moe_hook_handle is not None:
                moe_hook_handle.remove()

    @torch.inference_mode()
    def _process_moe_activations(
        self,
        block_idx: int,
        moe_module: nn.Module,
        input_hidden_states: torch.Tensor,
        device: torch.device,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ):
        num_experts = moe_module.experts.num_experts
        top_k = moe_module.gate.top_k

        batch_size, sequence_length, hidden_dim = input_hidden_states.shape
        flat_input = input_hidden_states.view(-1, hidden_dim)

        valid_token_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            if attention_mask.dim() == 4:
                mask_row = attention_mask[:, 0, -1, :]
                if mask_row.dtype == torch.bool:
                    valid_token_mask = mask_row
                else:
                    valid_token_mask = mask_row == 0
            elif attention_mask.dim() == 2:
                valid_token_mask = attention_mask.bool()
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Unexpected attention_mask shape {attention_mask.shape}, ignoring"
                )

            if valid_token_mask is not None:
                valid_token_mask = valid_token_mask.reshape(-1)

        if block_idx not in self.state:
            self.state[block_idx] = self._initialize_block_state(num_experts)

        # Detect hash router — two checks because DeepseekV4HashRouter doesn't
        # expose an `is_hash` attribute despite being a hash-based router. The
        # class-name fallback catches it for V4 Flash.
        is_hash = hasattr(moe_module.gate, 'is_hash') and moe_module.gate.is_hash
        if not is_hash and type(moe_module.gate).__name__ == 'DeepseekV4HashRouter':
            is_hash = True

        # Call router
        if is_hash:
            if input_ids is None:
                raise ValueError(
                    f"Hash router at block {block_idx} requires input_ids, "
                    f"but none were provided"
                )
            router_result = moe_module.gate(flat_input, input_ids)
        else:
            router_result = moe_module.gate(flat_input)

        if isinstance(router_result, tuple):
            router_logits = router_result[0]
            selected_experts = router_result[2]
        else:
            raise ValueError(
                f"Unexpected router output type at block {block_idx}: "
                f"{type(router_result)}"
            )

        selected_experts = selected_experts.to(device)

        # Compute total valid tokens for this batch
        if valid_token_mask is not None:
            total_tokens = valid_token_mask.sum()
        else:
            total_tokens = torch.tensor(flat_input.shape[0], device="cpu", dtype=torch.long)
        self.state[block_idx]["total_tokens"] += total_tokens

        # Choose processing mode
        if self._expert_batch_size >= 1:
            self._process_moe_activations_batched(
                block_idx, moe_module, flat_input, device,
                router_logits, selected_experts, valid_token_mask,
                num_experts,
            )
        else:
            # Incremental expert loop — one expert at a time (original behavior)
            for expert_idx in range(num_experts):
                gate_up = F.linear(
                    flat_input, moe_module.experts.gate_up_proj[expert_idx]
                )
                gate, up = gate_up.chunk(2, dim=-1)

                if hasattr(moe_module.experts, 'act_fn'):
                    act_fn = moe_module.experts.act_fn
                else:
                    act_fn = F.silu

                limit = getattr(moe_module.experts, 'limit', 10.0)
                hidden = act_fn(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
                expert_output = F.linear(
                    hidden, moe_module.experts.down_proj[expert_idx]
                )

                update_pruning_state_single_expert(
                    self.state[block_idx],
                    expert_idx,
                    expert_output,
                    router_logits,
                    selected_experts,
                    valid_token_mask=valid_token_mask,
                    renormalize_router_weights=self.hook_config.renormalize_router_weights,
                )

                del gate_up, gate, up, hidden, expert_output

                if expert_idx % 32 == 0:
                    gc.collect()

        del flat_input, router_logits, selected_experts
        if valid_token_mask is not None:
            del valid_token_mask
        gc.collect()

    @torch.inference_mode()
    def _process_moe_activations_batched(
        self,
        block_idx: int,
        moe_module: nn.Module,
        flat_input: torch.Tensor,
        _device: torch.device,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        num_experts: int,
    ):
        """Process experts in groups to reduce kernel launch overhead.

        Instead of 256 individual F.linear calls (one per expert), groups experts
        into chunks of expert_batch_size. Each group does one F.linear on
        [batch_size_e, B*S, D] tensors, then slices per-expert outputs for metrics.
        """
        gate_up_weight = moe_module.experts.gate_up_proj  # [E, 2*D, hidden]
        down_weight = moe_module.experts.down_proj          # [E, hidden, D]
        act_fn = getattr(moe_module.experts, 'act_fn', F.silu)
        limit = getattr(moe_module.experts, 'limit', 10.0)
        batch_size = self._expert_batch_size

        for start in range(0, num_experts, batch_size):
            end = min(start + batch_size, num_experts)
            current_batch_size = end - start

            # [current_batch_size, B*S, 2*D]
            batch_gate_up = torch.matmul(flat_input, gate_up_weight[start:end].transpose(-2, -1))
            # Split into gate (first half) and up (second half): 2 * [current_batch_size, B*S, D]
            batch_gate, batch_up = batch_gate_up.chunk(2, dim=-1)
            del batch_gate_up

            # [current_batch_size, B*S, D]
            batch_hidden = act_fn(batch_gate.clamp(max=limit)) * batch_up.clamp(min=-limit, max=limit)
            del batch_gate, batch_up

            # [current_batch_size, B*S, hidden]
            batch_output = torch.matmul(batch_hidden, down_weight[start:end].transpose(-2, -1))
            del batch_hidden

            # Slice per-expert outputs and update metrics
            for i in range(current_batch_size):
                expert_idx = start + i
                update_pruning_state_single_expert(
                    self.state[block_idx],
                    expert_idx,
                    batch_output[i],
                    router_logits,
                    selected_experts,
                    valid_token_mask=valid_token_mask,
                    renormalize_router_weights=self.hook_config.renormalize_router_weights,
                )

            del batch_output
            gc.collect()


def register_v4_standard_hooks(model, hook_config, state):
    """Register forward hooks on V4 gate submodules for standalone/custom usage.

    **Not part of the automatic pipeline.** This function is for standalone or
    custom observer scripts that bypass `LayerwiseMoEObserver`. The automatic
    pipeline uses `DeepseekV4MoEObserver` (via `layerwise_prune.py`) instead.

    V4's DeepseekV4SparseMoeBlock doesn't emit router logits in its forward output,
    so we hook the gate submodule directly to capture them.

    Args:
        model: The V4 model
        hook_config: MoETransformerObserverConfig instance
        state: Dictionary to populate with per-layer metrics

    Returns:
        List of registered hook handles
    """
    from reap.pruning_metrics import initialize_pruning_state

    hooks = []

    for name, module in model.named_modules():
        module_cls_name = module.__class__.__name__
        if module_cls_name not in ("DeepseekV4TopKRouter", "DeepseekV4HashRouter"):
            continue

        layer_number = int(re.search(r"\d+", name).group(0))

        moe_block_name = re.sub(r"\.gate$", "", name)
        moe_module = model.get_submodule(moe_block_name)

        num_experts = moe_module.experts.num_experts
        top_k = moe_module.gate.top_k

        if layer_number not in state:
            state[layer_number] = initialize_pruning_state(num_experts)

        @torch.no_grad()
        def _make_hook_fn(
            _moe=moe_module,
            _layer=layer_number,
            _num_experts=num_experts,
            _top_k=top_k,
            _is_hash=hasattr(module, 'is_hash') and module.is_hash,
        ):
            def _hook_fn(_, args, output):
                input_hidden = args[0]
                device = input_hidden.device
                batch_size, seq_len, hidden_dim = input_hidden.shape
                flat_input = input_hidden.reshape(-1, hidden_dim)

                router_logits = output[0]
                indices = output[2].to(device)

                for expert_idx in range(_num_experts):
                    gate_up = F.linear(
                        flat_input, _moe.experts.gate_up_proj[expert_idx]
                    )
                    gate, up = gate_up.chunk(2, dim=-1)
                    act_fn = getattr(_moe.experts, 'act_fn', F.silu)
                    limit = getattr(_moe.experts, 'limit', 10.0)
                    hidden = act_fn(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
                    expert_output = F.linear(
                        hidden, _moe.experts.down_proj[expert_idx]
                    )

                    update_pruning_state_single_expert(
                        state[_layer],
                        expert_idx,
                        expert_output,
                        router_logits,
                        indices,
                        renormalize_router_weights=hook_config.renormalize_router_weights,
                    )

                    del gate_up, gate, up, hidden, expert_output

                del flat_input, router_logits, indices

            return _hook_fn

        hook_fn = _make_hook_fn()
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

    return hooks
