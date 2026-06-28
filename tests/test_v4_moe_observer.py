import torch
import torch.nn as nn
import torch.nn.functional as F

from reap.metrics import OnlineStatsTracker
from reap.pruning_metrics import initialize_pruning_state, update_pruning_state_single_expert
from reap.v4_moe_observer import DeepseekV4MoEObserver, register_v4_standard_hooks
from reap.layerwise_observer import LayerwiseMoEObserver, ReplayBatch, ReplayCache


class TestOnlineStatsPartialUpdate:
    def test_partial_update_one_expert(self):
        tracker = OnlineStatsTracker(shape=(3,), count_shape=(3,), device="cpu", dtype=torch.float32)
        tracker._partial_update(0, torch.tensor(10.0), torch.tensor(2))
        assert tracker.count[0].item() == 2
        assert tracker.mean[0].item() == 10.0
        assert tracker.count[1].item() == 0
        assert tracker.count[2].item() == 0

    def test_partial_update_multiple_updates(self):
        tracker = OnlineStatsTracker(shape=(3,), count_shape=(3,), device="cpu", dtype=torch.float32)
        tracker._partial_update(0, torch.tensor(10.0), torch.tensor(2))
        tracker._partial_update(0, torch.tensor(20.0), torch.tensor(2))
        assert tracker.count[0].item() == 4
        assert tracker.mean[0].item() == 15.0
        assert tracker.count[1].item() == 0

    def test_partial_update_multiple_experts(self):
        tracker = OnlineStatsTracker(shape=(3,), count_shape=(3,), device="cpu", dtype=torch.float32)
        tracker._partial_update(0, torch.tensor(10.0), torch.tensor(2))
        tracker._partial_update(1, torch.tensor(30.0), torch.tensor(3))
        assert tracker.count[0].item() == 2
        assert tracker.mean[0].item() == 10.0
        assert tracker.count[1].item() == 3
        assert tracker.mean[1].item() == 30.0
        assert tracker.count[2].item() == 0

    def test_partial_update_does_not_affect_other_indices(self):
        tracker = OnlineStatsTracker(shape=(3,), count_shape=(3,), device="cpu", dtype=torch.float32)
        tracker.count[0] = 5
        tracker.mean[0] = 50.0
        tracker._partial_update(2, torch.tensor(100.0), torch.tensor(1))
        assert tracker.count[0].item() == 5
        assert tracker.mean[0].item() == 50.0
        assert tracker.count[2].item() == 1
        assert tracker.mean[2].item() == 100.0

    def test_partial_update_zero_count(self):
        tracker = OnlineStatsTracker(shape=(2,), count_shape=(2,), device="cpu", dtype=torch.float32)
        tracker._partial_update(0, torch.tensor(5.0), torch.tensor(0))
        assert tracker.count[0].item() == 0
        assert tracker.mean[0].item() == 0.0

    def test_partial_update_with_existing_mean(self):
        tracker = OnlineStatsTracker(shape=(2,), count_shape=(2,), device="cpu", dtype=torch.float32)
        tracker.count[0] = 5
        tracker.mean[0] = 10.0
        tracker._partial_update(0, torch.tensor(20.0), torch.tensor(5))
        assert tracker.count[0].item() == 10
        assert tracker.mean[0].item() == 15.0


class TestUpdatePruningStateSingleExpert:
    def test_single_expert_basic(self):
        state = initialize_pruning_state(2)

        expert_output = torch.tensor([[3.0, 4.0], [1.0, 0.0], [5.0, 12.0]], dtype=torch.float32)
        selected_experts = torch.tensor([[0], [1], [0]], dtype=torch.long)
        router_logits = torch.tensor([[2.0, 1.0], [0.0, 3.0], [4.0, 0.0]], dtype=torch.float32)

        update_pruning_state_single_expert(
            state, 1, expert_output, router_logits, selected_experts
        )

        assert state["expert_frequency"][1].item() == 1
        assert state["expert_frequency"][0].item() == 0
        assert state["ean_sum"][1].item() > 0

    def test_both_experts_independently(self):
        state = initialize_pruning_state(2)

        expert_output = torch.tensor([[3.0, 4.0], [1.0, 0.0], [5.0, 12.0]], dtype=torch.float32)
        selected_experts = torch.tensor([[0], [1], [0]], dtype=torch.long)
        router_logits = torch.tensor([[2.0, 1.0], [0.0, 3.0], [4.0, 0.0]], dtype=torch.float32)

        update_pruning_state_single_expert(
            state, 0, expert_output, router_logits, selected_experts
        )
        update_pruning_state_single_expert(
            state, 1, expert_output, router_logits, selected_experts
        )

        assert state["expert_frequency"][0].item() == 2
        assert state["expert_frequency"][1].item() == 1
        assert state["ean_sum"][0].item() > 0
        assert state["ean_sum"][1].item() > 0

    def test_no_active_tokens(self):
        state = initialize_pruning_state(3)

        expert_output = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        selected_experts = torch.tensor([[0]], dtype=torch.long)
        router_logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)

        update_pruning_state_single_expert(
            state, 2, expert_output, router_logits, selected_experts
        )

        assert state["expert_frequency"][2].item() == 0
        assert state["ean_sum"][2].item() == 0.0

    def test_with_valid_token_mask_filters_padding(self):
        state = initialize_pruning_state(2)

        expert_output = torch.tensor(
            [[3.0, 4.0], [1.0, 0.0], [5.0, 12.0]], dtype=torch.float32
        )
        selected_experts = torch.tensor([[0], [0], [1]], dtype=torch.long)
        router_logits = torch.tensor([[2.0, 1.0], [4.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
        valid_token_mask = torch.tensor([True, False, True])

        update_pruning_state_single_expert(
            state, 0, expert_output, router_logits, selected_experts,
            valid_token_mask=valid_token_mask,
        )

        assert state["expert_frequency"][0].item() == 1

    def test_ean_norm_computation(self):
        state = initialize_pruning_state(1)

        expert_output = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        selected_experts = torch.tensor([[0]], dtype=torch.long)
        router_logits = torch.tensor([[2.0]], dtype=torch.float32)

        update_pruning_state_single_expert(
            state, 0, expert_output, router_logits, selected_experts
        )

        expected_norm = torch.linalg.norm(torch.tensor([3.0, 4.0])).item()
        assert torch.allclose(state["ean_sum"][0], torch.tensor(expected_norm, dtype=torch.float64))
        assert torch.allclose(state["ean_mean"].mean[0], torch.tensor(expected_norm, dtype=torch.float32))

    def test_topk_selection(self):
        state = initialize_pruning_state(3)

        expert_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        selected_experts = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        router_logits = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]], dtype=torch.float32)

        update_pruning_state_single_expert(
            state, 0, expert_output, router_logits, selected_experts
        )

        assert state["expert_frequency"][0].item() == 1
        update_pruning_state_single_expert(
            state, 1, expert_output, router_logits, selected_experts
        )
        assert state["expert_frequency"][1].item() == 2
        update_pruning_state_single_expert(
            state, 2, expert_output, router_logits, selected_experts
        )
        assert state["expert_frequency"][2].item() == 1


class TestDeepseekV4MoEObserver:
    def test_class_imports(self):
        assert DeepseekV4MoEObserver is not None

    def test_is_subclass_of_layerwise_observer(self):
        assert issubclass(DeepseekV4MoEObserver, LayerwiseMoEObserver)

    def test_register_v4_standard_hooks_exists(self):
        assert callable(register_v4_standard_hooks)


class TestReplayBatchInputIds:
    def test_replay_batch_has_input_ids_field(self):
        batch = ReplayBatch(
            inputs=[torch.tensor([1, 2, 3])],
            kwargs={},
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        )
        assert batch.input_ids is not None
        assert torch.equal(batch.input_ids, torch.tensor([[1, 2, 3]], dtype=torch.long))

    def test_replay_cache_append_materialize_input_ids(self):
        cache = ReplayCache()
        input_ids = torch.tensor([[4, 5, 6]], dtype=torch.long)
        cache.append(
            inputs=[torch.tensor([1.0, 2.0, 3.0])],
            kwargs={},
            attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
            position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
            input_ids=input_ids,
        )
        assert len(cache) == 1

        _, materialized_kwargs = cache.materialize(0, torch.device("cpu"))
        assert "input_ids" in materialized_kwargs
        assert torch.equal(materialized_kwargs["input_ids"], input_ids)

    def test_replay_cache_input_ids_default_none(self):
        cache = ReplayCache()
        cache.append(
            inputs=[torch.tensor([1.0, 2.0, 3.0])],
            kwargs={},
        )
        _, materialized_kwargs = cache.materialize(0, torch.device("cpu"))
        assert "input_ids" not in materialized_kwargs


class MockV4Config:
    class Experts:
        def __init__(self, num=8):
            self.num_experts = num
            hidden = 64
            d = 32
            self.gate_up_proj = nn.Parameter(torch.randn(num, 2 * d, hidden))
            self.down_proj = nn.Parameter(torch.randn(num, hidden, d))
            self.act_fn = F.silu
            self.limit = 10.0

    class Gate:
        def __init__(self, num=8):
            self.top_k = 2
            self.is_hash = False
            self.weight = nn.Parameter(torch.randn(num, 64))

    def __init__(self, num=8):
        self.experts = self.Experts(num)
        self.gate = self.Gate(num)


def test_batched_experts_match_incremental():
    """Batched expert mode must produce identical metrics to incremental."""
    num_experts = 8
    hidden_dim = 64
    bs, seq = 2, 16

    moe = MockV4Config(num_experts)
    flat_input = torch.randn(bs * seq, hidden_dim)
    router_logits = torch.randn(bs * seq, num_experts)
    selected_experts = torch.randint(0, num_experts, (bs * seq, 2))

    # Capture incremental mode results
    state_inc = initialize_pruning_state(num_experts)
    for idx in range(num_experts):
        gate_up = F.linear(flat_input, moe.experts.gate_up_proj[idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        expert_output = F.linear(hidden, moe.experts.down_proj[idx])
        update_pruning_state_single_expert(
            state_inc, idx, expert_output, router_logits, selected_experts,
        )
        del gate_up, gate, up, hidden, expert_output

    # Capture batched mode results
    state_batch = initialize_pruning_state(num_experts)
    batch_size = 4
    gate_up_weight = moe.experts.gate_up_proj
    down_weight = moe.experts.down_proj
    for start in range(0, num_experts, batch_size):
        end = min(start + batch_size, num_experts)
        bg = torch.matmul(flat_input, gate_up_weight[start:end].transpose(-2, -1))
        bg_gate, bg_up = bg.chunk(2, dim=-1)
        del bg
        bh = F.silu(bg_gate) * bg_up
        del bg_gate, bg_up
        bo = torch.matmul(bh, down_weight[start:end].transpose(-2, -1))
        del bh
        for i in range(end - start):
            update_pruning_state_single_expert(
                state_batch, start + i, bo[i], router_logits, selected_experts,
            )
        del bo

    # Compare all metrics
    for key in state_inc:
        if isinstance(state_inc[key], torch.Tensor):
            assert torch.allclose(state_inc[key], state_batch[key], atol=1e-5), \
                f"Mismatch in {key}: inc={state_inc[key]}, batch={state_batch[key]}"
        elif hasattr(state_inc[key], 'mean') and hasattr(state_inc[key], 'count'):
            assert torch.allclose(state_inc[key].mean, state_batch[key].mean, atol=1e-5), \
                f"Mismatch OnlineStatsTracker.mean in {key}"
        elif key == "total_tokens":
            assert state_inc[key] == state_batch[key]

    print("Batched and incremental modes produce identical metrics.")
