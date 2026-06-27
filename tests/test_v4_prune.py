import torch
import torch.nn as nn

from reap.v4_prune_utils import _prune_v4_layer, _remap_hash_router_tid2eid


class MockDeepseekV4Experts(nn.Module):
    def __init__(self, num_experts=8, hidden_dim=64, intermediate_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim)
        )


class MockDeepseekV4TopKRouter(nn.Module):
    def __init__(self, num_experts=8, hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = 2
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim))
        self.e_score_correction_bias = torch.zeros(num_experts)


class MockDeepseekV4HashRouter(nn.Module):
    def __init__(self, num_experts=8, hidden_dim=64, vocab_size=100):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = 2
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim))
        self.is_hash = True
        self.register_buffer(
            "tid2eid",
            torch.randint(0, num_experts, (vocab_size, 2), dtype=torch.long),
            persistent=True,
        )


class MockDeepseekV4MLP(nn.Module):
    def __init__(self, hidden_dim=64, intermediate_dim=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim)


class MockDeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, num_experts=8, hidden_dim=64, intermediate_dim=128, use_hash=False, vocab_size=100):
        super().__init__()
        self.experts = MockDeepseekV4Experts(num_experts, hidden_dim, intermediate_dim)
        if use_hash:
            self.gate = MockDeepseekV4HashRouter(num_experts, hidden_dim, vocab_size)
        else:
            self.gate = MockDeepseekV4TopKRouter(num_experts, hidden_dim)
        self.shared_experts = MockDeepseekV4MLP(hidden_dim, intermediate_dim)


class MockDeepseekV4ForCausalLM(nn.Module):
    def __init__(self, num_layers=1, num_experts=8, hidden_dim=64, intermediate_dim=128):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "n_routed_experts": num_experts,
                "num_local_experts": num_experts,
            },
        )()
        self.model = type("Model", (), {"layers": []})()
        self.model.layers = nn.ModuleList()
        self._class_name = "DeepseekV4ForCausalLM"

    @property
    def __class__(self):
        return type(self._class_name, (), {"__name__": self._class_name})


class TestPruneV4Layer:
    def test_prune_v4_layer_shapes(self):
        num_experts = 8
        hidden_dim = 64
        intermediate_dim = 128
        moe_block = MockDeepseekV4SparseMoeBlock(num_experts, hidden_dim, intermediate_dim)
        model = MockDeepseekV4ForCausalLM(num_experts=num_experts, hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)

        retained_indices = [0, 2, 4, 6]

        _prune_v4_layer(moe_block, retained_indices)

        assert moe_block.experts.gate_up_proj.shape == (4, 2 * intermediate_dim, hidden_dim)
        assert moe_block.experts.down_proj.shape == (4, hidden_dim, intermediate_dim)
        assert moe_block.experts.num_experts == 4
        assert moe_block.gate.weight.shape == (4, hidden_dim)
        assert moe_block.gate.num_experts == 4

        # Verify shapes match retained count
        assert moe_block.experts.gate_up_proj.shape[0] == len(retained_indices)
        assert moe_block.experts.down_proj.shape[0] == len(retained_indices)

    def test_prune_v4_layer_no_retained(self):
        num_experts = 4
        moe_block = MockDeepseekV4SparseMoeBlock(num_experts)
        model = MockDeepseekV4ForCausalLM(num_experts=num_experts)

        retained_indices = [0]

        _prune_v4_layer(moe_block, retained_indices)

        assert moe_block.experts.num_experts == 1
        assert moe_block.gate.num_experts == 1

    def test_prune_v4_layer_keeps_correct_weights(self):
        num_experts = 4
        hidden_dim = 8
        intermediate_dim = 16
        moe_block = MockDeepseekV4SparseMoeBlock(num_experts, hidden_dim, intermediate_dim)
        model = MockDeepseekV4ForCausalLM(num_experts=num_experts, hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)

        original_gate_up = moe_block.experts.gate_up_proj.data.clone()
        original_down = moe_block.experts.down_proj.data.clone()
        original_gate_weight = moe_block.gate.weight.data.clone()

        retained_indices = [1, 3]

        _prune_v4_layer(moe_block, retained_indices)

        assert torch.equal(moe_block.experts.gate_up_proj[0], original_gate_up[1])
        assert torch.equal(moe_block.experts.gate_up_proj[1], original_gate_up[3])
        assert torch.equal(moe_block.experts.down_proj[0], original_down[1])
        assert torch.equal(moe_block.experts.down_proj[1], original_down[3])
        assert torch.equal(moe_block.gate.weight[0], original_gate_weight[1])
        assert torch.equal(moe_block.gate.weight[1], original_gate_weight[3])


class TestRemapHashRouterTid2Eid:
    def test_remap_all_retained(self):
        gate = MockDeepseekV4HashRouter(num_experts=4, vocab_size=10)
        original_tid2eid = gate.tid2eid.data.clone()

        old_to_new = [0, 1, 2, 3]

        _remap_hash_router_tid2eid(gate, old_to_new)

        assert torch.equal(gate.tid2eid.data, original_tid2eid)

    def test_remap_half_retained(self):
        num_experts = 4
        gate = MockDeepseekV4HashRouter(num_experts=num_experts, vocab_size=20)
        original_tid2eid = gate.tid2eid.data.clone()

        old_to_new = [0, -1, 1, -1]
        _remap_hash_router_tid2eid(gate, old_to_new)

        for vocab_idx in range(20):
            for k in range(2):
                old_val = original_tid2eid[vocab_idx, k].item()
                new_val = gate.tid2eid[vocab_idx, k].item()
                if old_val == -1:
                    assert new_val == -1, f"Unused TID at ({vocab_idx},{k}) should stay -1"
                elif old_val in (0, 2):
                    expected = old_to_new[old_val]
                    assert new_val == expected, (
                        f"Vocab {vocab_idx}, expert {old_val} should "
                        f"remap to {expected}, got {new_val}"
                    )
                else:
                    assert new_val == 0, (
                        f"Pruned expert {old_val} at ({vocab_idx},{k}) "
                        f"should fall back to 0, got {new_val}"
                    )

    def test_remap_with_unused_tids(self):
        num_experts = 4
        gate = MockDeepseekV4HashRouter(num_experts=num_experts, vocab_size=10)
        gate.tid2eid.data[0] = torch.tensor([-1, -1])
        gate.tid2eid.data[5] = torch.tensor([-1, -1])

        _remap_hash_router_tid2eid(gate, [0, -1, 1, -1])

        assert gate.tid2eid.data[0, 0].item() == -1
        assert gate.tid2eid.data[0, 1].item() == -1
        assert gate.tid2eid.data[5, 0].item() == -1

    def test_remap_tid2eid_buffer_preserved(self):
        gate = MockDeepseekV4HashRouter(num_experts=4, vocab_size=10)
        old_buffer_id = id(gate.tid2eid)
        _remap_hash_router_tid2eid(gate, [0, 1, 2, 3])
        assert id(gate.tid2eid) == old_buffer_id


class TestEScoreCorrectionBias:
    def test_bias_pruned_with_topk_router(self):
        num_experts = 8
        moe_block = MockDeepseekV4SparseMoeBlock(num_experts)
        model = MockDeepseekV4ForCausalLM(num_experts=num_experts)

        original_bias = moe_block.gate.e_score_correction_bias.data.clone()

        retained_indices = [0, 2, 4, 6]

        _prune_v4_layer(moe_block, retained_indices)

        assert moe_block.gate.e_score_correction_bias.shape == (4,)
        assert torch.equal(
            moe_block.gate.e_score_correction_bias[0],
            original_bias[0],
        )
        assert torch.equal(
            moe_block.gate.e_score_correction_bias[1],
            original_bias[2],
        )

    def test_bias_guard_no_crash_without_bias(self):
        gate = MockDeepseekV4HashRouter(num_experts=8)
        assert not hasattr(gate, "e_score_correction_bias"), (
            "Hash router should not have e_score_correction_bias"
        )
        moe_block = MockDeepseekV4SparseMoeBlock(num_experts=8)
        moe_block.gate = gate
        model = MockDeepseekV4ForCausalLM(num_experts=8)

        _prune_v4_layer(moe_block, [0, 1, 2, 3])
        assert moe_block.experts.num_experts == 4

    def test_bias_guard_no_crash_without_device(self):
        gate = nn.Module()
        gate.num_experts = 8
        gate.weight = nn.Parameter(torch.randn(8, 64))
        assert not hasattr(gate, "e_score_correction_bias")

        moe_block = MockDeepseekV4SparseMoeBlock(num_experts=8)
        moe_block.gate = gate
        model = MockDeepseekV4ForCausalLM(num_experts=8)

        _prune_v4_layer(moe_block, [0, 1, 2, 3])
        assert moe_block.gate.weight.shape == (4, 64)


class TestConfigUpdate:
    def test_config_fields_updated(self):
        num_experts = 8
        moe_block = MockDeepseekV4SparseMoeBlock(num_experts)
        model = MockDeepseekV4ForCausalLM(num_experts=num_experts)

        retained_indices = [0, 1, 2, 3, 4]

        _prune_v4_layer(moe_block, retained_indices)

        assert model.config.n_routed_experts == 8
        assert model.config.num_local_experts == 8


class TestSharedExpertsUnchanged:
    def test_shared_experts_left_untouched(self):
        num_experts = 8
        moe_block = MockDeepseekV4SparseMoeBlock(num_experts)
        model = MockDeepseekV4ForCausalLM(num_experts=num_experts)

        original_shared = moe_block.shared_experts.state_dict()

        _prune_v4_layer(moe_block, [0, 2, 4, 6])

        for key, param in moe_block.shared_experts.named_parameters():
            assert torch.equal(param.data, original_shared[key])

    def test_shared_experts_modules_preserved(self):
        moe_block = MockDeepseekV4SparseMoeBlock(num_experts=8)
        model = MockDeepseekV4ForCausalLM(num_experts=8)

        _prune_v4_layer(moe_block, [0, 2, 4, 6])

        assert isinstance(moe_block.shared_experts, MockDeepseekV4MLP)
        assert hasattr(moe_block.shared_experts, "gate_proj")
        assert hasattr(moe_block.shared_experts, "up_proj")
        assert hasattr(moe_block.shared_experts, "down_proj")


class TestHashRouterFullRoundtrip:
    def test_hash_router_prune_roundtrip(self):
        num_experts = 6
        hidden_dim = 32
        intermediate_dim = 64
        vocab_size = 50
        moe_block = MockDeepseekV4SparseMoeBlock(
            num_experts, hidden_dim, intermediate_dim, use_hash=True, vocab_size=vocab_size
        )
        model = MockDeepseekV4ForCausalLM(
            num_experts=num_experts, hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )

        retained_indices = [0, 2, 4]

        _prune_v4_layer(moe_block, retained_indices)

        assert moe_block.experts.num_experts == 3
        assert moe_block.experts.gate_up_proj.shape == (3, 2 * intermediate_dim, hidden_dim)
        assert moe_block.experts.down_proj.shape == (3, hidden_dim, intermediate_dim)
        assert not torch.isnan(moe_block.gate.tid2eid.data).any()
        assert moe_block.gate.tid2eid.shape == (vocab_size, 2)
