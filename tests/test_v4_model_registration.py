from transformers import DeepseekV4Config

from reap.model_util import MODEL_ATTRS, _is_v4_model
from reap.observer import (
    DeepseekV4MoEObserverHookConfig,
    OBSERVER_CONFIG_REGISTRY,
)


class TestModelAttrs:
    def test_model_attrs_v4(self):
        entry = MODEL_ATTRS["DeepseekV4ForCausalLM"]
        assert entry["moe_block"] == "mlp"
        assert entry["gate_proj"] == "gate_proj"
        assert entry["up_proj"] == "up_proj"
        assert entry["down_proj"] == "down_proj"
        assert entry["experts"] == "experts"
        assert entry["fused"] is False
        assert entry["router"] == "gate"
        assert entry["num_experts"] == "num_local_experts"
        assert entry["num_experts_per_tok"] == "num_experts_per_tok"

    def test_is_v4_model(self):
        model = type("DeepseekV4ForCausalLM", (), {"__class__": type("MockCls", (), {"__name__": "DeepseekV4ForCausalLM"})()})()
        assert _is_v4_model(model)

    def test_is_v4_model_other(self):
        qwen = type("Qwen3MoeForCausalLM", (), {"__class__": type("MockCls", (), {"__name__": "Qwen3MoeForCausalLM"})()})()
        mixtral = type("MixtralForCausalLM", (), {"__class__": type("MockCls", (), {"__name__": "MixtralForCausalLM"})()})()
        assert not _is_v4_model(qwen)
        assert not _is_v4_model(mixtral)


class TestObserverConfig:
    def test_v4_observer_config(self):
        config = DeepseekV4MoEObserverHookConfig()
        assert config.module_class_name_to_hook_regex == "DeepseekV4SparseMoeBlock"
        assert config.num_experts_attr_name == "experts.num_experts"
        assert config.top_k_attr_name == "gate.top_k"
        assert config.fused_experts is False

    def test_registry_contains_v4(self):
        assert "DeepseekV4ForCausalLM" in OBSERVER_CONFIG_REGISTRY
        assert OBSERVER_CONFIG_REGISTRY["DeepseekV4ForCausalLM"] is DeepseekV4MoEObserverHookConfig
