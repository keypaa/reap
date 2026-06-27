"""Tests for V4 pipeline dispatch helpers.

Observer dispatch logic (`if _is_v4_model(model): A else: B`) is verified
via source AST analysis since the layerwise_prune module has heavy deps
(vllm, lm_eval) not installed in test environments.
"""

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock

from reap.layerwise_observer import LayerwiseMoEObserver
from reap.model_util import _is_v4_model, _is_v4_model_from_name
from reap.v4_block_loader import V4BlockDiskLoader
from reap.v4_moe_observer import DeepseekV4MoEObserver


def _get_source(name: str) -> str:
    """Return source of a function or method, resolving to file content."""
    mod_path = Path("src/reap/layerwise_prune.py")
    return mod_path.read_text()


def _function_has_call(func_source: str, name: str) -> bool:
    """Check if a function source contains a call to `name`."""
    tree = ast.parse(func_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == name:
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == name:
                return True
    return False


class TestIsV4ModelFromName:
    def test_v4_full_name(self):
        assert _is_v4_model_from_name("deepseek-ai/DeepSeek-V4-Flash")
        assert _is_v4_model_from_name("deepseek-ai/DeepSeek-V4-Lite")
        assert _is_v4_model_from_name("deepseek-ai/DeepSeek-V4-0224")

    def test_v4_lowercase(self):
        assert _is_v4_model_from_name("deepseek-ai/deepseek-v4-flash")

    def test_not_v4(self):
        assert not _is_v4_model_from_name("deepseek-ai/DeepSeek-V2-Lite-Chat")
        assert not _is_v4_model_from_name("Qwen/Qwen3-30B-A3B")
        assert not _is_v4_model_from_name("mistralai/Mixtral-8x7B-v0.1")
        assert not _is_v4_model_from_name("")

    def test_v4_substring_not_fooled(self):
        assert not _is_v4_model_from_name("deepseek-ai/DeepSeek-V2-Lite")
        assert not _is_v4_model_from_name("XDeepSeek-V3-Y")


class TestIsV4Model:
    def test_v4_model_class(self):
        model = MagicMock()
        model.__class__.__name__ = "DeepseekV4ForCausalLM"
        assert _is_v4_model(model)

    def test_v4_lite_model_class(self):
        model = MagicMock()
        model.__class__.__name__ = "DeepseekV4LiteForCausalLM"
        assert _is_v4_model(model)

    def test_non_v4_model(self):
        model = MagicMock()
        model.__class__.__name__ = "Qwen3MoeForCausalLM"
        assert not _is_v4_model(model)

    def test_mixtral_model(self):
        model = MagicMock()
        model.__class__.__name__ = "MixtralForCausalLM"
        assert not _is_v4_model(model)


class TestV4BlockLoaderLoadNonBackbone:
    def test_accepts_model_parameter(self):
        params = inspect.signature(
            V4BlockDiskLoader.load_non_backbone_modules
        ).parameters
        assert "model" in params
        assert params["model"].default is None

    def test_sets_modules_on_model_when_provided(self):
        source = inspect.getsource(V4BlockDiskLoader.load_non_backbone_modules)
        assert "model is not None" in source
        assert "model.model.embed_tokens = embed" in source
        assert "model.model.norm = norm" in source
        assert "model.lm_head = lm" in source


class TestModelUtilExports:
    def test_model_util_has_both_helpers(self):
        assert callable(_is_v4_model)
        assert callable(_is_v4_model_from_name)

    def test_layerwise_prune_imports_helpers_from_model_util(self):
        source = Path("src/reap/layerwise_prune.py").read_text()
        assert "_is_v4_model" in source
        assert "_is_v4_model_from_name" in source
        assert "from reap.model_util import" in source


class TestRecordActivationsLayerwiseDispatch:
    """Verify dispatch pattern exists via AST analysis."""

    _FILE = Path("src/reap/layerwise_prune.py")

    def test_has_v4_dispatch_in_record_activations(self):
        source = self._FILE.read_text()
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "record_activations_layerwise":
                body_source = ast.get_source_segment(source, node)
                found = "_is_v4_model(model)" in body_source and "DeepseekV4MoEObserver" in body_source
                break
        assert found, "record_activations_layerwise missing V4 observer dispatch"

    def test_has_v4_dispatch_in_main_model_loading(self):
        source = self._FILE.read_text()
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                body_source = ast.get_source_segment(source, node)
                found = (
                    "_is_v4_model_from_name(model_name)" in body_source
                    and "V4BlockDiskLoader" in body_source
                )
                break
        assert found, "main() missing V4 model loading dispatch"

    def test_has_v4_dispatch_in_pruning_reload(self):
        source = self._FILE.read_text()
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                body_source = ast.get_source_segment(source, node)
                # Count occurrences of _is_v4_model_from_name
                import re
                matches = re.findall(r"_is_v4_model_from_name\(model_name\)", body_source)
                found = len(matches) >= 2  # model loading + pruning reload
                break
        assert found, "main() missing V4 dispatch for pruning reload"


class TestObserverClassRelationship:
    def test_v4_observer_is_subclass_of_base(self):
        assert issubclass(DeepseekV4MoEObserver, LayerwiseMoEObserver)

    def test_v4_observer_overrides_process_moe_activations(self):
        assert DeepseekV4MoEObserver._process_moe_activations is not LayerwiseMoEObserver._process_moe_activations
