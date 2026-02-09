"""Tests for the config-driven IR builder."""

import pytest

from src.arch.models_arch.auto import (
    auto_adapter_from_config,
    build_ir_from_config,
    infer_model_type,
    list_registered_builders,
    register_config_builder,
)
from src.arch.models_arch.auto.config_builder import (
    _has_mla,
    _has_moe,
    _infer_kv_cache_type,
)
from src.arch.models_arch.auto.ir import ComputationalGraph


class TestConfigBuilderRegistry:
    """Tests for the config builder registry."""

    def test_list_registered_builders(self):
        builders = list_registered_builders()
        assert isinstance(builders, dict)
        assert len(builders) > 0
        # Should have qwen3_moe, deepseek_v3, qwen3
        assert "qwen3_moe" in builders
        assert "deepseek_v3" in builders
        assert "qwen3" in builders

    def test_get_config_builder_existing(self):
        builder = list_registered_builders().get("qwen3_moe")
        assert builder is not None
        assert builder == "build_qwen3_moe_ir"

    def test_register_config_builder(self):
        # Register a test builder
        @register_config_builder("test_model")
        def build_test_model(config) -> ComputationalGraph:
            return ComputationalGraph(
                model_name="TestModel",
                model_type="dense",
                config={},
            )

        # Verify it was registered
        builders = list_registered_builders()
        assert "test_model" in builders


class TestModelTypeInference:
    """Tests for model type inference."""

    def test_infer_qwen3_moe(self):
        config = {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
        }
        model_type = infer_model_type(config)
        assert model_type == "qwen3_moe"

    def test_infer_deepseek_v3(self):
        config = {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
        }
        model_type = infer_model_type(config)
        assert model_type == "deepseek_v3"

    def test_infer_from_architecture(self):
        # Test inference from architecture name when model_type not registered
        config = {
            "model_type": "some_unknown_type",
            "architectures": ["Qwen3MoeForCausalLM"],
        }
        model_type = infer_model_type(config)
        assert model_type == "qwen3_moe"

    def test_infer_unknown(self):
        config = {"model_type": "unknown_model"}
        model_type = infer_model_type(config)
        # Should return None if not registered
        assert model_type is None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_has_moe_true(self):
        config = {"num_experts": 8}
        assert _has_moe(config) is True

    def test_has_moe_n_routed(self):
        config = {"n_routed_experts": 256}
        assert _has_moe(config) is True

    def test_has_moe_false(self):
        config = {"hidden_size": 4096}
        assert _has_moe(config) is False

    def test_has_mla_true(self):
        config = {"qk_rope_head_dim": 64, "kv_lora_rank": 512}
        assert _has_mla(config) is True

    def test_has_mla_false(self):
        config = {"hidden_size": 4096}
        assert _has_mla(config) is False

    def test_infer_kv_cache_type_mla(self):
        config = {"qk_rope_head_dim": 64, "kv_lora_rank": 512}
        assert _infer_kv_cache_type(config) == "mla"

    def test_infer_kv_cache_type_gqa(self):
        config = {"num_attention_heads": 32, "num_key_value_heads": 8}
        assert _infer_kv_cache_type(config) == "gqa"

    def test_infer_kv_cache_type_mha(self):
        config = {"num_attention_heads": 32}
        assert _infer_kv_cache_type(config) == "mha"


class TestBuildIRFromConfig:
    """Tests for building IR from config."""

    def test_build_qwen3_moe(self):
        config = {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "head_dim": 64,
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1536,
        }

        graph = build_ir_from_config(config, "qwen3_moe")

        assert isinstance(graph, ComputationalGraph)
        assert graph.model_type == "moe"
        assert graph.has_moe is True
        assert graph.kv_cache_type == "gqa"  # num_kv_heads != num_heads

        # Check nodes
        matmul_nodes = graph.get_matmul_nodes()
        assert len(matmul_nodes) > 0

        # Should have qkv_proj, o_proj, moe_gate, moe_up_proj, moe_down_proj
        node_names = [n.name for n in matmul_nodes]
        assert "qkv_proj" in node_names
        assert "o_proj" in node_names
        assert "moe_gate" in node_names
        assert "moe_up_proj" in node_names
        assert "moe_down_proj" in node_names

    def test_build_deepseek_v3(self):
        config = {
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "intermediate_size": 18432,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "v_head_dim": 128,
            "qk_nope_head_dim": 128,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
        }

        graph = build_ir_from_config(config, "deepseek_v3")

        assert isinstance(graph, ComputationalGraph)
        assert graph.model_type == "mla"
        assert graph.has_mla is True
        assert graph.has_moe is True
        assert graph.kv_cache_type == "mla"

    def test_build_qwen3_dense(self):
        config = {
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "head_dim": 128,
        }

        graph = build_ir_from_config(config, "qwen3")

        assert isinstance(graph, ComputationalGraph)
        assert graph.model_type == "dense"
        assert graph.has_moe is False
        assert graph.has_mla is False
        assert graph.kv_cache_type == "mha"  # num_kv_heads == num_heads

    def test_build_auto_infer_model_type(self):
        config = {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "head_dim": 64,
            "num_experts": 128,
            "num_experts_per_tok": 4,
        }

        # Don't pass model_type - should auto-infer
        graph = build_ir_from_config(config)
        assert graph.model_type == "moe"

    def test_build_unknown_model_type(self):
        config = {"model_type": "unknown_model"}

        with pytest.raises(ValueError, match="No config builder registered"):
            build_ir_from_config(config, "unknown_model")

    def test_build_cannot_infer(self):
        config = {"hidden_size": 4096}  # No model_type or architectures

        with pytest.raises(ValueError, match="Cannot infer model_type"):
            build_ir_from_config(config)


class TestAutoAdapterFromConfig:
    """Tests for the high-level auto_adapter_from_config API."""

    def test_auto_adapter_qwen3_moe(self):
        from src.arch.config import ForwardMode, ScheduleConfig

        config = {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "head_dim": 64,
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1536,
        }

        schedule_config = ScheduleConfig(
            mode=ForwardMode.EXTEND,
            tp_size=4,
            dp_size=2,
            ep_size=8,
        )

        model_arch = auto_adapter_from_config(config, schedule_config)

        # Check it's a valid ModelArch
        assert model_arch is not None
        assert hasattr(model_arch, "operators")
        assert hasattr(model_arch, "model_config")
        assert len(model_arch.operators) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
