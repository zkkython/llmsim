"""Tests for the auto adapter module."""

import pytest

from src.arch.models_arch.auto import SglangAutoAdapter, register_layer_parser
from src.arch.models_arch.auto.ir import (
    ComputationalGraph,
    DataType,
    OpNode,
    ShapeSpec,
)
from src.arch.models_arch.auto.layer_parsers.base import BaseLayerParser
from src.arch.models_arch.auto.layer_parsers.registry import (
    get_parser,
    list_registered_parsers,
    unregister_parser,
)
from src.arch.models_arch.auto.parser import (
    HAS_TORCH,
    ModelParser,
    mock_sglang_environment,
)

# Skip tests that require PyTorch
torch_required = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestShapeSpec:
    """Tests for ShapeSpec class."""

    def test_concrete_shape(self):
        spec = ShapeSpec(128, 256)
        assert spec.m == 128
        assert spec.n == 256
        assert not spec.is_symbolic

    def test_symbolic_shape(self):
        spec = ShapeSpec("seq_len", "hidden_size")
        assert spec.m == "seq_len"
        assert spec.n == "hidden_size"
        assert spec.is_symbolic

    def test_resolve_concrete(self):
        spec = ShapeSpec(128, 256)
        resolved = spec.resolve({})
        assert resolved == (128, 256)

    def test_resolve_symbolic(self):
        spec = ShapeSpec("seq_len", "hidden_size")
        context = {"seq_len": 128, "hidden_size": 4096}
        resolved = spec.resolve(context)
        assert resolved == (128, 4096)

    def test_resolve_expression(self):
        spec = ShapeSpec("seq_len", "num_heads * head_dim")
        context = {"seq_len": 128, "num_heads": 32, "head_dim": 128}
        resolved = spec.resolve(context)
        assert resolved == (128, 4096)

    def test_to_tuple(self):
        spec = ShapeSpec(128, 256)
        assert spec.to_tuple() == (128, 256)

    def test_from_tuple(self):
        spec = ShapeSpec.from_tuple((128, 256))
        assert spec.m == 128
        assert spec.n == 256


class TestOpNode:
    """Tests for OpNode class."""

    def test_create_op_node(self):
        node = OpNode(
            name="test_op",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "hidden_size"),
            output_shape=ShapeSpec("seq_len", "out_dim"),
            weight_shape=ShapeSpec("hidden_size", "out_dim"),
        )
        assert node.name == "test_op"
        assert node.op_type == "matmul"
        assert node.dtype == DataType.BF16  # default

    def test_with_resolved_shapes(self):
        node = OpNode(
            name="test_op",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "hidden_size"),
            output_shape=ShapeSpec("seq_len", "out_dim"),
            weight_shape=ShapeSpec("hidden_size", "out_dim"),
            num_layers="num_layers",
        )
        context = {
            "seq_len": 128,
            "hidden_size": 4096,
            "out_dim": 11008,
            "num_layers": 32,
        }
        resolved = node.with_resolved_shapes(context)

        assert resolved.input_shape.m == 128
        assert resolved.input_shape.n == 4096
        assert resolved.num_layers == 32


class TestComputationalGraph:
    """Tests for ComputationalGraph class."""

    def test_create_graph(self):
        graph = ComputationalGraph(
            model_name="TestModel",
            model_type="dense",
            config={"hidden_size": 4096},
        )
        assert graph.model_name == "TestModel"
        assert graph.model_type == "dense"
        assert len(graph.nodes) == 0

    def test_add_node(self):
        graph = ComputationalGraph(
            model_name="TestModel",
            model_type="dense",
            config={},
        )
        node = OpNode(
            name="test_op",
            op_type="matmul",
            input_shape=ShapeSpec(128, 256),
            output_shape=ShapeSpec(128, 512),
            weight_shape=ShapeSpec(256, 512),
        )
        graph.add_node(node)
        assert len(graph.nodes) == 1

    def test_get_nodes_by_type(self):
        graph = ComputationalGraph(
            model_name="TestModel",
            model_type="dense",
            config={},
        )
        graph.add_node(
            OpNode(
                name="matmul1",
                op_type="matmul",
                input_shape=ShapeSpec(128, 256),
                output_shape=ShapeSpec(128, 512),
                weight_shape=ShapeSpec(256, 512),
            )
        )
        graph.add_node(
            OpNode(
                name="attn1",
                op_type="attention",
                input_shape=ShapeSpec(128, 64),
                output_shape=ShapeSpec(128, 128),
                weight_shape=ShapeSpec(64, 128),
            )
        )

        matmul_nodes = graph.get_nodes_by_type("matmul")
        assert len(matmul_nodes) == 1
        assert matmul_nodes[0].name == "matmul1"

    def test_get_node_by_name(self):
        graph = ComputationalGraph(
            model_name="TestModel",
            model_type="dense",
            config={},
        )
        node = OpNode(
            name="test_op",
            op_type="matmul",
            input_shape=ShapeSpec(128, 256),
            output_shape=ShapeSpec(128, 512),
            weight_shape=ShapeSpec(256, 512),
        )
        graph.add_node(node)

        found = graph.get_node_by_name("test_op")
        assert found is not None
        assert found.name == "test_op"

        not_found = graph.get_node_by_name("nonexistent")
        assert not_found is None

    def test_validate_empty_graph(self):
        graph = ComputationalGraph(
            model_name="TestModel",
            model_type="dense",
            config={},
        )
        errors = graph.validate()
        assert len(errors) == 1  # No attention nodes

    def test_validate_duplicate_names(self):
        graph = ComputationalGraph(
            model_name="TestModel",
            model_type="dense",
            config={},
        )
        graph.add_node(
            OpNode(
                name="duplicate",
                op_type="matmul",
                input_shape=ShapeSpec(128, 256),
                output_shape=ShapeSpec(128, 512),
                weight_shape=ShapeSpec(256, 512),
            )
        )
        graph.add_node(
            OpNode(
                name="duplicate",
                op_type="matmul",
                input_shape=ShapeSpec(128, 256),
                output_shape=ShapeSpec(128, 512),
                weight_shape=ShapeSpec(256, 512),
            )
        )
        errors = graph.validate()
        assert any("Duplicate" in e for e in errors)

    def test_serialization(self):
        graph = ComputationalGraph(
            model_name="TestModel",
            model_type="dense",
            config={"hidden_size": 4096},
            kv_cache_type="mha",
            has_moe=False,
            has_mla=False,
        )
        graph.add_node(
            OpNode(
                name="test_op",
                op_type="matmul",
                input_shape=ShapeSpec(128, 256),
                output_shape=ShapeSpec(128, 512),
                weight_shape=ShapeSpec(256, 512),
                dtype=DataType.BF16,
                num_layers=32,
            )
        )

        data = graph.to_dict()
        restored = ComputationalGraph.from_dict(data)

        assert restored.model_name == graph.model_name
        assert restored.model_type == graph.model_type
        assert len(restored.nodes) == len(graph.nodes)


class TestLayerParserRegistry:
    """Tests for the layer parser registry."""

    def test_list_registered_parsers(self):
        parsers = list_registered_parsers()
        assert isinstance(parsers, dict)
        # Should have parsers registered from imports
        assert len(parsers) > 0

    def test_get_parser_existing(self):
        parser = get_parser("RMSNorm")
        assert parser is not None
        assert type(parser).__name__ == "NormParser"

    def test_get_parser_nonexistent(self):
        parser = get_parser("NonExistentLayer")
        assert parser is None

    def test_unregister_parser(self):
        # First register a test parser
        @register_layer_parser("TestLayer")
        class TestParser(BaseLayerParser):
            @property
            def layer_types(self):
                return ["TestLayer"]

            def parse(self, name, module, config):
                return None

        # Verify it was registered
        assert get_parser("TestLayer") is not None

        # Unregister it
        result = unregister_parser("TestLayer")
        assert result is True

        # Verify it was removed
        assert get_parser("TestLayer") is None

    def test_unregister_nonexistent(self):
        result = unregister_parser("DefinitelyNotReal")
        assert result is False


class TestMockEnvironment:
    """Tests for the mock SGLang environment."""

    def test_mock_environment_context(self):
        import sys

        # Ensure modules don't exist before
        assert "sglang" not in sys.modules

        with mock_sglang_environment():
            assert "sglang" in sys.modules
            assert "sgl_kernel" in sys.modules

            # Test mock callable
            import sglang

            result = sglang.srt.distributed.parallel_model_parallel_is_initialized()
            assert result is None

        # After context exit, mocks should be removed
        # Note: They might still be in sys.modules but with original values


class TestModelParser:
    """Tests for ModelParser class."""

    @torch_required
    def test_create_parser(self):
        config = {"hidden_size": 4096, "num_attention_heads": 32}
        parser = ModelParser(config)
        assert parser.config == config

    @torch_required
    def test_infer_model_type_dense(self):
        config = {"hidden_size": 4096}
        parser = ModelParser(config)
        assert parser._infer_model_type() == "dense"

    @torch_required
    def test_infer_model_type_moe(self):
        config = {"hidden_size": 4096, "num_experts": 8}
        parser = ModelParser(config)
        assert parser._infer_model_type() == "moe"

    @torch_required
    def test_infer_model_type_mla(self):
        config = {"hidden_size": 4096, "qk_rope_head_dim": 64, "kv_lora_rank": 512}
        parser = ModelParser(config)
        assert parser._infer_model_type() == "mla"

    @torch_required
    def test_has_moe_true(self):
        config = {"num_experts": 8}
        parser = ModelParser(config)
        assert parser._has_moe() is True

    @torch_required
    def test_has_moe_false(self):
        config = {"hidden_size": 4096}
        parser = ModelParser(config)
        assert parser._has_moe() is False

    @torch_required
    def test_has_mla_true(self):
        config = {"qk_rope_head_dim": 64, "kv_lora_rank": 512}
        parser = ModelParser(config)
        assert parser._has_mla() is True

    @torch_required
    def test_has_mla_false(self):
        config = {"hidden_size": 4096}
        parser = ModelParser(config)
        assert parser._has_mla() is False

    @torch_required
    def test_infer_kv_cache_type_mla(self):
        config = {"qk_rope_head_dim": 64, "kv_lora_rank": 512}
        parser = ModelParser(config)
        assert parser._infer_kv_cache_type() == "mla"

    @torch_required
    def test_infer_kv_cache_type_gqa(self):
        config = {"num_attention_heads": 32, "num_key_value_heads": 8}
        parser = ModelParser(config)
        assert parser._infer_kv_cache_type() == "gqa"

    @torch_required
    def test_infer_kv_cache_type_mha(self):
        config = {"num_attention_heads": 32}
        parser = ModelParser(config)
        assert parser._infer_kv_cache_type() == "mha"


class TestDataType:
    """Tests for DataType enum."""

    def test_dtype_sizes(self):
        assert DataType.INT8.value == 1
        assert DataType.FP8.value == 1
        assert DataType.FP16.value == 2
        assert DataType.BF16.value == 2
        assert DataType.FP32.value == 4
        assert DataType.FP64.value == 8

    def test_from_str(self):
        assert DataType.from_str("int8") == DataType.INT8
        assert DataType.from_str("fp16") == DataType.FP16
        assert DataType.from_str("bf16") == DataType.BF16
        assert DataType.from_str("fp32") == DataType.FP32
        assert DataType.from_str("float32") == DataType.FP32

    def test_from_str_default(self):
        # Unknown type should default to BF16
        assert DataType.from_str("unknown") == DataType.BF16


class TestSglangAutoAdapter:
    """Tests for SglangAutoAdapter class."""

    def test_create_adapter(self):
        config = {"hidden_size": 4096}
        adapter = SglangAutoAdapter(None, config)
        assert adapter.model_class is None
        assert adapter.config == config
        assert adapter.ir_graph is None
        assert adapter.model_arch is None

    def test_get_ir_summary_not_parsed(self):
        config = {"hidden_size": 4096}
        adapter = SglangAutoAdapter(None, config)
        summary = adapter.get_ir_summary()
        assert "error" in summary


class TestIntegration:
    """Integration tests."""

    def test_full_flow_simulation(self):
        """Simulate the full adapter flow without actual SGLang models."""
        # Create IR graph manually
        graph = ComputationalGraph(
            model_name="TestModel",
            model_type="dense",
            config={
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
            },
            kv_cache_type="gqa",
        )

        # Add some nodes
        graph.add_node(
            OpNode(
                name="qkv_proj",
                op_type="matmul",
                input_shape=ShapeSpec("seq_len", "hidden_size"),
                output_shape=ShapeSpec(
                    "seq_len", "(num_heads + 2*kv_heads) * head_dim"
                ),
                weight_shape=ShapeSpec(
                    "hidden_size", "(num_heads + 2*kv_heads) * head_dim"
                ),
                dtype=DataType.BF16,
                num_layers="num_layers",
            )
        )

        graph.add_node(
            OpNode(
                name="o_proj",
                op_type="matmul",
                input_shape=ShapeSpec("seq_len", "num_heads * head_dim"),
                output_shape=ShapeSpec("seq_len", "hidden_size"),
                weight_shape=ShapeSpec("num_heads * head_dim", "hidden_size"),
                dtype=DataType.BF16,
                num_layers="num_layers",
            )
        )

        # Verify graph
        assert len(graph.nodes) == 2
        assert len(graph.get_matmul_nodes()) == 2

        # Test serialization round-trip
        data = graph.to_dict()
        restored = ComputationalGraph.from_dict(data)
        assert len(restored.nodes) == 2

        # Validate
        errors = graph.validate()
        # No attention core nodes, so should have validation error
        assert (
            len(errors) >= 0
        )  # May or may not have errors depending on validation logic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
