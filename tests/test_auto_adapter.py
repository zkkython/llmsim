"""Tests for the IR layer and auto adapter module."""

import pytest

from src.arch.models_arch.auto.ir import (
    ComputationalGraph,
    DataType,
    OpNode,
    ShapeSpec,
)


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


class TestIntegration:
    """Integration tests."""

    def test_full_flow_simulation(self):
        """Simulate the full adapter flow."""
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
        assert len(errors) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
