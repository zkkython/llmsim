"""
Linear layer parsers - Parse various linear layer types.

Supports:
- QKVParallelLinear: QKV projection layers
- RowParallelLinear: Output projections (o_proj, down_proj)
- ColumnParallelLinear: Gate-up projections
- ReplicatedLinear: MoE gates and other replicated layers
"""

from typing import Any, List, Optional

from src.arch.models_arch.auto.ir import DataType, OpNode, ParallelStrategy, ShapeSpec
from src.arch.models_arch.auto.layer_parsers.base import BaseLayerParser
from src.arch.models_arch.auto.layer_parsers.registry import register_layer_parser

try:
    import torch.nn as nn

    ModuleType = nn.Module
except ImportError:
    nn = None  # type: ignore
    ModuleType = Any


@register_layer_parser("QKVParallelLinear", "QKVSepParallelLinear")
class QKVLinearParser(BaseLayerParser):
    """Parser for QKV projection layers (attention)."""

    @property
    def layer_types(self) -> List[str]:
        return ["QKVParallelLinear", "QKVSepParallelLinear"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        # Extract config values for potential future use
        _ = config.hidden_size
        _ = getattr(config, "num_attention_heads", 32)
        _ = getattr(
            config, "num_key_value_heads", getattr(config, "num_attention_heads", 32)
        )
        _ = getattr(
            config,
            "head_dim",
            config.hidden_size // getattr(config, "num_attention_heads", 32),
        )

        # Check if this is MLA-style separated QKV
        if "QKVSepParallelLinear" in type(module).__name__:
            return self._parse_mla_qkv(name, module, config)

        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "hidden_size"),
            output_shape=ShapeSpec(
                "seq_len", "(num_heads_per_rank + kv_heads_per_rank * 2) * head_dim"
            ),
            weight_shape=ShapeSpec(
                "hidden_size", "(num_heads_per_rank + kv_heads_per_rank * 2) * head_dim"
            ),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),  # Column-wise sharding
            num_layers="num_layers",
            extra_attrs={"is_attention": True, "is_qkv": True},
        )

    def _parse_mla_qkv(self, name: str, module: ModuleType, config: Any) -> OpNode:
        """Parse MLA-style separated QKV (q_a, kv_a projection)."""
        # Get output size from module if available
        if hasattr(module, "output_size_per_partition"):
            output_size = module.output_size_per_partition
        else:
            # Fallback to config-based expression
            q_lora_rank = getattr(config, "q_lora_rank", 1536)
            kv_lora_rank = getattr(config, "kv_lora_rank", 512)
            qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
            output_size = q_lora_rank + kv_lora_rank + qk_rope_head_dim

        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "hidden_size"),
            output_shape=ShapeSpec("seq_len", output_size),
            weight_shape=ShapeSpec("hidden_size", output_size),
            dtype=DataType.INT8,  # MLA uses INT8 for compression
            parallel_strategy=ParallelStrategy(tp_dim=1),
            num_layers="num_layers",
            extra_attrs={"is_attention": True, "is_mla_qkv": True},
        )


@register_layer_parser("RowParallelLinear")
class RowLinearParser(BaseLayerParser):
    """Parser for RowParallelLinear (typically o_proj, down_proj)."""

    @property
    def layer_types(self) -> List[str]:
        return ["RowParallelLinear"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        hidden_size = config.hidden_size
        num_heads = getattr(config, "num_attention_heads", 32)
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)

        # Determine if this is attention o_proj or FFN down_proj
        # Note: num_heads and head_dim are used in _parse_attention_output
        if (
            "attn" in name.lower()
            or "attention" in name.lower()
            or "o_proj" in name.lower()
        ):
            return self._parse_attention_output(
                name, module, config, hidden_size, num_heads, head_dim
            )
        else:
            return self._parse_ffn_down(name, module, config, hidden_size)

    def _parse_attention_output(
        self,
        name: str,
        module: ModuleType,
        config: Any,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
    ) -> OpNode:
        """Parse attention output projection (o_proj)."""
        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "num_heads_per_rank * head_dim"),
            output_shape=ShapeSpec("seq_len", "hidden_size"),
            weight_shape=ShapeSpec("num_heads_per_rank * head_dim", "hidden_size"),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=0),  # Row-wise sharding
            num_layers="num_layers",
            extra_attrs={"is_attention": True, "is_output_proj": True},
        )

    def _parse_ffn_down(
        self, name: str, module: ModuleType, config: Any, hidden_size: int
    ) -> OpNode:
        """Parse FFN down projection."""
        _ = getattr(
            config, "intermediate_size", hidden_size * 4
        )  # For potential future use

        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "intermediate_size"),
            output_shape=ShapeSpec("seq_len", "hidden_size"),
            weight_shape=ShapeSpec("intermediate_size", "hidden_size"),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=0),  # Row-wise sharding
            num_layers="num_layers",
            extra_attrs={"is_ffn": True, "is_down_proj": True},
        )


@register_layer_parser("ColumnParallelLinear")
class ColumnLinearParser(BaseLayerParser):
    """Parser for ColumnParallelLinear (typically gate_up_proj)."""

    @property
    def layer_types(self) -> List[str]:
        return ["ColumnParallelLinear"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        _ = config.hidden_size  # For potential future use
        _ = getattr(config, "intermediate_size", config.hidden_size * 4)

        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "hidden_size"),
            output_shape=ShapeSpec("seq_len", "2 * intermediate_size"),
            weight_shape=ShapeSpec("hidden_size", "2 * intermediate_size"),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),  # Column-wise sharding
            num_layers="num_layers",
            extra_attrs={"is_ffn": True, "is_gate_up_proj": True},
        )


@register_layer_parser("ReplicatedLinear")
class ReplicatedLinearParser(BaseLayerParser):
    """Parser for ReplicatedLinear (typically MoE gate)."""

    @property
    def layer_types(self) -> List[str]:
        return ["ReplicatedLinear"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        _ = config.hidden_size  # For potential future use

        # Check if this is MoE gate
        if "gate" in name.lower():
            num_experts = getattr(
                config, "num_experts", getattr(config, "n_routed_experts", 1)
            )
            return OpNode(
                name=name,
                op_type="matmul",
                input_shape=ShapeSpec("seq_len", "hidden_size"),
                output_shape=ShapeSpec("seq_len", num_experts),
                weight_shape=ShapeSpec("hidden_size", num_experts),
                dtype=DataType.FP32,  # Gate typically uses FP32
                parallel_strategy=ParallelStrategy(replicated=True),
                num_layers="num_layers",
                extra_attrs={
                    "is_moe": True,
                    "is_gate": True,
                    "num_experts": num_experts,
                },
            )

        # Generic replicated linear
        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "hidden_size"),
            output_shape=ShapeSpec("seq_len", "hidden_size"),
            weight_shape=ShapeSpec("hidden_size", "hidden_size"),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(replicated=True),
            num_layers="num_layers",
        )


@register_layer_parser("MergedColumnParallelLinear")
class MergedColumnLinearParser(BaseLayerParser):
    """Parser for MergedColumnParallelLinear (combined gate and up projections)."""

    @property
    def layer_types(self) -> List[str]:
        return ["MergedColumnParallelLinear"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        _ = config.hidden_size  # For potential future use
        _ = getattr(config, "intermediate_size", config.hidden_size * 4)

        # MergedColumnParallelLinear outputs 2 * intermediate_size for gate + up
        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "hidden_size"),
            output_shape=ShapeSpec("seq_len", "2 * intermediate_size"),
            weight_shape=ShapeSpec("hidden_size", "2 * intermediate_size"),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),
            num_layers="num_layers",
            extra_attrs={"is_ffn": True, "is_merged_gate_up": True},
        )
