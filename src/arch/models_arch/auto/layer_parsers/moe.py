"""
MoE (Mixture of Experts) layer parsers.

Supports:
- FusedMoE: Fused MoE computation
- Qwen3MoeSparseMoeBlock: Qwen3 MoE block
- DeepseekV3MoE: DeepSeek V3 MoE implementation
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


@register_layer_parser("FusedMoE")
class FusedMoEParser(BaseLayerParser):
    """Parser for FusedMoE (fused expert computation)."""

    @property
    def layer_types(self) -> List[str]:
        return ["FusedMoE"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        _ = config.hidden_size  # For potential future use
        _ = getattr(
            config,
            "moe_intermediate_size",
            getattr(config, "intermediate_size", config.hidden_size * 4),
        )
        num_experts = getattr(module, "num_experts", getattr(config, "num_experts", 1))

        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("L_per_rank", "hidden_size"),
            output_shape=ShapeSpec("L_per_rank", "2 * moe_intermediate_size"),
            weight_shape=ShapeSpec("hidden_size", "2 * moe_intermediate_size"),
            dtype=DataType.INT8,
            parallel_strategy=ParallelStrategy(ep_size=getattr(config, "ep_size", 1)),
            num_layers="num_moe_layers",
            extra_attrs={
                "is_moe": True,
                "is_fused": True,
                "num_experts": num_experts,
                "has_gate": True,
            },
        )


@register_layer_parser("Qwen3MoeSparseMoeBlock")
class Qwen3MoEBlockParser(BaseLayerParser):
    """Parser for Qwen3 MoE Block (composite, returns None to process children)."""

    @property
    def layer_types(self) -> List[str]:
        return ["Qwen3MoeSparseMoeBlock"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        # This is a composite block - children will be processed separately
        # Return None to indicate we should traverse children
        return None


@register_layer_parser("DeepseekV3MoE")
class DeepSeekV3MoEParser(BaseLayerParser):
    """Parser for DeepSeek V3 MoE block (composite)."""

    @property
    def layer_types(self) -> List[str]:
        return ["DeepseekV3MoE"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        # This is a composite block - children will be processed separately
        return None


@register_layer_parser("MoEGate", "TopKGate")
class MoEGateParser(BaseLayerParser):
    """Parser for MoE routing/gate layers."""

    @property
    def layer_types(self) -> List[str]:
        return ["MoEGate", "TopKGate"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        _ = config.hidden_size  # For potential future use
        num_experts = getattr(
            config, "num_experts", getattr(config, "n_routed_experts", 1)
        )

        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "hidden_size"),
            output_shape=ShapeSpec("seq_len", num_experts),
            weight_shape=ShapeSpec("hidden_size", num_experts),
            dtype=DataType.FP32,  # Gate uses FP32
            parallel_strategy=ParallelStrategy(replicated=True),
            num_layers="num_moe_layers",
            extra_attrs={
                "is_moe": True,
                "is_gate": True,
                "num_experts": num_experts,
                "top_k": getattr(config, "num_experts_per_tok", 1),
            },
        )


@register_layer_parser("Expert")
class ExpertParser(BaseLayerParser):
    """Parser for individual expert layers."""

    @property
    def layer_types(self) -> List[str]:
        return ["Expert"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        _ = config.hidden_size  # For potential future use
        _ = getattr(
            config,
            "moe_intermediate_size",
            getattr(config, "intermediate_size", config.hidden_size * 4),
        )

        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("L_per_rank", "hidden_size"),
            output_shape=ShapeSpec("L_per_rank", "2 * moe_intermediate_size"),
            weight_shape=ShapeSpec("hidden_size", "2 * moe_intermediate_size"),
            dtype=DataType.INT8,
            parallel_strategy=ParallelStrategy(ep_size=getattr(config, "ep_size", 1)),
            num_layers="num_moe_layers",
            extra_attrs={
                "is_moe": True,
                "is_expert": True,
                "expert_id": getattr(module, "expert_id", None),
            },
        )
