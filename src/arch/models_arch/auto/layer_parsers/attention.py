"""
Attention layer parsers - Parse attention-related layers.

Supports:
- RadixAttention: Standard MHA/GQA attention
- MLAAttention: Multi-head Latent Attention (DeepSeek-style)
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


@register_layer_parser("RadixAttention")
class RadixAttentionParser(BaseLayerParser):
    """Parser for RadixAttention (standard MHA/GQA attention)."""

    @property
    def layer_types(self) -> List[str]:
        return ["RadixAttention"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        _ = getattr(  # head_dim value for potential future use
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        return OpNode(
            name=name,
            op_type="attention",
            input_shape=ShapeSpec("seq_len", "head_dim"),
            output_shape=ShapeSpec("seq_len", "max_seqlen"),
            weight_shape=ShapeSpec("head_dim", "max_seqlen"),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(),  # Attention is parallelized across heads
            num_layers="num_layers",
            attention_type="mha",  # Will be refined based on config
            extra_attrs={"is_attention_core": True},
        )


@register_layer_parser("MLAAttention")
class MLAAttentionParser(BaseLayerParser):
    """Parser for MLAAttention (Multi-head Latent Attention - DeepSeek)."""

    @property
    def layer_types(self) -> List[str]:
        return ["MLAAttention"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128)

        return OpNode(
            name=name,
            op_type="attention",
            input_shape=ShapeSpec("seq_len", qk_nope_head_dim),
            output_shape=ShapeSpec("seq_len", "max_seqlen"),
            weight_shape=ShapeSpec(qk_nope_head_dim, "max_seqlen"),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(),
            num_layers="num_layers",
            attention_type="mla",
            extra_attrs={
                "is_attention_core": True,
                "is_mla": True,
                "qk_nope_head_dim": qk_nope_head_dim,
                "qk_rope_head_dim": getattr(config, "qk_rope_head_dim", 64),
                "kv_lora_rank": getattr(config, "kv_lora_rank", 512),
            },
        )


@register_layer_parser("LlamaAttention", "Qwen2Attention")
class StandardAttentionParser(BaseLayerParser):
    """Parser for standard attention implementations (Llama, Qwen2, etc.)."""

    @property
    def layer_types(self) -> List[str]:
        return ["LlamaAttention", "Qwen2Attention", "Qwen3Attention"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        # Determine attention type based on config
        num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        if num_kv_heads == config.num_attention_heads:
            attn_type = "mha"
        else:
            attn_type = "gqa"

        return OpNode(
            name=name,
            op_type="attention",
            input_shape=ShapeSpec("seq_len", head_dim),
            output_shape=ShapeSpec("seq_len", "max_seqlen"),
            weight_shape=ShapeSpec(head_dim, "max_seqlen"),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(),
            num_layers="num_layers",
            attention_type=attn_type,
            extra_attrs={
                "is_attention_core": True,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": num_kv_heads,
            },
        )
