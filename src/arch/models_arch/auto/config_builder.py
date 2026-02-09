"""
Config-driven IR Builder - Build ComputationalGraph directly from HF Config.

This module provides a registry-based system for building IR from HuggingFace
config objects without instantiating SGLang models.

Usage:
    # Register a new model type builder
    @register_config_builder("qwen3_moe")
    def build_qwen3_moe(config) -> ComputationalGraph:
        ...

    # Use the builder
    graph = build_ir_from_config(config, "qwen3_moe")
"""

from typing import Any, Callable, Dict, Optional, TypeVar

from src.arch.models_arch.auto.ir import (
    ComputationalGraph,
    DataType,
    OpNode,
    ParallelStrategy,
    ShapeSpec,
)

# Registry of config builders: model_type -> builder function
_CONFIG_BUILDERS: Dict[str, Callable[[Any], ComputationalGraph]] = {}

T = TypeVar("T")


def register_config_builder(model_type: str):
    """
    Decorator to register a config builder for a model type.

    Args:
        model_type: The model type identifier (e.g., "qwen3_moe", "deepseek_v3")

    Example:
        @register_config_builder("qwen3_moe")
        def build_qwen3_moe(config) -> ComputationalGraph:
            graph = ComputationalGraph(...)
            # Build graph from config
            return graph
    """

    def decorator(
        builder_fn: Callable[[Any], ComputationalGraph]
    ) -> Callable[[Any], ComputationalGraph]:
        _CONFIG_BUILDERS[model_type] = builder_fn
        return builder_fn

    return decorator


def get_config_builder(
    model_type: str,
) -> Optional[Callable[[Any], ComputationalGraph]]:
    """
    Get the registered builder for a model type.

    Args:
        model_type: The model type identifier

    Returns:
        Builder function or None if not registered
    """
    return _CONFIG_BUILDERS.get(model_type)


def list_registered_builders() -> Dict[str, str]:
    """
    List all registered config builders.

    Returns:
        Dictionary mapping model_type to builder function name
    """
    return {k: v.__name__ for k, v in _CONFIG_BUILDERS.items()}


def infer_model_type(config: Any) -> Optional[str]:
    """
    Infer model type from HuggingFace config.

    Args:
        config: HuggingFace config object or dict

    Returns:
        Model type string or None if cannot infer
    """
    # Get model_type from config
    if isinstance(config, dict):
        model_type = config.get("model_type", "")
    else:
        model_type = getattr(config, "model_type", "")

    # Check for registered builders first
    if model_type in _CONFIG_BUILDERS:
        return model_type

    # Try to infer from architecture
    architectures = []
    if isinstance(config, dict):
        architectures = config.get("architectures", [])
    else:
        architectures = getattr(config, "architectures", [])

    if architectures:
        arch = architectures[0].lower()
        # Map architecture names to model types
        if "qwen3moe" in arch or "qwen3_moe" in arch:
            return "qwen3_moe"
        elif "deepseekv3" in arch:
            return "deepseek_v3"
        elif "qwen3" in arch:
            return "qwen3"
        elif "llama" in arch:
            return "llama"

    return model_type if model_type in _CONFIG_BUILDERS else None


def build_ir_from_config(
    config: Any, model_type: Optional[str] = None
) -> ComputationalGraph:
    """
    Build ComputationalGraph from HuggingFace config.

    Args:
        config: HuggingFace config object or dict
        model_type: Optional model type override. If not provided, will be inferred.

    Returns:
        ComputationalGraph IR

    Raises:
        ValueError: If model_type is not provided and cannot be inferred,
                   or if no builder is registered for the model type.
    """
    # Infer model type if not provided
    if model_type is None:
        model_type = infer_model_type(config)

    if model_type is None:
        raise ValueError(
            "Cannot infer model_type from config. "
            "Please provide model_type explicitly. "
            f"Config has model_type={getattr(config, 'model_type', 'N/A')}, "
            f"architectures={getattr(config, 'architectures', 'N/A')}"
        )

    # Get the builder
    builder = get_config_builder(model_type)
    if builder is None:
        registered = list_registered_builders()
        raise ValueError(
            f"No config builder registered for model_type '{model_type}'. "
            f"Registered builders: {list(registered.keys())}"
        )

    # Build the IR
    return builder(config)


# =============================================================================
# Helper functions for building IR
# =============================================================================


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Get config value supporting both dict and object access."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _has_mla(config: Any) -> bool:
    """Check if config indicates MLA attention."""
    qk_rope_head_dim = _get_config_value(config, "qk_rope_head_dim", None)
    kv_lora_rank = _get_config_value(config, "kv_lora_rank", None)
    return isinstance(qk_rope_head_dim, int) and isinstance(kv_lora_rank, int)


def _has_moe(config: Any) -> bool:
    """Check if config indicates MoE architecture."""
    num_experts = _get_config_value(config, "num_experts", None)
    n_routed_experts = _get_config_value(config, "n_routed_experts", None)
    return (isinstance(num_experts, int) and num_experts > 0) or (
        isinstance(n_routed_experts, int) and n_routed_experts > 0
    )


def _infer_kv_cache_type(config: Any) -> str:
    """Infer KV cache type from config."""
    if _has_mla(config):
        return "mla"

    kv_heads = _get_config_value(config, "num_key_value_heads", 0)
    num_heads = _get_config_value(config, "num_attention_heads", 1)

    try:
        if kv_heads > 0 and kv_heads != num_heads:
            return "gqa"
    except TypeError:
        pass

    return "mha"


def _create_base_graph(
    config: Any, model_name: str, model_type: str
) -> ComputationalGraph:
    """Create base ComputationalGraph with common metadata."""
    config_dict = _config_to_dict(config)

    return ComputationalGraph(
        model_name=model_name,
        model_type=model_type,
        config=config_dict,
        has_moe=_has_moe(config),
        has_mla=_has_mla(config),
        has_dense_layers=_get_config_value(config, "first_k_dense_replace", 0) > 0,
        first_k_dense_replace=_get_config_value(config, "first_k_dense_replace", 0),
        kv_cache_type=_infer_kv_cache_type(config),
    )


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Convert config to dictionary."""
    if isinstance(config, dict):
        return config
    if hasattr(config, "to_dict"):
        return config.to_dict()

    # Extract attributes
    result = {}
    for key in dir(config):
        if not key.startswith("_"):
            try:
                value = getattr(config, key)
                if not callable(value):
                    result[key] = value
            except Exception:
                pass
    return result


# =============================================================================
# Qwen3 MoE Builder
# =============================================================================


@register_config_builder("qwen3_moe")
def build_qwen3_moe_ir(config: Any) -> ComputationalGraph:
    """
    Build IR for Qwen3 MoE models.

    Supports Qwen3-30B-A3B, Qwen3-235B-A22B, etc.
    """
    # Get config values
    hidden_size = _get_config_value(config, "hidden_size")
    num_hidden_layers = _get_config_value(config, "num_hidden_layers")
    num_attention_heads = _get_config_value(config, "num_attention_heads")
    num_key_value_heads = _get_config_value(
        config, "num_key_value_heads", num_attention_heads
    )
    intermediate_size = _get_config_value(config, "intermediate_size")
    head_dim = _get_config_value(config, "head_dim", hidden_size // num_attention_heads)

    # MoE specific
    num_experts = _get_config_value(config, "num_experts")
    num_experts_per_tok = _get_config_value(config, "num_experts_per_tok", 4)
    moe_intermediate_size = _get_config_value(
        config, "moe_intermediate_size", intermediate_size
    )

    # Create graph
    graph = _create_base_graph(config, "Qwen3MoeForCausalLM", "moe")

    # QKV Projection (Column Parallel)
    # Input: [seq_len, hidden_size]
    # Weight: [hidden_size, (num_heads + 2*kv_heads) * head_dim]
    # Output: [seq_len, (num_heads + 2*kv_heads) * head_dim]
    qkv_output_dim = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    graph.add_node(
        OpNode(
            name="qkv_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", qkv_output_dim),
            weight_shape=ShapeSpec(hidden_size, qkv_output_dim),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),  # Column parallel
            num_layers=num_hidden_layers,
            extra_attrs={"is_attention": True, "is_qkv": True},
        )
    )

    # Attention Core
    graph.add_node(
        OpNode(
            name="attention",
            op_type="attention",
            input_shape=ShapeSpec("seq_len", qkv_output_dim),
            output_shape=ShapeSpec("seq_len", num_attention_heads * head_dim),
            weight_shape=ShapeSpec(0, 0),  # No weights for attention core
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(),  # No TP for attention core
            num_layers=num_hidden_layers,
            attention_type=(
                "gqa" if num_key_value_heads != num_attention_heads else "mha"
            ),
            extra_attrs={
                "num_heads": num_attention_heads,
                "kv_heads": num_key_value_heads,
                "head_dim": head_dim,
            },
        )
    )

    # O Projection (Row Parallel)
    # Input: [seq_len, num_heads * head_dim]
    # Weight: [num_heads * head_dim, hidden_size]
    # Output: [seq_len, hidden_size]
    graph.add_node(
        OpNode(
            name="o_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", num_attention_heads * head_dim),
            output_shape=ShapeSpec("seq_len", hidden_size),
            weight_shape=ShapeSpec(num_attention_heads * head_dim, hidden_size),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=0),  # Row parallel
            num_layers=num_hidden_layers,
            extra_attrs={"is_attention": True, "is_o_proj": True},
        )
    )

    # MoE Gate (Replicated)
    graph.add_node(
        OpNode(
            name="moe_gate",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", num_experts),
            weight_shape=ShapeSpec(hidden_size, num_experts),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(replicated=True),
            num_layers=num_hidden_layers,
            extra_attrs={"is_moe": True, "is_gate": True, "top_k": num_experts_per_tok},
        )
    )

    # MoE Expert FFN (Column + Row Parallel)
    # For each expert: up_proj (gate_up) + down_proj
    # We model the aggregated experts
    graph.add_node(
        OpNode(
            name="moe_up_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", moe_intermediate_size * 2),  # gate + up
            weight_shape=ShapeSpec(hidden_size, moe_intermediate_size * 2),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),  # Column parallel
            num_layers=num_hidden_layers,
            extra_attrs={
                "is_moe": True,
                "is_expert": True,
                "num_experts": num_experts,
                "top_k": num_experts_per_tok,
            },
        )
    )

    graph.add_node(
        OpNode(
            name="moe_down_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", moe_intermediate_size),
            output_shape=ShapeSpec("seq_len", hidden_size),
            weight_shape=ShapeSpec(moe_intermediate_size, hidden_size),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=0),  # Row parallel
            num_layers=num_hidden_layers,
            extra_attrs={
                "is_moe": True,
                "is_expert": True,
                "num_experts": num_experts,
                "top_k": num_experts_per_tok,
            },
        )
    )

    # Shared Expert (if present)
    shared_expert_intermediate_size = _get_config_value(
        config, "shared_expert_intermediate_size", 0
    )
    if shared_expert_intermediate_size > 0:
        graph.add_node(
            OpNode(
                name="shared_expert_up_proj",
                op_type="matmul",
                input_shape=ShapeSpec("seq_len", hidden_size),
                output_shape=ShapeSpec("seq_len", shared_expert_intermediate_size * 2),
                weight_shape=ShapeSpec(
                    hidden_size, shared_expert_intermediate_size * 2
                ),
                dtype=DataType.BF16,
                parallel_strategy=ParallelStrategy(tp_dim=1),
                num_layers=num_hidden_layers,
                extra_attrs={"is_moe": True, "is_shared_expert": True},
            )
        )

        graph.add_node(
            OpNode(
                name="shared_expert_down_proj",
                op_type="matmul",
                input_shape=ShapeSpec("seq_len", shared_expert_intermediate_size),
                output_shape=ShapeSpec("seq_len", hidden_size),
                weight_shape=ShapeSpec(shared_expert_intermediate_size, hidden_size),
                dtype=DataType.BF16,
                parallel_strategy=ParallelStrategy(tp_dim=0),
                num_layers=num_hidden_layers,
                extra_attrs={"is_moe": True, "is_shared_expert": True},
            )
        )

    return graph


# =============================================================================
# DeepSeek V3 Builder
# =============================================================================


@register_config_builder("deepseek_v3")
def build_deepseek_v3_ir(config: Any) -> ComputationalGraph:
    """
    Build IR for DeepSeek V3 models.

    Uses MLA (Multi-head Latent Attention) and MoE.
    """
    # Get config values
    hidden_size = _get_config_value(config, "hidden_size")
    num_hidden_layers = _get_config_value(config, "num_hidden_layers")
    num_attention_heads = _get_config_value(config, "num_attention_heads")

    # MLA specific
    qk_rope_head_dim = _get_config_value(config, "qk_rope_head_dim", 64)
    kv_lora_rank = _get_config_value(config, "kv_lora_rank", 512)
    q_lora_rank = _get_config_value(config, "q_lora_rank", 1536)
    v_head_dim = _get_config_value(config, "v_head_dim", 128)
    qk_nope_head_dim = _get_config_value(config, "qk_nope_head_dim", 128)

    # MoE specific
    n_routed_experts = _get_config_value(config, "n_routed_experts", 256)
    n_shared_experts = _get_config_value(config, "n_shared_experts", 1)
    num_experts_per_tok = _get_config_value(config, "num_experts_per_tok", 8)
    moe_intermediate_size = _get_config_value(config, "moe_intermediate_size", 2048)

    # Create graph
    graph = _create_base_graph(config, "DeepseekV3ForCausalLM", "mla")

    # MLA Q Projection (compressed)
    graph.add_node(
        OpNode(
            name="mla_q_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", q_lora_rank),
            weight_shape=ShapeSpec(hidden_size, q_lora_rank),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),
            num_layers=num_hidden_layers,
            extra_attrs={"is_attention": True, "is_mla": True, "is_q": True},
        )
    )

    # MLA Q Up-projection (to B, C, A)
    q_b_dim = num_attention_heads * qk_nope_head_dim
    q_c_dim = num_attention_heads * qk_rope_head_dim
    graph.add_node(
        OpNode(
            name="mla_q_up_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", q_lora_rank),
            output_shape=ShapeSpec("seq_len", q_b_dim + q_c_dim),
            weight_shape=ShapeSpec(q_lora_rank, q_b_dim + q_c_dim),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),
            num_layers=num_hidden_layers,
            extra_attrs={"is_attention": True, "is_mla": True, "is_q_up": True},
        )
    )

    # MLA KV Compression
    c_kv_dim = kv_lora_rank + qk_rope_head_dim
    graph.add_node(
        OpNode(
            name="mla_kv_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", c_kv_dim),
            weight_shape=ShapeSpec(hidden_size, c_kv_dim),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),
            num_layers=num_hidden_layers,
            extra_attrs={"is_attention": True, "is_mla": True, "is_kv_compress": True},
        )
    )

    # MLA Attention Core
    graph.add_node(
        OpNode(
            name="mla_attention",
            op_type="attention",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", num_attention_heads * v_head_dim),
            weight_shape=ShapeSpec(0, 0),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(),
            num_layers=num_hidden_layers,
            attention_type="mla",
            extra_attrs={
                "num_heads": num_attention_heads,
                "kv_lora_rank": kv_lora_rank,
                "qk_rope_head_dim": qk_rope_head_dim,
                "qk_nope_head_dim": qk_nope_head_dim,
                "v_head_dim": v_head_dim,
            },
        )
    )

    # MLA Output Projection
    graph.add_node(
        OpNode(
            name="mla_o_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", num_attention_heads * v_head_dim),
            output_shape=ShapeSpec("seq_len", hidden_size),
            weight_shape=ShapeSpec(num_attention_heads * v_head_dim, hidden_size),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=0),
            num_layers=num_hidden_layers,
            extra_attrs={"is_attention": True, "is_mla": True, "is_o_proj": True},
        )
    )

    # MoE Gate
    graph.add_node(
        OpNode(
            name="moe_gate",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", n_routed_experts),
            weight_shape=ShapeSpec(hidden_size, n_routed_experts),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(replicated=True),
            num_layers=num_hidden_layers,
            extra_attrs={"is_moe": True, "is_gate": True, "top_k": num_experts_per_tok},
        )
    )

    # MoE Expert FFN
    graph.add_node(
        OpNode(
            name="moe_up_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", moe_intermediate_size * 2),
            weight_shape=ShapeSpec(hidden_size, moe_intermediate_size * 2),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),
            num_layers=num_hidden_layers,
            extra_attrs={
                "is_moe": True,
                "is_expert": True,
                "num_experts": n_routed_experts,
                "top_k": num_experts_per_tok,
            },
        )
    )

    graph.add_node(
        OpNode(
            name="moe_down_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", moe_intermediate_size),
            output_shape=ShapeSpec("seq_len", hidden_size),
            weight_shape=ShapeSpec(moe_intermediate_size, hidden_size),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=0),
            num_layers=num_hidden_layers,
            extra_attrs={
                "is_moe": True,
                "is_expert": True,
                "num_experts": n_routed_experts,
                "top_k": num_experts_per_tok,
            },
        )
    )

    # Shared Expert
    shared_expert_size = moe_intermediate_size * n_shared_experts
    if shared_expert_size > 0:
        graph.add_node(
            OpNode(
                name="shared_expert_up_proj",
                op_type="matmul",
                input_shape=ShapeSpec("seq_len", hidden_size),
                output_shape=ShapeSpec("seq_len", shared_expert_size * 2),
                weight_shape=ShapeSpec(hidden_size, shared_expert_size * 2),
                dtype=DataType.BF16,
                parallel_strategy=ParallelStrategy(tp_dim=1),
                num_layers=num_hidden_layers,
                extra_attrs={"is_moe": True, "is_shared_expert": True},
            )
        )

        graph.add_node(
            OpNode(
                name="shared_expert_down_proj",
                op_type="matmul",
                input_shape=ShapeSpec("seq_len", shared_expert_size),
                output_shape=ShapeSpec("seq_len", hidden_size),
                weight_shape=ShapeSpec(shared_expert_size, hidden_size),
                dtype=DataType.BF16,
                parallel_strategy=ParallelStrategy(tp_dim=0),
                num_layers=num_hidden_layers,
                extra_attrs={"is_moe": True, "is_shared_expert": True},
            )
        )

    return graph


# =============================================================================
# Qwen3 Dense Builder
# =============================================================================


@register_config_builder("qwen3")
def build_qwen3_ir(config: Any) -> ComputationalGraph:
    """
    Build IR for Qwen3 dense models.
    """
    # Get config values
    hidden_size = _get_config_value(config, "hidden_size")
    num_hidden_layers = _get_config_value(config, "num_hidden_layers")
    num_attention_heads = _get_config_value(config, "num_attention_heads")
    num_key_value_heads = _get_config_value(
        config, "num_key_value_heads", num_attention_heads
    )
    intermediate_size = _get_config_value(config, "intermediate_size")
    head_dim = _get_config_value(config, "head_dim", hidden_size // num_attention_heads)

    # Create graph
    graph = _create_base_graph(config, "Qwen3ForCausalLM", "dense")

    # QKV Projection
    qkv_output_dim = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    graph.add_node(
        OpNode(
            name="qkv_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", qkv_output_dim),
            weight_shape=ShapeSpec(hidden_size, qkv_output_dim),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),
            num_layers=num_hidden_layers,
            extra_attrs={"is_attention": True, "is_qkv": True},
        )
    )

    # Attention Core
    graph.add_node(
        OpNode(
            name="attention",
            op_type="attention",
            input_shape=ShapeSpec("seq_len", qkv_output_dim),
            output_shape=ShapeSpec("seq_len", num_attention_heads * head_dim),
            weight_shape=ShapeSpec(0, 0),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(),
            num_layers=num_hidden_layers,
            attention_type=(
                "gqa" if num_key_value_heads != num_attention_heads else "mha"
            ),
            extra_attrs={
                "num_heads": num_attention_heads,
                "kv_heads": num_key_value_heads,
                "head_dim": head_dim,
            },
        )
    )

    # O Projection
    graph.add_node(
        OpNode(
            name="o_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", num_attention_heads * head_dim),
            output_shape=ShapeSpec("seq_len", hidden_size),
            weight_shape=ShapeSpec(num_attention_heads * head_dim, hidden_size),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=0),
            num_layers=num_hidden_layers,
            extra_attrs={"is_attention": True, "is_o_proj": True},
        )
    )

    # FFN Gate-Up Projection
    graph.add_node(
        OpNode(
            name="gate_up_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", hidden_size),
            output_shape=ShapeSpec("seq_len", intermediate_size * 2),
            weight_shape=ShapeSpec(hidden_size, intermediate_size * 2),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),
            num_layers=num_hidden_layers,
            extra_attrs={"is_ffn": True, "is_gate_up": True},
        )
    )

    # FFN Down Projection
    graph.add_node(
        OpNode(
            name="down_proj",
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", intermediate_size),
            output_shape=ShapeSpec("seq_len", hidden_size),
            weight_shape=ShapeSpec(intermediate_size, hidden_size),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=0),
            num_layers=num_hidden_layers,
            extra_attrs={"is_ffn": True, "is_down": True},
        )
    )

    return graph
