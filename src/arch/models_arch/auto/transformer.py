"""
IR to ModelArch Transformer - Converts ComputationalGraph to ModelArch.

This module transforms the framework-agnostic IR representation into
LLMSim ModelArch instances with actual operators.
"""

from typing import TYPE_CHECKING, Any, Dict, List

from src.arch.config import ForwardMode, ScheduleConfig
from src.arch.model_type import AttentionType
from src.arch.models_arch.auto.ir import ComputationalGraph, DataType, OpNode, ShapeSpec
from src.arch.op.op_register import create_operator
from src.arch.op.operator_base import OperatorIO, OperatorMetadata, Tensor

if TYPE_CHECKING:
    from src.arch.models_arch.base_model_arch import BaseModelArch


class IRToModelArchTransformer:
    """
        Transforms ComputationalGraph IR into LLMSim ModelArch.

        This class handles the conversion from the intermediate representation
    to actual ModelArch instances with properly configured operators.
    """

    def __init__(self, ir_graph: ComputationalGraph, schedule_config: ScheduleConfig):
        self.ir_graph = ir_graph
        self.schedule_config = schedule_config
        self._shape_context: Dict[str, Any] = {}

    def transform(self) -> "BaseModelArch":
        """
        Transform IR graph to ModelArch.

        Returns:
            Configured ModelArch instance
        """
        from src.arch.models_arch.model_arch import create_model_arch

        # Create appropriate ModelConfig from IR
        model_config = self._create_model_config()

        # Create ModelArch instance
        arch = create_model_arch(model_config, self.schedule_config)

        # Build shape context for resolving symbolic shapes
        self._build_shape_context(model_config)

        # Build operators from IR nodes
        self._build_operators(arch)

        return arch

    def _create_model_config(self):
        """Create ModelConfig from IR graph config."""
        from src.arch.config import (
            DeepSeekV3Config,
            ModelConfig,
            Qwen3Config,
            Qwen3MoEConfig,
        )

        config_dict = self.ir_graph.config
        model_type = self.ir_graph.model_type

        # Determine specific config class based on model type and features
        if model_type == "mla" or self.ir_graph.has_mla:
            config_class = DeepSeekV3Config
        elif model_type == "moe" or self.ir_graph.has_moe:
            # Check if it's Qwen3 MoE or other MoE type
            if "qwen" in self.ir_graph.model_name.lower():
                config_class = Qwen3MoEConfig
            else:
                config_class = Qwen3MoEConfig  # Default MoE config
        elif "qwen" in self.ir_graph.model_name.lower():
            config_class = Qwen3Config
        else:
            config_class = ModelConfig

        # Create config instance
        config = config_class()

        # Copy attributes from IR config
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Store extra attributes
                setattr(config, key, value)

        # Ensure model_type is set
        config.model_type = config_dict.get("model_type", model_type)

        # Set attention type based on IR
        if self.ir_graph.kv_cache_type == "mla":
            config.attention_type = AttentionType.MLA
        elif self.ir_graph.kv_cache_type == "gqa":
            config.attention_type = AttentionType.MHA  # GQA uses MHA operator
        else:
            config.attention_type = AttentionType.MHA

        return config

    def _build_shape_context(self, model_config):
        """Build context for resolving symbolic shapes."""
        mc = model_config
        sc = self.schedule_config

        # Basic dimensions
        self._shape_context = {
            "hidden_size": mc.hidden_size,
            "num_heads": mc.num_attention_heads,
            "num_attention_heads": mc.num_attention_heads,
            "num_key_value_heads": getattr(
                mc, "num_key_value_heads", mc.num_attention_heads
            ),
            "head_dim": getattr(
                mc, "head_dim", mc.hidden_size // mc.num_attention_heads
            ),
            "intermediate_size": getattr(mc, "intermediate_size", mc.hidden_size * 4),
            "moe_intermediate_size": getattr(
                mc,
                "moe_intermediate_size",
                getattr(mc, "intermediate_size", mc.hidden_size * 4),
            ),
            "seq_len": (
                sc.max_seqlen if sc.mode == ForwardMode.EXTEND else sc.batch_size
            ),
            "max_seqlen": sc.max_seqlen,
            "batch_size": sc.batch_size,
            "tp_size": sc.tp_size,
            "ep_size": sc.ep_size,
            # TP-sharded dimensions
            "num_heads_per_rank": mc.num_attention_heads // sc.tp_size,
            "kv_heads_per_rank": max(
                1,
                getattr(mc, "num_key_value_heads", mc.num_attention_heads)
                // sc.tp_size,
            ),
            # Layer counts
            "num_layers": mc.num_hidden_layers + (1 if sc.is_mtp else 0),
            "num_moe_layers": self._get_num_moe_layers(mc, sc),
            "num_dense_layers": getattr(mc, "first_k_dense_replace", 0),
        }

        # MLA-specific dimensions
        if self.ir_graph.has_mla:
            self._shape_context.update(
                {
                    "qk_nope_head_dim": getattr(mc, "qk_nope_head_dim", 128),
                    "qk_rope_head_dim": getattr(mc, "qk_rope_head_dim", 64),
                    "v_head_dim": getattr(mc, "v_head_dim", 128),
                    "q_lora_rank": getattr(mc, "q_lora_rank", 1536),
                    "kv_lora_rank": getattr(mc, "kv_lora_rank", 512),
                }
            )

        # MoE-specific dimensions
        if self.ir_graph.has_moe:
            num_experts = getattr(mc, "num_experts", getattr(mc, "n_routed_experts", 1))
            experts_per_rank = (
                num_experts // sc.ep_size if sc.ep_size > 0 else num_experts
            )

            # Calculate L_per_rank for MoE
            L = sc.max_seqlen if sc.mode == ForwardMode.EXTEND else sc.batch_size
            num_experts_per_tok = getattr(mc, "num_experts_per_tok", 1)
            L_per_rank = (
                L // sc.tp_size * num_experts_per_tok // experts_per_rank
                if experts_per_rank > 0
                else L
            )

            self._shape_context.update(
                {
                    "num_experts": num_experts,
                    "n_routed_experts": getattr(mc, "n_routed_experts", num_experts),
                    "experts_per_rank": experts_per_rank,
                    "L": L,
                    "L_per_rank": L_per_rank,
                    "num_experts_per_tok": num_experts_per_tok,
                }
            )

    def _get_num_moe_layers(self, model_config, schedule_config) -> int:
        """Calculate number of MoE layers."""
        total_layers = model_config.num_hidden_layers + (
            1 if schedule_config.is_mtp else 0
        )
        dense_layers = getattr(model_config, "first_k_dense_replace", 0)
        return total_layers - dense_layers

    def _build_operators(self, arch: "BaseModelArch"):
        """Build operators from IR nodes and add to arch."""
        # Group nodes by category
        attention_proj_nodes = []
        attention_core_nodes = []
        ffn_nodes = []
        moe_nodes = []
        transfer_nodes = []

        for node in self.ir_graph.nodes:
            if node.op_type == "matmul":
                if node.extra_attrs.get("is_attention"):
                    attention_proj_nodes.append(node)
                elif node.extra_attrs.get("is_moe"):
                    moe_nodes.append(node)
                elif node.extra_attrs.get("is_ffn"):
                    ffn_nodes.append(node)
                else:
                    ffn_nodes.append(node)
            elif node.op_type == "attention":
                attention_core_nodes.append(node)
            elif node.op_type == "transfer":
                transfer_nodes.append(node)

        # Build attention projection operators
        for node in attention_proj_nodes:
            self._add_matmul_operator(arch, node)

        # Build TP AllReduce for attention (if TP > 1)
        if self.schedule_config.tp_size > 1:
            self._add_attention_allreduce(arch)

        # Build attention core operators
        if attention_core_nodes:
            self._build_attention_core(arch, attention_core_nodes)

        # Build FFN operators
        for node in ffn_nodes:
            self._add_matmul_operator(arch, node)

        # Build TP AllReduce for FFN (if TP > 1 and not fully DP)
        if (
            self.schedule_config.tp_size > 1
            and not self.schedule_config.enable_moe_dense_fully_dp
        ):
            self._add_ffn_allreduce(arch)

        # Build MoE operators
        for node in moe_nodes:
            self._add_matmul_operator(arch, node)

        # Build DeepEP transfer operators (if enabled)
        if self.schedule_config.deepep and self.ir_graph.has_moe:
            self._add_deepep_transfers(arch)

    def _add_matmul_operator(self, arch: "BaseModelArch", node: OpNode):
        """Add a matmul operator from IR node."""
        # Resolve num_layers
        num_layers = self._resolve_num_layers(node.num_layers)

        # Resolve shapes
        input_shape = self._resolve_shape(node.input_shape)
        output_shape = self._resolve_shape(node.output_shape)
        weight_shape = self._resolve_shape(node.weight_shape)

        # Get batch size
        batch_size = node.extra_attrs.get("batch_size", 1)
        if isinstance(batch_size, str):
            batch_size = self._shape_context.get(batch_size, 1)

        metadata = OperatorMetadata(
            name=node.name.split(".")[-1] if "." in node.name else node.name,
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(input_shape[0], input_shape[1]),
                output_shape=Tensor(output_shape[0], output_shape[1]),
                weight_shape=Tensor(weight_shape[0], weight_shape[1]),
                input_dtype=self._convert_dtype(node.dtype),
                output_dtype=self._convert_dtype(node.dtype),
                weight_dtype=self._convert_dtype(node.dtype),
            ),
            batch_size=batch_size,
            num_layers=num_layers,
        )

        op = create_operator("matmul", metadata)
        arch._add_operator(op)

    def _build_attention_core(self, arch: "BaseModelArch", nodes: List[OpNode]):
        """Build attention core operators."""
        attn_operators = []

        for node in nodes:
            num_layers = self._resolve_num_layers(node.num_layers)
            input_shape = self._resolve_shape(node.input_shape)
            output_shape = self._resolve_shape(node.output_shape)
            weight_shape = self._resolve_shape(node.weight_shape)

            # Get attention type
            attention_type = node.attention_type or "mha"
            if attention_type == "mla":
                attn_enum = AttentionType.MLA
            else:
                attn_enum = AttentionType.MHA

            # Get batch size from extra_attrs or default
            batch_size = node.extra_attrs.get(
                "num_attention_heads", self._shape_context.get("num_heads_per_rank", 1)
            )
            if isinstance(batch_size, str):
                batch_size = self._shape_context.get(batch_size, 1)

            metadata = OperatorMetadata(
                name=node.name.split(".")[-1] if "." in node.name else node.name,
                op_type="attention",
                io_config=OperatorIO(
                    input_shape=Tensor(input_shape[0], input_shape[1]),
                    output_shape=Tensor(output_shape[0], output_shape[1]),
                    weight_shape=Tensor(weight_shape[0], weight_shape[1]),
                    input_dtype=self._convert_dtype(node.dtype),
                    output_dtype=self._convert_dtype(node.dtype),
                    weight_dtype=self._convert_dtype(node.dtype),
                ),
                batch_size=batch_size,
                num_layers=num_layers,
            )

            op = create_operator("attention", metadata, attn_enum)
            attn_operators.append(op)

        if attn_operators:
            arch._add_attention_operator("attention", attn_operators)

    def _add_attention_allreduce(self, arch: "BaseModelArch"):
        """Add TP AllReduce operator for attention."""
        sc = self.schedule_config

        # Select bandwidth based on mode
        if sc.mode == ForwardMode.EXTEND:
            reduce_bandwidth = 85.0  # GB/s
        else:
            reduce_bandwidth = 22.64  # GB/s

        metadata = OperatorMetadata(
            name="attn_all_reduce",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(
                    self._shape_context["seq_len"], self._shape_context["hidden_size"]
                ),
                output_shape=Tensor(
                    self._shape_context["seq_len"], self._shape_context["hidden_size"]
                ),
                weight_shape=Tensor(0, 0),
                input_dtype=self._convert_dtype(DataType.BF16),
                output_dtype=self._convert_dtype(DataType.BF16),
            ),
            batch_size=1,
            num_layers=self._shape_context["num_layers"],
        )

        op = create_operator("transfer", metadata)
        op._bandwidth_gb_s = reduce_bandwidth
        arch._add_transfer_operator(op)

    def _add_ffn_allreduce(self, arch: "BaseModelArch"):
        """Add TP AllReduce operator for FFN."""
        sc = self.schedule_config

        if sc.mode == ForwardMode.EXTEND:
            reduce_bandwidth = 85.0  # GB/s
        else:
            reduce_bandwidth = 22.64  # GB/s

        num_layers = (
            self._shape_context.get("num_dense_layers", 0)
            or self._shape_context["num_layers"]
        )

        metadata = OperatorMetadata(
            name="dense_all_reduce",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(
                    self._shape_context["seq_len"], self._shape_context["hidden_size"]
                ),
                output_shape=Tensor(
                    self._shape_context["seq_len"], self._shape_context["hidden_size"]
                ),
                weight_shape=Tensor(0, 0),
                input_dtype=self._convert_dtype(DataType.BF16),
                output_dtype=self._convert_dtype(DataType.BF16),
            ),
            batch_size=1,
            num_layers=num_layers,
        )

        op = create_operator("transfer", metadata)
        op._bandwidth_gb_s = reduce_bandwidth
        arch._add_transfer_operator(op)

    def _add_deepep_transfers(self, arch: "BaseModelArch"):
        """Add DeepEP dispatch and combine transfer operators."""
        sc = self.schedule_config

        if sc.mode == ForwardMode.EXTEND:
            L = sc.max_seqlen // sc.tp_size
            dispatch_bandwidth = 85.0  # GB/s
            combine_bandwidth = 85.0
        else:
            L = sc.batch_size // sc.tp_size
            dispatch_bandwidth = 18.58  # GB/s
            combine_bandwidth = 22.64

        num_moe_layers = self._shape_context.get(
            "num_moe_layers", self._shape_context["num_layers"]
        )
        num_experts_per_tok = self._shape_context.get("num_experts_per_tok", 1)

        # Dispatch
        dispatch_metadata = OperatorMetadata(
            name="dispatch",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(L, self._shape_context["hidden_size"]),
                output_shape=Tensor(L, self._shape_context["hidden_size"]),
                weight_shape=Tensor(0, 0),
                input_dtype=self._convert_dtype(DataType.INT8),
            ),
            batch_size=num_experts_per_tok,
            num_layers=num_moe_layers,
        )
        dispatch_op = create_operator("transfer", dispatch_metadata)
        dispatch_op._bandwidth_gb_s = dispatch_bandwidth
        arch._add_transfer_operator(dispatch_op)

        # Combine
        combine_metadata = OperatorMetadata(
            name="combine",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(L, self._shape_context["hidden_size"]),
                output_shape=Tensor(L, self._shape_context["hidden_size"]),
                weight_shape=Tensor(0, 0),
                input_dtype=self._convert_dtype(DataType.BF16),
            ),
            batch_size=num_experts_per_tok,
            num_layers=num_moe_layers,
        )
        combine_op = create_operator("transfer", combine_metadata)
        combine_op._bandwidth_gb_s = combine_bandwidth
        arch._add_transfer_operator(combine_op)

    def _resolve_shape(self, shape_spec: ShapeSpec) -> tuple:
        """Resolve a ShapeSpec to concrete dimensions."""
        try:
            return shape_spec.resolve(self._shape_context)
        except ValueError as e:
            # If resolution fails, return defaults
            print(f"Warning: Could not resolve shape {shape_spec}: {e}")
            return (1, 1)

    def _resolve_num_layers(self, num_layers: Any) -> int:
        """Resolve num_layers which may be a string reference."""
        if isinstance(num_layers, int):
            return num_layers
        if isinstance(num_layers, str):
            return self._shape_context.get(num_layers, 1)
        return 1

    def _convert_dtype(self, dtype: DataType):
        """Convert IR DataType to operator_base DataType."""
        from src.arch.op.operator_base import DataType as OpDataType

        mapping = {
            DataType.INT8: OpDataType.INT8,
            DataType.FP8: OpDataType.FP8,
            DataType.FP16: OpDataType.FP16,
            DataType.BF16: OpDataType.BF16,
            DataType.FP32: OpDataType.FP32,
            DataType.FP64: OpDataType.FP64,
        }
        return mapping.get(dtype, OpDataType.BF16)
