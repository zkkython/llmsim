"""
Intermediate Representation (IR) Layer - Framework-agnostic model representation.

This module defines the IR data structures used as an intermediate layer between
SGLang models and LLMSim ModelArch. The IR is:
- Framework-agnostic: Not tied to PyTorch or any specific framework
- Serializable: Can be saved/loaded for debugging and caching
- Analyzable: Supports inspection, visualization, and validation
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


class DataType(Enum):
    """Data type enumeration with size in bytes."""

    INT8 = 1
    FP8 = 1
    FP16 = 2
    BF16 = 2
    FP32 = 4
    FP64 = 8

    @classmethod
    def from_str(cls, dtype_str: str) -> "DataType":
        """Parse DataType from string representation."""
        mapping = {
            "int8": cls.INT8,
            "fp8": cls.FP8,
            "fp16": cls.FP16,
            "bf16": cls.BF16,
            "fp32": cls.FP32,
            "float32": cls.FP32,
            "fp64": cls.FP64,
        }
        return mapping.get(dtype_str.lower(), cls.BF16)


class OpType(Enum):
    """Operator type enumeration."""

    MATMUL = auto()
    ATTENTION = auto()
    TRANSFER = auto()
    FFN = auto()
    NORM = auto()
    EMBEDDING = auto()


@dataclass
class ShapeSpec:
    """
    Shape specification supporting both concrete values and symbolic expressions.

    Examples:
        ShapeSpec(128, 256)  # Concrete shape
        ShapeSpec("seq_len", "hidden_size")  # Symbolic shape
        ShapeSpec("seq_len", "num_heads * head_dim")  # Expression
    """

    m: Union[int, str] = 0
    n: Union[int, str] = 0

    @property
    def is_symbolic(self) -> bool:
        """Check if shape contains symbolic expressions."""
        return isinstance(self.m, str) or isinstance(self.n, str)

    def resolve(self, context: Dict[str, Any]) -> Tuple[int, int]:
        """
        Resolve symbolic shape to concrete values.

        Args:
            context: Dictionary mapping variable names to values

        Returns:
            Tuple of (m, n) as concrete integers
        """
        m_val = self._resolve_dim(self.m, context)
        n_val = self._resolve_dim(self.n, context)
        return (m_val, n_val)

    def _resolve_dim(self, dim: Union[int, str], context: Dict[str, Any]) -> int:
        """Resolve a single dimension."""
        if isinstance(dim, int):
            return dim

        # Evaluate expression in context
        try:
            # Handle simple variable names
            if dim in context:
                return int(context[dim])

            # Handle expressions like "num_heads * head_dim"
            # Use eval with limited context for safety
            allowed_names = {
                "max": max,
                "min": min,
            }
            return int(eval(dim, {"__builtins__": {}}, {**allowed_names, **context}))
        except (NameError, SyntaxError, TypeError) as e:
            raise ValueError(f"Cannot resolve shape dimension '{dim}': {e}")

    def to_tuple(self) -> Tuple[Union[int, str], Union[int, str]]:
        """Convert to tuple representation."""
        return (self.m, self.n)

    @staticmethod
    def from_tuple(t: Tuple[Union[int, str], Union[int, str]]) -> "ShapeSpec":
        """Create ShapeSpec from tuple."""
        return ShapeSpec(m=t[0], n=t[1])


@dataclass
class ParallelStrategy:
    """
    Parallelization strategy for an operator.

    Defines how an operator is distributed across devices via
    Tensor Parallelism (TP) and Expert Parallelism (EP).
    """

    # TP dimension: 0 = row-wise (input), 1 = column-wise (output), None = no TP
    tp_dim: Optional[int] = None

    # EP group size for MoE layers
    ep_size: int = 1

    # Whether this operator is replicated across TP ranks
    replicated: bool = False

    @property
    def is_sharded(self) -> bool:
        """Check if operator is sharded (TP or EP)."""
        return self.tp_dim is not None or self.ep_size > 1


@dataclass
class OpNode:
    """
    Computational graph node - framework-agnostic operator representation.

    Attributes:
        name: Unique identifier (e.g., "model.layers.0.self_attn.qkv_proj")
        op_type: Operator category (matmul, attention, transfer, etc.)
        input_shape: Input tensor shape specification
        output_shape: Output tensor shape specification
        weight_shape: Weight tensor shape specification (if applicable)
        dtype: Data type for computation
        parallel_strategy: TP/EP parallelization configuration
        num_layers: Number of layers this operator appears in, or reference to config
        extra_attrs: Additional operator-specific attributes
    """

    name: str
    op_type: str  # "matmul", "attention", "transfer", "ffn", "norm"
    input_shape: ShapeSpec
    output_shape: ShapeSpec
    weight_shape: ShapeSpec
    dtype: DataType = DataType.BF16
    parallel_strategy: ParallelStrategy = field(default_factory=ParallelStrategy)
    num_layers: Union[int, str] = 1
    extra_attrs: Dict[str, Any] = field(default_factory=dict)

    # For attention operators
    attention_type: Optional[str] = None  # "mha", "gqa", "mla"

    def get_flops(self, context: Dict[str, Any]) -> int:
        """
        Calculate FLOPs for this operator.

        Args:
            context: Shape resolution context

        Returns:
            Estimated FLOPs count
        """
        in_shape = self.input_shape.resolve(context)

        if self.op_type == "matmul":
            # Matmul FLOPs: 2 * M * N * K (where K is reduction dim)
            # Assuming weight_shape gives us the K dimension
            weight_shape = self.weight_shape.resolve(context)
            return 2 * in_shape[0] * in_shape[1] * weight_shape[1]
        elif self.op_type == "attention":
            # Attention FLOPs depend on specific attention type
            seq_len = in_shape[0]
            head_dim = in_shape[1]
            return 2 * seq_len * seq_len * head_dim

        return 0

    def with_resolved_shapes(self, context: Dict[str, Any]) -> "OpNode":
        """
        Create a new OpNode with resolved concrete shapes.

        Args:
            context: Shape resolution context

        Returns:
            New OpNode with concrete shapes
        """
        return OpNode(
            name=self.name,
            op_type=self.op_type,
            input_shape=ShapeSpec.from_tuple(self.input_shape.resolve(context)),
            output_shape=ShapeSpec.from_tuple(self.output_shape.resolve(context)),
            weight_shape=ShapeSpec.from_tuple(self.weight_shape.resolve(context)),
            dtype=self.dtype,
            parallel_strategy=self.parallel_strategy,
            num_layers=(
                self.num_layers
                if isinstance(self.num_layers, int)
                else context.get(self.num_layers, 1)
            ),
            extra_attrs=self.extra_attrs.copy(),
            attention_type=self.attention_type,
        )


@dataclass
class ComputationalGraph:
    """
    Complete computational graph representing a model architecture.

    This is the top-level IR structure that contains all operators
    and metadata needed to reconstruct the model in LLMSim.
    """

    model_name: str
    model_type: str  # "dense", "moe", "mla"
    config: Dict[str, Any]  # Reference to original config
    nodes: List[OpNode] = field(default_factory=list)
    kv_cache_type: str = "mha"  # "mha", "gqa", "mla"

    # Model feature flags
    has_moe: bool = False
    has_mla: bool = False
    has_dense_layers: bool = False
    first_k_dense_replace: int = 0

    def get_nodes_by_type(self, op_type: str) -> List[OpNode]:
        """Get all nodes of a specific operator type."""
        return [n for n in self.nodes if n.op_type == op_type]

    def get_attention_nodes(self) -> List[OpNode]:
        """Get all attention-related nodes."""
        return self.get_nodes_by_type("attention")

    def get_matmul_nodes(self) -> List[OpNode]:
        """Get all matrix multiplication nodes."""
        return self.get_nodes_by_type("matmul")

    def get_transfer_nodes(self) -> List[OpNode]:
        """Get all transfer/communication nodes."""
        return self.get_nodes_by_type("transfer")

    def get_node_by_name(self, name: str) -> Optional[OpNode]:
        """Find a node by its name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def add_node(self, node: OpNode) -> None:
        """Add a node to the computational graph."""
        self.nodes.append(node)

    def validate(self) -> List[str]:
        """
        Validate the computational graph.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for duplicate names
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            duplicates = set(n for n in names if names.count(n) > 1)
            errors.append(f"Duplicate node names: {duplicates}")

        # Check for required nodes
        if not self.get_attention_nodes() and self.model_type != "embedding_only":
            errors.append("No attention nodes found")

        # Validate individual nodes
        for node in self.nodes:
            if not node.name:
                errors.append(f"Node with empty name (op_type={node.op_type})")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "config": self.config,
            "kv_cache_type": self.kv_cache_type,
            "has_moe": self.has_moe,
            "has_mla": self.has_mla,
            "has_dense_layers": self.has_dense_layers,
            "first_k_dense_replace": self.first_k_dense_replace,
            "nodes": [
                {
                    "name": n.name,
                    "op_type": n.op_type,
                    "input_shape": n.input_shape.to_tuple(),
                    "output_shape": n.output_shape.to_tuple(),
                    "weight_shape": n.weight_shape.to_tuple(),
                    "dtype": n.dtype.name,
                    "parallel_strategy": {
                        "tp_dim": n.parallel_strategy.tp_dim,
                        "ep_size": n.parallel_strategy.ep_size,
                        "replicated": n.parallel_strategy.replicated,
                    },
                    "num_layers": n.num_layers,
                    "extra_attrs": n.extra_attrs,
                    "attention_type": n.attention_type,
                }
                for n in self.nodes
            ],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ComputationalGraph":
        """Create ComputationalGraph from dictionary."""
        graph = ComputationalGraph(
            model_name=data["model_name"],
            model_type=data["model_type"],
            config=data["config"],
            kv_cache_type=data.get("kv_cache_type", "mha"),
            has_moe=data.get("has_moe", False),
            has_mla=data.get("has_mla", False),
            has_dense_layers=data.get("has_dense_layers", False),
            first_k_dense_replace=data.get("first_k_dense_replace", 0),
        )

        for node_data in data.get("nodes", []):
            node = OpNode(
                name=node_data["name"],
                op_type=node_data["op_type"],
                input_shape=ShapeSpec.from_tuple(tuple(node_data["input_shape"])),  # type: ignore
                output_shape=ShapeSpec.from_tuple(tuple(node_data["output_shape"])),  # type: ignore
                weight_shape=ShapeSpec.from_tuple(tuple(node_data["weight_shape"])),  # type: ignore
                dtype=DataType.from_str(node_data["dtype"]),
                parallel_strategy=ParallelStrategy(
                    tp_dim=node_data["parallel_strategy"]["tp_dim"],
                    ep_size=node_data["parallel_strategy"]["ep_size"],
                    replicated=node_data["parallel_strategy"]["replicated"],
                ),
                num_layers=node_data["num_layers"],
                extra_attrs=node_data.get("extra_attrs", {}),
                attention_type=node_data.get("attention_type"),
            )
            graph.add_node(node)

        return graph
