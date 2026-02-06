"""
SGLang Model Architecture Auto-Generator

Automatically generates model_arch.py from SGLang model implementations.
Uses runtime dynamic analysis to extract layer structure and dimensions.

Key Features:
1. Mock environment to bypass SGLang distributed dependencies
2. Automatic layer type recognition and mapping
3. Dynamic shape extraction with expression-based dimensions
4. Comprehensive error reporting for unsupported operators

Usage:
    from src.arch.models_arch.auto_generator import generate_model_arch
    code = generate_model_arch(ModelClass, config, 'output.py')
"""

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
    ModuleType = nn.Module
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False
    ModuleType = Any


# =============================================================================
# Exception Classes for Explicit Error Reporting
# =============================================================================


class AutoGeneratorError(Exception):
    """Base exception for auto-generator errors."""

    pass


class UnsupportedLayerError(AutoGeneratorError):
    """Raised when encountering an unsupported layer type.

    Attributes:
        layer_name: Full path of the layer in model hierarchy
        layer_type: The unsupported layer class name
        module_file: Source file where the layer is defined
        suggestion: Hint on how to add support for this layer
    """

    def __init__(
        self,
        layer_name: str,
        layer_type: str,
        module_file: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.module_file = module_file
        self.suggestion = suggestion or (
            f"Add '{layer_type}' to LayerTypeRecognizer.LAYER_PATTERNS "
            f"with appropriate shape_extractor method"
        )

        message = (
            f"\n{'='*70}\n"
            f"UNSUPPORTED LAYER DETECTED\n"
            f"{'='*70}\n"
            f"Layer Name:    {layer_name}\n"
            f"Layer Type:    {layer_type}\n"
            f"Source File:   {module_file or 'Unknown'}\n"
            f"\n"
            f"To fix this issue:\n"
            f"  1. Add the following entry to LAYER_PATTERNS in auto_generator.py:\n"
            f"\n"
            f"     '{layer_type}': {{\n"
            f"         'category': 'matmul',  # or 'attention', 'transfer', etc.\n"
            f"         'op_name': '<operator_name>',\n"
            f"         'shape_extractor': '_extract_<layer_type>_shape',\n"
            f"     }}\n"
            f"\n"
            f"  2. Implement the shape extractor method:\n"
            f"\n"
            f"     def _extract_{layer_type.lower()}_shape(\n"
            f"         self, name: str, module: Any\n"
            f"     ) -> Dict:\n"
            f"         return {{\n"
            f"             'input_shape': ('seq_len', 'mc.hidden_size'),\n"
            f"             'output_shape': ('seq_len', 'mc.hidden_size'),\n"
            f"             'weight_shape': ('mc.hidden_size', 'mc.hidden_size'),\n"
            f"             'dtype': 'BF16',\n"
            f"             'batch_size': 1,\n"
            f"         }}\n"
            f"{'='*70}"
        )
        super().__init__(message)


class MissingDependencyError(AutoGeneratorError):
    """Raised when required dependencies are not installed."""

    pass


class ModelInstantiationError(AutoGeneratorError):
    """Raised when model instantiation fails."""

    pass


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class LayerInfo:
    """Container for layer information extracted from model."""

    name: str
    layer_type: str  # Original PyTorch layer type name
    op_category: str  # Mapped operator category: matmul/attention/transfer/ffn
    op_name: str  # Operator name, e.g., qkv_proj, moe_gate
    input_shape: Tuple[
        Union[int, str], ...
    ]  # Supports string expressions like "seq_len"
    output_shape: Tuple[Union[int, str], ...]
    weight_shape: Tuple[Union[int, str], ...]
    dtype: str = "BF16"
    batch_size: Union[int, str] = 1
    num_layers: Union[int, str] = "num_layers"
    attention_type: Optional[str] = None  # Required only for attention operators
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelStructure:
    """Complete model structure information."""

    model_name: str
    config_class: str
    base_config_class: str = "ModelConfig"

    # Operator collections by category
    matmul_ops: List[LayerInfo] = field(default_factory=list)
    attention_ops: Dict[str, List[LayerInfo]] = field(default_factory=dict)
    transfer_ops: List[LayerInfo] = field(default_factory=list)
    ffn_ops: List[LayerInfo] = field(default_factory=list)

    # Model feature flags
    has_moe: bool = False
    has_mla: bool = False
    has_dense_layers: bool = False
    first_k_dense_replace: int = 0

    # KV Cache configuration
    kv_cache_type: str = "mha_gqa"  # Alternative: "mla"
    kv_cache_dtype: str = "BF16"

    # Tracking for error reporting
    unsupported_layers: List[Dict[str, str]] = field(default_factory=list)


# =============================================================================
# Mock Environment for Bypassing SGLang Dependencies
# =============================================================================


class MockModule:
    """Base class for mock modules."""

    def __init__(self, name: str):
        self.__name__ = name

    def __getattr__(self, name: str):
        # Return a new MockCallable for any attribute access
        return MockCallable(f"{self.__name__}.{name}")


class MockCallable:
    """Mock callable object that accepts any arguments."""

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name: str):
        return MockCallable(f"{self._name}.{name}")


def create_mock_environment() -> Dict[str, Any]:
    """
    Create a complete mock environment to bypass SGLang distributed dependencies.

    Mocked modules include:
    - sglang.srt.distributed.* (Tensor/Expert Parallelism)
    - sglang.srt.layers.communicator.* (Layer communication)
    - sglang.srt.server_args.* (Server arguments)
    - sgl_kernel.* (CUDA kernels)
    """
    mock_modules = {}

    # SGLang core modules
    mock_modules["sglang"] = MockModule("sglang")
    mock_modules["sglang.srt"] = MockModule("sglang.srt")
    mock_modules["sglang.srt.distributed"] = MockModule("sglang.srt.distributed")
    mock_modules["sglang.srt.layers"] = MockModule("sglang.srt.layers")
    mock_modules["sglang.srt.layers.communicator"] = MockModule(
        "sglang.srt.layers.communicator"
    )
    mock_modules["sglang.srt.model_executor"] = MockModule("sglang.srt.model_executor")
    mock_modules["sglang.srt.server_args"] = MockModule("sglang.srt.server_args")
    mock_modules["sglang.srt.eplb"] = MockModule("sglang.srt.eplb")

    # CUDA kernels
    mock_modules["sgl_kernel"] = MockModule("sgl_kernel")
    mock_modules["sgl_kernel_npu"] = MockModule("sgl_kernel_npu")

    # Optional dependencies
    mock_modules["vllm"] = MockModule("vllm")
    mock_modules["vllm.distributed"] = MockModule("vllm.distributed")

    return mock_modules


@contextmanager
def mock_sglang_environment():
    """
    Context manager to temporarily replace SGLang dependencies with mock objects.

    Usage:
        with mock_sglang_environment():
            from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
            model = Qwen3MoeForCausalLM(config)
    """
    mock_modules = create_mock_environment()
    original_modules = {}

    # Save original modules and replace with mocks
    for name, mock in mock_modules.items():
        if name in sys.modules:
            original_modules[name] = sys.modules[name]
        sys.modules[name] = mock

    try:
        yield
    finally:
        # Restore original modules
        for name in mock_modules:
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            elif name in sys.modules:
                del sys.modules[name]


# =============================================================================
# Layer Type Recognition and Mapping
# =============================================================================


class LayerTypeRecognizer:
    """
    Recognizes SGLang/PyTorch layer types and maps them to operator categories.

    Supported layer types:
    - Linear variants: QKVParallelLinear, RowParallelLinear, ColumnParallelLinear, ReplicatedLinear
    - MoE related: FusedMoE, Qwen3MoeSparseMoeBlock, DeepseekV3MoE
    - Attention: RadixAttention, MLAAttention
    - Normalization: RMSNorm (skipped, no operator generated)
    """

    # Mapping from layer type to operator category and extractor
    LAYER_PATTERNS = {
        # Attention projection layers
        "QKVParallelLinear": {
            "category": "matmul",
            "op_name": "qkv_proj",
            "shape_extractor": "_extract_qkv_shape",
        },
        "QKVSepParallelLinear": {
            "category": "matmul",
            "op_name": "qkv_proj",
            "shape_extractor": "_extract_qkv_sep_shape",
        },
        "RowParallelLinear": {
            "category": "matmul",
            "op_name": None,  # Determined dynamically from path
            "shape_extractor": "_extract_row_linear_shape",
        },
        "ColumnParallelLinear": {
            "category": "matmul",
            "op_name": "gate_up_proj",
            "shape_extractor": "_extract_column_linear_shape",
        },
        "ReplicatedLinear": {
            "category": "matmul",
            "op_name": None,  # Determined dynamically from path
            "shape_extractor": "_extract_replicated_linear_shape",
        },
        # MoE related
        "FusedMoE": {
            "category": "moe",
            "op_name": "moe_experts",
            "shape_extractor": "_extract_fused_moe_shape",
        },
        "Qwen3MoeSparseMoeBlock": {
            "category": "moe_block",
            "op_name": "moe_block",
            "shape_extractor": "_extract_moe_block_shape",
        },
        "DeepseekV3MoE": {
            "category": "moe_block",
            "op_name": "moe_block",
            "shape_extractor": "_extract_deepseek_moe_shape",
        },
        # Attention core
        "RadixAttention": {
            "category": "attention",
            "op_name": "attention",
            "shape_extractor": "_extract_radix_attention_shape",
        },
        "MLAAttention": {
            "category": "attention",
            "op_name": "mla_attention",
            "shape_extractor": "_extract_mla_attention_shape",
        },
        # Normalization layers (skipped)
        "RMSNorm": {
            "category": "norm",
            "op_name": None,
            "shape_extractor": None,
        },
    }

    def __init__(self, config: Any):
        self.config = config
        self._shape_cache: Dict[str, Any] = {}

    def recognize(self, name: str, module: Any) -> Optional[LayerInfo]:
        """
        Recognize layer type and extract information.

        Args:
            name: Full path of module in model hierarchy, e.g., "model.layers.0.self_attn.qkv_proj"
            module: nn.Module instance

        Returns:
            LayerInfo or None if layer should be skipped

        Raises:
            UnsupportedLayerError: If layer type is not in LAYER_PATTERNS
        """
        layer_type = type(module).__name__
        pattern = self.LAYER_PATTERNS.get(layer_type)

        if not pattern:
            # Layer type not supported - raise explicit error
            module_file = getattr(type(module), "__module__", None)
            raise UnsupportedLayerError(
                layer_name=name, layer_type=layer_type, module_file=module_file
            )

        if pattern["category"] == "norm":
            return None  # Norm layers don't generate operators

        # Determine op_name from context if not fixed
        op_name = pattern["op_name"]
        if op_name is None:
            op_name = self._infer_op_name_from_path(name, layer_type)

        # Extract shape information
        extractor_name = pattern["shape_extractor"]
        if extractor_name and hasattr(self, extractor_name):
            shape_info = getattr(self, extractor_name)(name, module)
        else:
            shape_info = self._default_shape_extraction(name, module)

        return LayerInfo(
            name=name,
            layer_type=layer_type,
            op_category=pattern["category"],
            op_name=op_name,
            **shape_info,
        )

    def _infer_op_name_from_path(self, path: str, layer_type: str) -> str:
        """Infer operator name from module path."""
        path_lower = path.lower()

        if "gate" in path_lower:
            return "moe_gate"
        elif "qkv" in path_lower or "query" in path_lower:
            return "qkv_proj"
        elif "o_proj" in path_lower or "dense" in path_lower:
            return "o_proj"
        elif "up" in path_lower or "gate_up" in path_lower:
            return "gate_up_proj"
        elif "down" in path_lower:
            return "down_proj"
        else:
            return f"{layer_type.lower()}_proj"

    # =============================================================================
    # Shape Extraction Methods
    # =============================================================================

    def _extract_qkv_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for QKVParallelLinear."""
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        kv_heads = getattr(self.config, "num_key_value_heads", num_heads)
        head_dim = getattr(self.config, "head_dim", hidden_size // num_heads)

        # Use expression strings for TP-sharded dimensions
        return {
            "input_shape": ("seq_len", "mc.hidden_size"),
            "output_shape": (
                "seq_len",
                "(num_heads_per_rank + kv_heads_per_rank * 2) * head_dim",
            ),
            "weight_shape": (
                "mc.hidden_size",
                "(num_heads_per_rank + kv_heads_per_rank * 2) * head_dim",
            ),
            "dtype": "BF16",
            "batch_size": 1,
        }

    def _extract_qkv_sep_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for separated QKV Linear (e.g., MLA's q_a, kv_a)."""
        if hasattr(module, "output_size_per_partition"):
            output_size = module.output_size_per_partition
        else:
            output_size = "mc.q_lora_rank + mc.kv_lora_rank + mc.qk_rope_head_dim"

        return {
            "input_shape": ("seq_len", "mc.hidden_size"),
            "output_shape": ("seq_len", output_size),
            "weight_shape": ("mc.hidden_size", output_size),
            "dtype": "INT8",
            "batch_size": 1,
        }

    def _extract_row_linear_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for RowParallelLinear (typically o_proj)."""
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = getattr(self.config, "head_dim", hidden_size // num_heads)

        # Determine if this is attention o_proj or FFN down_proj
        if "attn" in name.lower() or "attention" in name.lower():
            return {
                "input_shape": ("seq_len", "num_heads_per_rank * head_dim"),
                "output_shape": ("seq_len", "mc.hidden_size"),
                "weight_shape": ("num_heads_per_rank * head_dim", "mc.hidden_size"),
                "dtype": "BF16",
                "batch_size": 1,
            }
        else:
            # FFN down_proj
            return {
                "input_shape": ("seq_len", "intermediate_size"),
                "output_shape": ("seq_len", "mc.hidden_size"),
                "weight_shape": ("intermediate_size", "mc.hidden_size"),
                "dtype": "BF16",
                "batch_size": 1,
            }

    def _extract_column_linear_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for ColumnParallelLinear (typically gate_up_proj)."""
        hidden_size = self.config.hidden_size
        intermediate_size = getattr(self.config, "intermediate_size", hidden_size * 4)

        return {
            "input_shape": ("seq_len", "mc.hidden_size"),
            "output_shape": ("seq_len", "2 * intermediate_size"),
            "weight_shape": ("mc.hidden_size", "2 * intermediate_size"),
            "dtype": "BF16",
            "batch_size": 1,
        }

    def _extract_replicated_linear_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for ReplicatedLinear (typically MoE gate)."""
        hidden_size = self.config.hidden_size

        # Check if this is MoE gate
        if "gate" in name.lower():
            num_experts = getattr(
                self.config, "num_experts", getattr(self.config, "n_routed_experts", 1)
            )
            return {
                "input_shape": ("seq_len", "mc.hidden_size"),
                "output_shape": ("seq_len", num_experts),
                "weight_shape": ("mc.hidden_size", num_experts),
                "dtype": "FP32",
                "batch_size": 1,
            }
        else:
            return {
                "input_shape": ("seq_len", "mc.hidden_size"),
                "output_shape": ("seq_len", "mc.hidden_size"),
                "weight_shape": ("mc.hidden_size", "mc.hidden_size"),
                "dtype": "BF16",
                "batch_size": 1,
            }

    def _extract_fused_moe_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for FusedMoE."""
        hidden_size = self.config.hidden_size
        intermediate_size = getattr(
            self.config,
            "moe_intermediate_size",
            getattr(self.config, "intermediate_size", hidden_size * 4),
        )
        num_experts = getattr(
            module, "num_experts", getattr(self.config, "num_experts", 1)
        )

        return {
            "input_shape": ("L_per_rank", "mc.hidden_size"),
            "output_shape": ("L_per_rank", "2 * mc.moe_intermediate_size"),
            "weight_shape": ("mc.hidden_size", "2 * mc.moe_intermediate_size"),
            "dtype": "INT8",
            "batch_size": "experts_per_rank",
            "extra_config": {
                "has_gate": True,
                "num_experts": num_experts,
            },
        }

    def _extract_moe_block_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for MoE Block (composite, processed recursively)."""
        return {
            "input_shape": (0, 0),
            "output_shape": (0, 0),
            "weight_shape": (0, 0),
            "is_composite": True,
        }

    def _extract_deepseek_moe_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for DeepSeek V3 MoE block."""
        return {
            "input_shape": (0, 0),
            "output_shape": (0, 0),
            "weight_shape": (0, 0),
            "is_composite": True,
        }

    def _extract_radix_attention_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for RadixAttention (standard MHA/GQA)."""
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )

        return {
            "input_shape": ("seq_len", "head_dim"),
            "output_shape": ("seq_len", "sc.max_seqlen"),
            "weight_shape": ("head_dim", "sc.max_seqlen"),
            "dtype": "BF16",
            "batch_size": "num_heads_per_rank",
            "attention_type": "MHA",
        }

    def _extract_mla_attention_shape(self, name: str, module: Any) -> Dict:
        """Extract shape for MLA Attention."""
        return {
            "input_shape": ("seq_len", "mc.qk_nope_head_dim"),
            "output_shape": ("seq_len", "sc.max_seqlen"),
            "weight_shape": ("mc.qk_nope_head_dim", "sc.max_seqlen"),
            "dtype": "BF16",
            "batch_size": "num_heads_per_rank",
            "attention_type": "MLA",
        }

    def _default_shape_extraction(self, name: str, module: Any) -> Dict:
        """Default shape extraction based on in_features/out_features."""
        in_features = getattr(module, "in_features", getattr(module, "input_size", 0))
        out_features = getattr(
            module, "out_features", getattr(module, "output_size", 0)
        )

        return {
            "input_shape": ("seq_len", in_features or "mc.hidden_size"),
            "output_shape": ("seq_len", out_features or "mc.hidden_size"),
            "weight_shape": (
                in_features or "mc.hidden_size",
                out_features or "mc.hidden_size",
            ),
            "dtype": "BF16",
            "batch_size": 1,
        }


# =============================================================================
# Model Structure Extractor
# =============================================================================


class ModelStructureExtractor:
    """
    Traverses SGLang model structure and extracts complete architecture info.

    Processing logic:
    1. Identify model type (MoE/Dense/MLA)
    2. Traverse layers hierarchically, identifying key operators
    3. Handle special structures (e.g., DeepSeek's first_k_dense_replace)
    4. Generate Transfer operators (TP AllReduce, DeepEP dispatch/combine)
    """

    def __init__(self, config: Any, model_class: Type):
        if not HAS_TORCH:
            raise MissingDependencyError(
                "PyTorch is required for model analysis. "
                "Install with: pip install torch"
            )
        self.config = config
        self.model_class = model_class
        self.recognizer = LayerTypeRecognizer(config)

    def extract(self) -> ModelStructure:
        """Extract complete model structure information."""
        with mock_sglang_environment():
            model = self._instantiate_model()

        structure = ModelStructure(
            model_name=self.model_class.__name__,
            config_class=self._infer_config_class(),
        )

        # Identify model features
        self._identify_model_features(structure)

        # Traverse model layers
        self._traverse_model(model, structure)

        # Generate Transfer operators
        self._generate_transfer_ops(structure)

        # Report unsupported layers if any
        if structure.unsupported_layers:
            self._report_unsupported_layers(structure)

        return structure

    def _instantiate_model(self) -> Any:
        """Instantiate model within mock environment."""
        try:
            return self.model_class(self.config)
        except Exception as e:
            error_msg = (
                f"\n{'='*70}\n"
                f"MODEL INSTANTIATION FAILED\n"
                f"{'='*70}\n"
                f"Model Class: {self.model_class.__name__}\n"
                f"Error: {str(e)}\n"
                f"\n"
                f"Possible causes:\n"
                f"  1. Missing required config attributes\n"
                f"  2. New SGLang dependencies not mocked\n"
                f"  3. Model requires actual CUDA/GPU\n"
                f"\n"
                f"To debug:\n"
                f"  - Check if config has all required fields\n"
                f"  - Update mock_sglang_environment() if new imports added\n"
                f"{'='*70}"
            )
            raise ModelInstantiationError(error_msg) from e

    def _fallback_instantiation(self) -> Any:
        """Fallback instantiation strategy when main method fails."""
        try:
            # Some models support creating only specific parts
            if hasattr(self.model_class, "model"):
                return self.model_class.model(self.config)
        except:
            pass

        # Return empty Module for AST-based fallback
        if nn is not None:
            return nn.Module()
        return None

    def _infer_config_class(self) -> str:
        """Infer corresponding Config class name from model class."""
        model_name = self.model_class.__name__

        # Common mappings
        mappings = {
            "Qwen3MoeForCausalLM": "Qwen3MoEConfig",
            "Qwen3ForCausalLM": "Qwen3Config",
            "DeepseekV3ForCausalLM": "DeepSeekV3Config",
            "DeepseekV2ForCausalLM": "DeepSeekV2Config",
        }

        if model_name in mappings:
            return mappings[model_name]

        # Generic rule: remove ForCausalLM/Model suffix, add Config
        base_name = model_name.replace("ForCausalLM", "").replace("Model", "")
        return f"{base_name}Config"

    def _identify_model_features(self, structure: ModelStructure):
        """Identify key model features from config."""
        config = self.config

        # Check for MoE
        if hasattr(config, "num_experts") or hasattr(config, "n_routed_experts"):
            structure.has_moe = True
            structure.first_k_dense_replace = getattr(
                config, "first_k_dense_replace", 0
            )
            if structure.first_k_dense_replace > 0:
                structure.has_dense_layers = True

        # Check for MLA
        if hasattr(config, "qk_rope_head_dim") and hasattr(config, "kv_lora_rank"):
            structure.has_mla = True
            structure.kv_cache_type = "mla"
            structure.kv_cache_dtype = "INT8"

        # Check for GQA
        kv_heads = getattr(config, "num_key_value_heads", 0)
        num_heads = getattr(config, "num_attention_heads", 1)
        if kv_heads > 0 and kv_heads != num_heads:
            structure.kv_cache_type = "mha_gqa"

    def _traverse_model(self, model: Any, structure: ModelStructure):
        """Traverse model and extract all layers."""
        if model is None or not hasattr(model, "named_modules"):
            return

        for name, module in model.named_modules():
            try:
                layer_info = self.recognizer.recognize(name, module)

                if not layer_info:
                    continue

                # Categorize by operator type
                if layer_info.op_category == "matmul":
                    structure.matmul_ops.append(layer_info)
                elif layer_info.op_category == "attention":
                    key = layer_info.op_name
                    structure.attention_ops.setdefault(key, []).append(layer_info)
                elif layer_info.op_category == "ffn":
                    structure.ffn_ops.append(layer_info)
                elif layer_info.op_category == "moe":
                    structure.matmul_ops.append(layer_info)
                elif layer_info.op_category == "moe_block":
                    self._traverse_moe_block(name, module, structure)

            except UnsupportedLayerError as e:
                # Collect unsupported layers for batch reporting
                structure.unsupported_layers.append(
                    {
                        "name": e.layer_name,
                        "type": e.layer_type,
                        "file": e.module_file or "Unknown",
                    }
                )

    def _traverse_moe_block(
        self, block_name: str, block: Any, structure: ModelStructure
    ):
        """Recursively traverse MoE Block internals."""
        if block is None or not hasattr(block, "named_modules"):
            return

        for name, module in block.named_modules():
            full_name = f"{block_name}.{name}"
            try:
                layer_info = self.recognizer.recognize(full_name, module)

                if not layer_info:
                    continue

                if layer_info.op_category == "matmul":
                    structure.matmul_ops.append(layer_info)
                elif layer_info.op_category == "moe":
                    structure.matmul_ops.append(layer_info)

            except UnsupportedLayerError as e:
                structure.unsupported_layers.append(
                    {
                        "name": e.layer_name,
                        "type": e.layer_type,
                        "file": e.module_file or "Unknown",
                    }
                )

    def _generate_transfer_ops(self, structure: ModelStructure):
        """Generate Transfer operators (AllReduce, DeepEP, etc.)."""
        # TP AllReduce for attention
        attn_allreduce = LayerInfo(
            name="attn_all_reduce",
            layer_type="TransferOperator",
            op_category="transfer",
            op_name="attn_all_reduce",
            input_shape=("seq_len", "mc.hidden_size"),
            output_shape=("seq_len", "mc.hidden_size"),
            weight_shape=(0, 0),
            dtype="BF16",
            batch_size=1,
        )
        structure.transfer_ops.append(attn_allreduce)

        # TP AllReduce for FFN (if not fully DP)
        ffn_allreduce = LayerInfo(
            name="dense_all_reduce",
            layer_type="TransferOperator",
            op_category="transfer",
            op_name="dense_all_reduce",
            input_shape=("seq_len", "mc.hidden_size"),
            output_shape=("seq_len", "mc.hidden_size"),
            weight_shape=(0, 0),
            dtype="BF16",
            batch_size=1,
        )
        structure.transfer_ops.append(ffn_allreduce)

        # DeepEP dispatch/combine (if MoE)
        if structure.has_moe:
            dispatch = LayerInfo(
                name="dispatch",
                layer_type="TransferOperator",
                op_category="transfer",
                op_name="dispatch",
                input_shape=("L", "mc.hidden_size"),
                output_shape=("L", "mc.hidden_size"),
                weight_shape=(0, 0),
                dtype="FP32",
                batch_size="mc.num_experts_per_tok",
            )
            combine = LayerInfo(
                name="combine",
                layer_type="TransferOperator",
                op_category="transfer",
                op_name="combine",
                input_shape=("L", "mc.hidden_size"),
                output_shape=("L", "mc.hidden_size"),
                weight_shape=(0, 0),
                dtype="BF16",
                batch_size="mc.num_experts_per_tok",
            )
            structure.transfer_ops.extend([dispatch, combine])

    def _report_unsupported_layers(self, structure: ModelStructure):
        """Report all unsupported layers found during extraction."""
        if not structure.unsupported_layers:
            return

        print(f"\n{'='*70}")
        print("WARNING: Unsupported layers detected during model analysis")
        print(f"{'='*70}")
        print(f"Total unsupported layers: {len(structure.unsupported_layers)}")
        print()

        # Group by layer type
        by_type: Dict[str, List[Dict]] = {}
        for layer in structure.unsupported_layers:
            layer_type = layer["type"]
            by_type.setdefault(layer_type, []).append(layer)

        for layer_type, layers in by_type.items():
            print(f"Layer Type: {layer_type}")
            print(f"  Count: {len(layers)}")
            print("  Examples:")
            for layer in layers[:3]:  # Show first 3 examples
                print(f"    - {layer['name']}")
            if len(layers) > 3:
                print(f"    ... and {len(layers) - 3} more")
            print()

        print("These layers were SKIPPED in the generated model_arch.")
        print("To add support, update LayerTypeRecognizer.LAYER_PATTERNS")
        print(f"{'='*70}\n")


# =============================================================================
# Code Generator
# =============================================================================


class ModelArchCodeGenerator:
    """
    Generates model_arch.py code from ModelStructure.

    Generated content:
    1. Import statements
    2. Class definition and docstring
    3. build_operators main method
    4. Helper methods (_build_attention_operators, _build_moe_operators, etc.)
    5. KV Cache calculation methods
    """

    INDENT = "    "

    def __init__(self, structure: ModelStructure):
        self.structure = structure
        self.lines: List[str] = []

    def generate(self) -> str:
        """Generate complete code string."""
        self._generate_imports()
        self._generate_class_def()
        self._generate_build_operators()
        self._generate_helper_methods()
        self._generate_kv_cache_methods()

        return "\n".join(self.lines)

    def _generate_imports(self):
        """Generate import statements."""
        imports = [
            "from src.arch.config import ForwardMode, " + self.structure.config_class,
            "from src.arch.models_arch.base_model_arch import BaseModelArch",
            "from src.arch.op.op_register import create_operator",
            "from src.arch.op.operator_base import DataType, OperatorIO, OperatorMetadata, Tensor",
        ]

        # Add KV Cache imports based on type
        if self.structure.kv_cache_type == "mla":
            imports.append(
                "from src.arch.kvcache.kvcache import mla_kvcache, mla_kvcache_per_gpu"
            )
        else:
            imports.append(
                "from src.arch.kvcache.kvcache import mha_gqa_kvcache, mha_gqa_kvcache_per_gpu"
            )

        self.lines.extend(imports)
        self.lines.append("")

    def _generate_class_def(self):
        """Generate class definition."""
        class_name = (
            self.structure.model_name.replace("ForCausalLM", "").replace("Model", "")
            + "Arch"
        )

        self.lines.extend(
            [
                f"class {class_name}(BaseModelArch):",
                f'    """Auto-generated architecture for {self.structure.model_name}"""',
                "",
            ]
        )

    def _generate_build_operators(self):
        """Generate build_operators main method."""
        self.lines.extend(
            [
                self.INDENT + "def build_operators(self) -> None:",
                self.INDENT * 2 + '"""Build operators for model."""',
                self.INDENT * 2 + "mc = self.model_config",
                self.INDENT * 2 + "sc = self.schedule_config",
                "",
                self.INDENT * 2
                + f"if not isinstance(mc, {self.structure.config_class}):",
                self.INDENT * 3
                + f"raise ValueError(f'Expected {self.structure.config_class}')",
                "",
            ]
        )

        # Calculate layer counts
        if self.structure.has_moe and self.structure.has_dense_layers:
            # DeepSeek style: dense first, then MoE
            self.lines.extend(
                [
                    self.INDENT * 2 + "# Layer configuration",
                    self.INDENT * 2
                    + "num_layers = mc.num_hidden_layers + (1 if sc.is_mtp else 0)",
                    self.INDENT * 2 + "num_dense_layers = mc.first_k_dense_replace",
                    self.INDENT * 2 + "num_moe_layers = num_layers - num_dense_layers",
                    "",
                    self.INDENT * 2 + "# 1. Build attention layers",
                    self.INDENT * 2 + "self._build_attention_operators(num_layers)",
                    "",
                    self.INDENT * 2 + "# 2. Build dense layers",
                    self.INDENT * 2 + "self._build_dense_operators(num_dense_layers)",
                    "",
                    self.INDENT * 2 + "# 3. Build MoE layers",
                    self.INDENT * 2 + "self._build_moe_operators(num_moe_layers)",
                ]
            )
        elif self.structure.has_moe:
            # Pure MoE model (e.g., Qwen3-MoE)
            self.lines.extend(
                [
                    self.INDENT * 2
                    + "num_layers = mc.num_hidden_layers + (1 if sc.is_mtp else 0)",
                    "",
                    self.INDENT * 2 + "# 1. Build attention layers",
                    self.INDENT * 2 + "self._build_attention_operators(num_layers)",
                    "",
                    self.INDENT * 2 + "# 2. Build MoE layers",
                    self.INDENT * 2 + "self._build_moe_operators(num_layers)",
                ]
            )
        else:
            # Dense model
            self.lines.extend(
                [
                    self.INDENT * 2 + "num_layers = mc.num_hidden_layers",
                    self.INDENT * 2 + "self._build_attention_operators(num_layers)",
                    self.INDENT * 2 + "self._build_ffn_operators(num_layers)",
                ]
            )

        # DeepEP
        if self.structure.has_moe:
            self.lines.extend(
                [
                    "",
                    self.INDENT * 2 + "# 4. Build Deep-EP transfer operators",
                    self.INDENT * 2 + "if sc.deepep:",
                    self.INDENT * 3
                    + "self._build_deepep_operators(num_moe_layers if num_dense_layers > 0 else num_layers)",
                ]
            )

        self.lines.append("")

    def _generate_helper_methods(self):
        """Generate helper methods for building operators."""
        # Attention operators
        if self.structure.attention_ops or self.structure.matmul_ops:
            self._generate_attention_method()

        # Dense/FFN operators
        if self.structure.has_dense_layers or not self.structure.has_moe:
            self._generate_dense_method()

        # MoE operators
        if self.structure.has_moe:
            self._generate_moe_method()

        # DeepEP operators
        if self.structure.has_moe:
            self._generate_deepep_method()

    def _generate_attention_method(self):
        """Generate _build_attention_operators method."""
        self.lines.extend(
            [
                self.INDENT
                + "def _build_attention_operators(self, num_layers: int) -> None:",
                self.INDENT * 2 + '"""Build attention operators."""',
                self.INDENT * 2 + "mc = self.model_config",
                self.INDENT * 2 + "sc = self.schedule_config",
                "",
            ]
        )

        # TP assertions and dimension calculations
        self.lines.extend(
            [
                self.INDENT * 2 + "# TP assertions",
                self.INDENT * 2 + "assert mc.num_attention_heads % sc.tp_size == 0",
                self.INDENT * 2 + "if mc.num_key_value_heads > sc.tp_size:",
                self.INDENT * 3 + "assert mc.num_key_value_heads % sc.tp_size == 0",
                self.INDENT * 2 + "else:",
                self.INDENT * 3 + "assert sc.tp_size % mc.num_key_value_heads == 0",
                "",
                self.INDENT * 2 + "# Calculate dimensions",
                self.INDENT * 2
                + "num_heads_per_rank = mc.num_attention_heads // sc.tp_size",
                self.INDENT * 2
                + "kv_heads_per_rank = max(1, mc.num_key_value_heads // sc.tp_size)",
                self.INDENT * 2 + "seq_len = self.get_seq_length()",
            ]
        )

        # Add head_dim calculation based on MLA or standard
        if self.structure.has_mla:
            self.lines.extend(
                [
                    self.INDENT * 2
                    + "qk_head_dim = mc.qk_nope_head_dim + mc.qk_rope_head_dim",
                ]
            )
        else:
            self.lines.append(
                self.INDENT * 2
                + "head_dim = getattr(mc, 'head_dim', mc.hidden_size // mc.num_attention_heads)"
            )

        self.lines.append("")

        # Generate matmul operators
        for op in self.structure.matmul_ops:
            if "proj" in op.op_name and ("attn" in op.name or "attention" in op.name):
                self._generate_operator_code(op, indent=2)

        # TP AllReduce
        self.lines.extend(
            [
                self.INDENT * 2 + "# TP AllReduce",
                self.INDENT * 2 + "if sc.tp_size > 1:",
                self.INDENT * 3 + "if sc.mode == ForwardMode.EXTEND:",
                self.INDENT * 4 + "reduce_bandwidth = 85.0  # GB/s",
                self.INDENT * 3 + "else:",
                self.INDENT * 4 + "reduce_bandwidth = 22.64  # GB/s",
                "",
            ]
        )

        # Attention core operators
        if self.structure.attention_ops:
            self.lines.extend(
                [
                    self.INDENT * 2 + "# Attention core",
                    self.INDENT * 2 + "attn_operators = []",
                    "",
                ]
            )

            for key, ops in self.structure.attention_ops.items():
                for op in ops:
                    self._generate_attention_operator_code(op, indent=2)

            self.lines.append(
                self.INDENT * 2
                + "self._add_attention_operator('attention', attn_operators)"
            )

        self.lines.append("")

    def _generate_dense_method(self):
        """Generate _build_dense_operators method."""
        self.lines.extend(
            [
                self.INDENT
                + "def _build_dense_operators(self, num_layers: int) -> None:",
                self.INDENT * 2 + '"""Build dense FFN operators."""',
                self.INDENT * 2 + "mc = self.model_config",
                self.INDENT * 2 + "sc = self.schedule_config",
                "",
                self.INDENT * 2 + "seq_len = self.get_seq_length()",
                self.INDENT * 2 + "assert mc.intermediate_size % sc.tp_size == 0",
                self.INDENT * 2
                + "intermediate_size = mc.intermediate_size // sc.tp_size",
                "",
            ]
        )

        # Generate dense layer matmul operators
        for op in self.structure.matmul_ops:
            if (
                "dense" in op.op_name
                or "gate_up" in op.op_name
                or "down_proj" in op.name
            ):
                if "moe" not in op.name.lower():
                    self._generate_operator_code(op, indent=2)

        self.lines.append("")

    def _generate_moe_method(self):
        """Generate _build_moe_operators method."""
        self.lines.extend(
            [
                self.INDENT
                + "def _build_moe_operators(self, num_layers: int) -> None:",
                self.INDENT * 2 + '"""Build MoE operators."""',
                self.INDENT * 2 + "mc = self.model_config",
                self.INDENT * 2 + "sc = self.schedule_config",
                "",
                self.INDENT * 2 + "seq_len = self.get_seq_length()",
            ]
        )

        # MoE specific calculations
        self.lines.extend(
            [
                self.INDENT * 2 + "# MoE configuration",
                self.INDENT * 2
                + "num_experts = getattr(mc, 'num_experts', getattr(mc, 'n_routed_experts', 1))",
                self.INDENT * 2 + "assert num_experts % sc.ep_size == 0",
                self.INDENT * 2 + "experts_per_rank = num_experts // sc.ep_size",
                "",
                self.INDENT * 2 + "# Calculate tokens per rank based on mode",
                self.INDENT * 2 + "if sc.mode == ForwardMode.EXTEND:",
                self.INDENT * 3 + "L = sc.max_seqlen",
                self.INDENT * 2 + "else:",
                self.INDENT * 3 + "L = sc.batch_size",
                self.INDENT * 2
                + "L_per_rank = L // sc.tp_size * mc.num_experts_per_tok // experts_per_rank",
                "",
            ]
        )

        # Generate MoE operators
        for op in self.structure.matmul_ops:
            if "moe" in op.op_name or "gate" in op.op_name:
                self._generate_operator_code(op, indent=2)

        self.lines.append("")

    def _generate_deepep_method(self):
        """Generate _build_deepep_operators method."""
        self.lines.extend(
            [
                self.INDENT
                + "def _build_deepep_operators(self, num_layers: int) -> None:",
                self.INDENT * 2 + '"""Build Deep-EP transfer operators."""',
                self.INDENT * 2 + "mc = self.model_config",
                self.INDENT * 2 + "sc = self.schedule_config",
                "",
                self.INDENT * 2 + "if sc.mode == ForwardMode.EXTEND:",
                self.INDENT * 3 + "L = sc.max_seqlen // sc.tp_size",
                self.INDENT * 3 + "dispatch_bandwidth = 100.0  # GB/s",
                self.INDENT * 3 + "combine_bandwidth = 100.0",
                self.INDENT * 2 + "else:",
                self.INDENT * 3 + "L = sc.batch_size // sc.tp_size",
                self.INDENT * 3 + "dispatch_bandwidth = 18.58  # GB/s",
                self.INDENT * 3 + "combine_bandwidth = 22.64",
                "",
            ]
        )

        # Generate dispatch/combine
        for op in self.structure.transfer_ops:
            if op.op_name in ["dispatch", "combine"]:
                self._generate_transfer_operator_code(op, indent=2)

        self.lines.append("")

    def _generate_operator_code(self, op: LayerInfo, indent: int = 2):
        """Generate code for a single matmul operator."""
        ind = self.INDENT * indent

        self.lines.extend(
            [
                f"{ind}# {op.op_name}",
                f"{ind}{op.op_name}_metadata = OperatorMetadata(",
                f"{ind}    name='{op.op_name}',",
                f"{ind}    op_type='matmul',",
                f"{ind}    io_config=OperatorIO(",
                f"{ind}        input_shape=Tensor({self._shape_to_str(op.input_shape)}),",
                f"{ind}        output_shape=Tensor({self._shape_to_str(op.output_shape)}),",
                f"{ind}        weight_shape=Tensor({self._shape_to_str(op.weight_shape)}),",
                f"{ind}        input_dtype=DataType.{op.dtype},",
                f"{ind}        output_dtype=DataType.{op.dtype},",
                f"{ind}        weight_dtype=DataType.{op.dtype},",
                f"{ind}    ),",
                f"{ind}    batch_size={op.batch_size},",
                f"{ind}    num_layers=num_layers,",
                f"{ind})",
                f"{ind}self._add_operator(create_operator('matmul', {op.op_name}_metadata))",
                "",
            ]
        )

    def _generate_attention_operator_code(self, op: LayerInfo, indent: int = 2):
        """Generate code for an attention operator."""
        ind = self.INDENT * indent
        attn_type = op.attention_type or "mc.attention_type"

        self.lines.extend(
            [
                f"{ind}{op.op_name}_metadata = OperatorMetadata(",
                f"{ind}    name='{op.op_name}',",
                f"{ind}    op_type='attention',",
                f"{ind}    io_config=OperatorIO(",
                f"{ind}        input_shape=Tensor({self._shape_to_str(op.input_shape)}),",
                f"{ind}        output_shape=Tensor({self._shape_to_str(op.output_shape)}),",
                f"{ind}        weight_shape=Tensor({self._shape_to_str(op.weight_shape)}),",
                f"{ind}        input_dtype=DataType.{op.dtype},",
                f"{ind}        output_dtype=DataType.{op.dtype},",
                f"{ind}        weight_dtype=DataType.{op.dtype},",
                f"{ind}    ),",
                f"{ind}    batch_size={op.batch_size},",
                f"{ind}    num_layers=num_layers,",
                f"{ind})",
                f"{ind}attn_operators.append(create_operator('attention', {op.op_name}_metadata, {attn_type}))",
                "",
            ]
        )

    def _generate_transfer_operator_code(self, op: LayerInfo, indent: int = 2):
        """Generate code for a transfer operator."""
        ind = self.INDENT * indent
        bandwidth_var = f"{op.op_name}_bandwidth"

        self.lines.extend(
            [
                f"{ind}# {op.op_name}",
                f"{ind}{op.op_name}_metadata = OperatorMetadata(",
                f"{ind}    name='{op.op_name}',",
                f"{ind}    op_type='transfer',",
                f"{ind}    io_config=OperatorIO(",
                f"{ind}        input_shape=Tensor({self._shape_to_str(op.input_shape)}),",
                f"{ind}        output_shape=Tensor({self._shape_to_str(op.output_shape)}),",
                f"{ind}        weight_shape=Tensor(0, 0),",
                f"{ind}        input_dtype=DataType.{op.dtype},",
                f"{ind}        output_dtype=DataType.{op.dtype},",
                f"{ind}    ),",
                f"{ind}    batch_size={op.batch_size},",
                f"{ind}    num_layers=num_layers,",
                f"{ind})",
                f"{ind}{op.op_name}_op = create_operator('transfer', {op.op_name}_metadata)",
                f"{ind}{op.op_name}_op._bandwidth_gb_s = {bandwidth_var}",
                f"{ind}self._add_transfer_operator({op.op_name}_op)",
                "",
            ]
        )

    def _generate_kv_cache_methods(self):
        """Generate KV Cache calculation methods."""
        if self.structure.kv_cache_type == "mla":
            self.lines.extend(
                [
                    self.INDENT + "def get_kv_cache(self):",
                    self.INDENT * 2
                    + "return mla_kvcache(self.model_config, DataType.INT8)",
                    "",
                    self.INDENT + "def get_kv_cache_per_gpu(self):",
                    self.INDENT * 2
                    + "return mla_kvcache_per_gpu(self.model_config, DataType.INT8, self.schedule_config.tp_size)",
                ]
            )
        else:
            dtype = self.structure.kv_cache_dtype
            self.lines.extend(
                [
                    self.INDENT + "def get_kv_cache(self):",
                    self.INDENT * 2
                    + f"return mha_gqa_kvcache(self.model_config, DataType.{dtype})",
                    "",
                    self.INDENT + "def get_kv_cache_per_gpu(self):",
                    self.INDENT * 2
                    + f"return mha_gqa_kvcache_per_gpu(self.model_config, DataType.{dtype}, self.schedule_config.tp_size)",
                ]
            )

    def _shape_to_str(self, shape: Tuple) -> str:
        """Convert shape tuple to code string."""
        parts = []
        for s in shape:
            if isinstance(s, str):
                parts.append(s)
            else:
                parts.append(str(s))
        return ", ".join(parts)


# =============================================================================
# Main Entry Point
# =============================================================================


def generate_model_arch(
    model_class: Type, config: Any, output_path: Optional[str] = None
) -> str:
    """
    Main entry point: Generate model_arch.py from SGLang model class.

    Args:
        model_class: SGLang model class, e.g., Qwen3MoeForCausalLM
        config: Model configuration object (transformers.PretrainedConfig or similar)
        output_path: Optional output file path

    Returns:
        Generated code string

    Raises:
        MissingDependencyError: If PyTorch is not installed
        ModelInstantiationError: If model cannot be instantiated
        UnsupportedLayerError: If model contains unsupported layers

    Example:
        >>> from transformers import AutoConfig
        >>> config = AutoConfig.from_pretrained("Qwen/Qwen3-235B-A22B")
        >>> from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
        >>> code = generate_model_arch(Qwen3MoeForCausalLM, config, "output.py")
    """
    # Check dependencies
    if not HAS_TORCH:
        raise MissingDependencyError(
            "PyTorch is required for model analysis.\n"
            "Install with: pip install torch"
        )

    # Extract model structure
    extractor = ModelStructureExtractor(config, model_class)
    structure = extractor.extract()

    # Generate code
    generator = ModelArchCodeGenerator(structure)
    code = generator.generate()

    # Optionally save to file
    if output_path:
        with open(output_path, "w") as f:
            f.write(code)
        print(f"Generated model arch saved to: {output_path}")

    return code


if __name__ == "__main__":
    print("SGLang Model Architecture Auto-Generator")
    print("=" * 50)
    print("\nUsage:")
    print("  from auto_generator import generate_model_arch")
    print("  code = generate_model_arch(ModelClass, config, 'output.py')")
