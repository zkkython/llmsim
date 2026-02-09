"""
SGLang Model Parser - Extracts IR from SGLang models.

This module provides the parsing logic to convert SGLang PyTorch models
into the framework-agnostic ComputationalGraph IR representation.
"""

import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Type

from src.arch.models_arch.auto.ir import ComputationalGraph, OpNode
from src.arch.models_arch.auto.layer_parsers.registry import get_parser

try:
    import torch.nn as nn

    HAS_TORCH = True
    ModuleType = nn.Module
except ImportError:
    nn = None  # type: ignore
    HAS_TORCH = False
    ModuleType = Any


# =============================================================================
# Mock Environment for Bypassing SGLang Dependencies
# =============================================================================


class MockModule:
    """Base class for mock modules."""

    def __init__(self, name: str):
        self.__name__ = name

    def __getattr__(self, name: str):
        return MockCallable(f"{self.__name__}.{name}")


class MockCallable:
    """Mock callable that accepts any arguments."""

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name: str):
        return MockCallable(f"{self._name}.{name}")


def create_mock_environment() -> Dict[str, Any]:
    """Create mock environment to bypass SGLang dependencies."""
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
    """
    mock_modules = create_mock_environment()
    original_modules = {}

    for name, mock in mock_modules.items():
        if name in sys.modules:
            original_modules[name] = sys.modules[name]
        sys.modules[name] = mock

    try:
        yield
    finally:
        for name in mock_modules:
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            elif name in sys.modules:
                del sys.modules[name]


# =============================================================================
# Exception Classes
# =============================================================================


class ParserError(Exception):
    """Base exception for parser errors."""

    pass


class MissingDependencyError(ParserError):
    """Raised when required dependencies are not installed."""

    pass


class ModelInstantiationError(ParserError):
    """Raised when model instantiation fails."""

    pass


class UnsupportedLayerError(ParserError):
    """Raised when encountering an unsupported layer type."""

    def __init__(
        self, layer_name: str, layer_type: str, module_file: Optional[str] = None
    ):
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.module_file = module_file
        super().__init__(f"Unsupported layer: {layer_type} at {layer_name}")


# =============================================================================
# Model Structure Extractor
# =============================================================================


class ModelParser:
    """
    Parses SGLang models and extracts ComputationalGraph IR.

    Usage:
        parser = ModelParser(config)
        graph = parser.parse(model_class)
    """

    # Layer types to skip (embeddings, etc.)
    SKIP_LAYER_TYPES: Set[str] = {
        "Embedding",
        "RotaryEmbedding",
        "LlamaRotaryEmbedding",
        "Qwen2RotaryEmbedding",
    }

    def __init__(self, config: Any):
        if not HAS_TORCH:
            raise MissingDependencyError(
                "PyTorch is required. Install with: pip install torch"
            )
        self.config = config
        self._unsupported_layers: List[Dict[str, str]] = []

    def parse(self, model_class: Type) -> ComputationalGraph:
        """
        Parse a model class and return ComputationalGraph.

        Args:
            model_class: SGLang model class (e.g., Qwen3MoeForCausalLM)

        Returns:
            ComputationalGraph IR representation
        """
        with mock_sglang_environment():
            model = self._instantiate_model(model_class)

        graph = self._create_graph(model_class)
        self._traverse_model(model, graph)

        # Report unsupported layers
        if self._unsupported_layers:
            self._report_unsupported_layers()

        return graph

    def _instantiate_model(self, model_class: Type) -> Any:
        """Instantiate model within mock environment."""
        try:
            return model_class(self.config)
        except Exception as e:
            raise ModelInstantiationError(
                f"Failed to instantiate {model_class.__name__}: {e}"
            ) from e

    def _create_graph(self, model_class: Type) -> ComputationalGraph:
        """Create initial ComputationalGraph with model metadata."""
        model_name = model_class.__name__

        # Determine model type
        model_type = self._infer_model_type()

        # Create config dict for reference
        config_dict = self._config_to_dict()

        return ComputationalGraph(
            model_name=model_name,
            model_type=model_type,
            config=config_dict,
            has_moe=self._has_moe(),
            has_mla=self._has_mla(),
            has_dense_layers=self._has_dense_layers(),
            first_k_dense_replace=getattr(self.config, "first_k_dense_replace", 0),
            kv_cache_type=self._infer_kv_cache_type(),
        )

    def _infer_model_type(self) -> str:
        """Infer model type from config."""
        if self._has_mla():
            return "mla"
        elif self._has_moe():
            return "moe"
        return "dense"

    def _has_moe(self) -> bool:
        """Check if model has MoE layers."""
        num_experts = getattr(self.config, "num_experts", None)
        n_routed_experts = getattr(self.config, "n_routed_experts", None)
        return (isinstance(num_experts, int) and num_experts > 0) or (
            isinstance(n_routed_experts, int) and n_routed_experts > 0
        )

    def _has_mla(self) -> bool:
        """Check if model uses MLA attention."""
        qk_rope_head_dim = getattr(self.config, "qk_rope_head_dim", None)
        kv_lora_rank = getattr(self.config, "kv_lora_rank", None)
        return isinstance(qk_rope_head_dim, int) and isinstance(kv_lora_rank, int)

    def _has_dense_layers(self) -> bool:
        """Check if MoE model has initial dense layers."""
        first_k = getattr(self.config, "first_k_dense_replace", 0)
        try:
            return first_k > 0
        except TypeError:
            return False

    def _infer_kv_cache_type(self) -> str:
        """Infer KV cache type from config."""
        if self._has_mla():
            return "mla"

        kv_heads = getattr(self.config, "num_key_value_heads", 0)
        num_heads = getattr(self.config, "num_attention_heads", 1)

        try:
            if kv_heads > 0 and kv_heads != num_heads:
                return "gqa"
        except TypeError:
            pass

        return "mha"

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        if hasattr(self.config, "to_dict"):
            return self.config.to_dict()

        # Extract attributes
        result = {}
        for key in dir(self.config):
            if not key.startswith("_"):
                try:
                    value = getattr(self.config, key)
                    if not callable(value):
                        result[key] = value
                except Exception:
                    pass
        return result

    def _traverse_model(self, model: Any, graph: ComputationalGraph, prefix: str = ""):
        """Traverse model and extract all layers."""
        if model is None or not hasattr(model, "named_modules"):
            return

        for name, module in model.named_modules():
            full_name = f"{prefix}.{name}" if prefix else name

            # Skip if no module or is the root
            if module is None or name == "":
                continue

            try:
                op_node = self._parse_layer(full_name, module)

                if op_node is None:
                    continue

                graph.add_node(op_node)

            except UnsupportedLayerError as e:
                self._unsupported_layers.append(
                    {
                        "name": e.layer_name,
                        "type": e.layer_type,
                        "file": e.module_file or "Unknown",
                    }
                )

    def _parse_layer(self, name: str, module: ModuleType) -> Optional[OpNode]:
        """
        Parse a single layer and return OpNode.

        Args:
            name: Full path of the layer
            module: PyTorch module

        Returns:
            OpNode or None if layer should be skipped
        """
        layer_type = type(module).__name__

        # Skip certain layer types
        if layer_type in self.SKIP_LAYER_TYPES:
            return None

        # Get registered parser
        parser = get_parser(layer_type)

        if parser is not None:
            return parser.parse(name, module, self.config)

        # No parser found - check if it's a composite we should traverse
        if self._is_composite_layer(layer_type):
            return None  # Will be handled by traversing children

        # Truly unsupported layer
        module_file = getattr(type(module), "__module__", None)
        raise UnsupportedLayerError(name, layer_type, module_file)

    def _is_composite_layer(self, layer_type: str) -> bool:
        """Check if layer type is a composite that should be traversed."""
        composite_types = {
            "Qwen3MoeSparseMoeBlock",
            "DeepseekV3MoE",
            "TransformerBlock",
            "LlamaDecoderLayer",
            "Qwen2DecoderLayer",
        }
        return layer_type in composite_types

    def _report_unsupported_layers(self):
        """Report all unsupported layers found during parsing."""
        print(f"\n{'='*70}")
        print("WARNING: Unsupported layers detected during model parsing")
        print(f"{'='*70}")
        print(f"Total unsupported layers: {len(self._unsupported_layers)}")
        print()

        # Group by layer type
        by_type: Dict[str, List[Dict]] = {}
        for layer in self._unsupported_layers:
            layer_type = layer["type"]
            by_type.setdefault(layer_type, []).append(layer)

        for layer_type, layers in by_type.items():
            print(f"Layer Type: {layer_type}")
            print(f"  Count: {len(layers)}")
            print("  Examples:")
            for layer in layers[:3]:
                print(f"    - {layer['name']}")
            if len(layers) > 3:
                print(f"    ... and {len(layers) - 3} more")
            print()

        print("These layers were SKIPPED.")
        print("To add support, register a new layer parser with @register_layer_parser")
        print(f"{'='*70}\n")
