"""
Base layer parser - Abstract interface for all layer parsers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from src.arch.models_arch.auto.ir import OpNode


try:
    import torch.nn as nn

    ModuleType = nn.Module
except ImportError:
    nn = None  # type: ignore
    ModuleType = Any


class BaseLayerParser(ABC):
    """
    Abstract base class for layer parsers.

    Each parser handles specific PyTorch layer types and extracts
    shape information to create OpNode IR representations.

    Example:
        @register_layer_parser("Linear", "RowParallelLinear")
        class LinearParser(BaseLayerParser):
            @property
            def layer_types(self) -> List[str]:
                return ["Linear", "RowParallelLinear"]

            def parse(self, name: str, module: nn.Module, config: Any) -> Optional[OpNode]:
                # Extract dimensions and create OpNode
                return OpNode(...)
    """

    @property
    @abstractmethod
    def layer_types(self) -> List[str]:
        """
        Return list of PyTorch layer type names this parser supports.

        Returns:
            List of class names (e.g., ["Linear", "RowParallelLinear"])
        """
        pass

    @abstractmethod
    def parse(self, name: str, module: ModuleType, config: Any) -> Optional["OpNode"]:
        """
        Parse a layer and return an OpNode.

        Args:
            name: Full path of the layer in model hierarchy
                  (e.g., "model.layers.0.self_attn.qkv_proj")
            module: PyTorch nn.Module instance
            config: Model configuration object

        Returns:
            OpNode instance, or None if layer should be skipped
        """
        pass

    def can_parse(self, module: ModuleType) -> bool:
        """
        Check if this parser can handle the given module.

        Args:
            module: PyTorch module to check

        Returns:
            True if this parser can handle the module type
        """
        layer_type = type(module).__name__
        return layer_type in self.layer_types
