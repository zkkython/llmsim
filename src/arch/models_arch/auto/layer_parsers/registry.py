"""
Layer parser registry - Plugin system for layer parsers.

Provides a decorator-based registration system for layer parsers,
enabling extensible support for new layer types without modifying core code.
"""

from typing import TYPE_CHECKING, Dict, Optional, Type

if TYPE_CHECKING:
    from src.arch.models_arch.auto.layer_parsers.base import BaseLayerParser

# Global registry mapping layer type names to parser instances
_layer_parsers: Dict[str, "BaseLayerParser"] = {}


def register_layer_parser(*layer_types: str):
    """
    Decorator to register a layer parser for specific layer types.

    Args:
        *layer_types: Variable number of layer type names to register for

    Returns:
        Decorator function that registers the parser class

    Example:
        @register_layer_parser("Linear", "RowParallelLinear")
        class LinearParser(BaseLayerParser):
            @property
            def layer_types(self) -> List[str]:
                return ["Linear", "RowParallelLinear"]

            def parse(self, name: str, module: nn.Module, config: Any) -> OpNode:
                ...
    """

    def decorator(parser_class: Type["BaseLayerParser"]):
        parser = parser_class()
        for layer_type in layer_types:
            _layer_parsers[layer_type] = parser
        return parser_class

    return decorator


def get_parser(layer_type: str) -> Optional["BaseLayerParser"]:
    """
    Get the registered parser for a layer type.

    Args:
        layer_type: PyTorch layer class name

    Returns:
        Registered parser instance, or None if not found
    """
    return _layer_parsers.get(layer_type)


def list_registered_parsers() -> Dict[str, str]:
    """
    List all registered parsers.

    Returns:
        Dictionary mapping layer types to parser class names
    """
    return {
        layer_type: type(parser).__name__
        for layer_type, parser in _layer_parsers.items()
    }


def unregister_parser(layer_type: str) -> bool:
    """
    Unregister a parser for a specific layer type.

    Args:
        layer_type: Layer type to unregister

    Returns:
        True if a parser was removed, False if not found
    """
    if layer_type in _layer_parsers:
        del _layer_parsers[layer_type]
        return True
    return False
