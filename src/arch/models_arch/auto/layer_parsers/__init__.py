"""
Layer parsers package - Plugin-based layer parsing system.

Each parser extracts shape information from SGLang/PyTorch layers
and converts them to OpNode IR representations.
"""

# Import all parser modules to trigger registration
# These imports register parsers via the @register_layer_parser decorator
from src.arch.models_arch.auto.layer_parsers import (
    attention,
    linear,
    moe,
    norm,
)
from src.arch.models_arch.auto.layer_parsers.base import BaseLayerParser
from src.arch.models_arch.auto.layer_parsers.registry import (
    get_parser,
    register_layer_parser,
)

__all__ = [
    "BaseLayerParser",
    "register_layer_parser",
    "get_parser",
    # Re-export parser classes for direct use
    "linear",
    "attention",
    "moe",
    "norm",
]
