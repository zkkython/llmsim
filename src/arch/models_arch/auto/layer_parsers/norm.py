"""
Normalization layer parsers.

Normalization layers (RMSNorm, LayerNorm) typically don't generate
compute operators in LLMSim as their cost is negligible compared to
matmul and attention operations.
"""

from typing import Any, List, Optional

from src.arch.models_arch.auto.ir import OpNode
from src.arch.models_arch.auto.layer_parsers.base import BaseLayerParser
from src.arch.models_arch.auto.layer_parsers.registry import register_layer_parser

try:
    import torch.nn as nn

    ModuleType = nn.Module
except ImportError:
    nn = None  # type: ignore
    ModuleType = Any


@register_layer_parser("RMSNorm", "LayerNorm", "FusedRMSNorm")
class NormParser(BaseLayerParser):
    """
    Parser for normalization layers.

    Normalization layers are skipped in LLMSim as their computational
    cost is negligible compared to matrix multiplications and attention.
    """

    @property
    def layer_types(self) -> List[str]:
        return ["RMSNorm", "LayerNorm", "FusedRMSNorm"]

    def parse(self, name: str, module: ModuleType, config: Any) -> Optional[OpNode]:
        # Return None to skip normalization layers
        return None
