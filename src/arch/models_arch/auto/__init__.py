"""
SGLang Auto Adapter - Automatic model architecture adaptation from SGLang models.

This package provides a layered IR architecture for adapting SGLang models to LLMSim:
- IR Layer: Framework-agnostic intermediate representation
- Parser Layer: SGLang model to IR conversion
- Transformer Layer: IR to LLMSim ModelArch conversion

Usage:
    from src.arch.models_arch.auto import auto_adapter
    model_arch = auto_adapter(Qwen3MoeForCausalLM, config, schedule_config)
"""

from src.arch.models_arch.auto.adapter import SglangAutoAdapter, auto_adapter
from src.arch.models_arch.auto.layer_parsers.registry import register_layer_parser

__all__ = [
    "auto_adapter",
    "SglangAutoAdapter",
    "register_layer_parser",
]
