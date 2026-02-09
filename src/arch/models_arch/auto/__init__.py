"""
SGLang Auto Adapter - Automatic model architecture adaptation from HuggingFace Config.

This package provides config-driven model adaptation to LLMSim:

    from transformers import AutoConfig
    from src.arch.config import ScheduleConfig, ForwardMode
    from src.arch.models_arch.auto import auto_adapter_from_config

    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
    schedule_config = ScheduleConfig(
        mode=ForwardMode.EXTEND,
        tp_size=4,
        dp_size=2,
        ep_size=8,
    )

    model_arch = auto_adapter_from_config(config, schedule_config)

Benefits:
- No SGLang model instantiation required
- No mocking of distributed environment needed
- Faster and more stable
- Easy to extend for new model types
"""

from src.arch.models_arch.auto.adapter import auto_adapter_from_config
from src.arch.models_arch.auto.config_builder import (
    build_ir_from_config,
    get_config_builder,
    infer_model_type,
    list_registered_builders,
    register_config_builder,
)

__all__ = [
    # Main API
    "auto_adapter_from_config",
    # Config builder utilities
    "build_ir_from_config",
    "get_config_builder",
    "infer_model_type",
    "list_registered_builders",
    "register_config_builder",
]
