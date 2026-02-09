"""
Main Adapter API - High-level interface for model adaptation from HuggingFace Config.

This module provides the main entry point for building LLMSim ModelArch directly
from HuggingFace config objects, without requiring SGLang model instantiation.

Usage:
    from src.arch.models_arch.auto import auto_adapter_from_config

    # Build ModelArch directly from config
    model_arch = auto_adapter_from_config(config, schedule_config)
"""

from typing import TYPE_CHECKING, Any

from src.arch.config import ScheduleConfig

if TYPE_CHECKING:
    from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.models_arch.auto.config_builder import build_ir_from_config
from src.arch.models_arch.auto.transformer import IRToModelArchTransformer


def auto_adapter_from_config(
    config: Any,
    schedule_config: ScheduleConfig,
    model_type: str | None = None,
) -> "BaseModelArch":
    """
    Build ModelArch directly from HuggingFace config.

    This is the config-driven approach that does NOT require:
    - SGLang model class instantiation
    - Mocking distributed environment
    - PyTorch model creation

    It directly builds the IR from config parameters and transforms to ModelArch.

    Args:
        config: HuggingFace config object (from AutoConfig.from_pretrained())
        schedule_config: Runtime scheduling configuration
        model_type: Optional model type override. If not provided, will be inferred.

    Returns:
        Configured ModelArch instance

    Raises:
        ValueError: If model_type cannot be inferred or no builder is registered

    Example:
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

        # Model type is automatically inferred from config
        model_arch = auto_adapter_from_config(config, schedule_config)

        # Or explicitly specify model type
        model_arch = auto_adapter_from_config(config, schedule_config, model_type="qwen3_moe")
    """
    # Build IR from config
    ir_graph = build_ir_from_config(config, model_type)

    # Transform to ModelArch
    transformer = IRToModelArchTransformer(ir_graph, schedule_config)
    return transformer.transform()
