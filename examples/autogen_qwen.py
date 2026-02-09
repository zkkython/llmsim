# From HuggingFace config
# Import SGLang model
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from transformers import AutoConfig

from src.arch.config import ForwardMode, ScheduleConfig
from src.arch.models_arch.auto import auto_adapter

# Use for performance calculation
from src.arch.perf_calculator import PerformanceCalculator
from src.hardware import hardware_config

config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")


# Create schedule config
schedule_config = ScheduleConfig(
    mode=ForwardMode.EXTEND,
    tp_size=4,
    dp_size=2,
    ep_size=8,
)

# Adapt model
model_arch = auto_adapter(Qwen3MoeForCausalLM, config, schedule_config)


calculator = PerformanceCalculator(hardware_config.DEFAULT_HARDWARE)
result = calculator.calculate_model_performance(model_arch)
