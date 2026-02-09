"""
Example: Config-driven model adaptation for Qwen3 MoE.

This example demonstrates the new recommended approach:
1. Load HuggingFace config
2. Create schedule config
3. Build ModelArch directly from config (no SGLang model instantiation needed)

Benefits:
- No need to mock distributed environment
- Faster (no PyTorch model creation)
- More stable (not affected by SGLang internal changes)
"""

from transformers import AutoConfig

from src.arch.config import ForwardMode, ScheduleConfig
from src.arch.models_arch.auto import auto_adapter_from_config
from src.arch.perf_calculator import PerformanceCalculator
from src.hardware import hardware_config

# Load HuggingFace config
print("Loading config from HuggingFace...")
config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")

# Create schedule config
schedule_config = ScheduleConfig(
    mode=ForwardMode.EXTEND,
    tp_size=4,
    dp_size=2,
    ep_size=8,
)

# Build model architecture directly from config
# This does NOT instantiate SGLang models - it builds IR directly from config
print("Building model architecture from config...")
model_arch = auto_adapter_from_config(config, schedule_config)

print(f"Model type: {model_arch.model_config.model_type}")
print(f"Number of layers: {model_arch.model_config.num_hidden_layers}")
print(f"Hidden size: {model_arch.model_config.hidden_size}")

# Run performance calculation
print("\nRunning performance calculation...")
calculator = PerformanceCalculator(hardware_config.DEFAULT_HARDWARE)
result = calculator.calculate_model_performance(model_arch)

print("\nDone!")
