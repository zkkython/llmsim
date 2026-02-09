# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMSim is a Python-based LLM inference performance analysis tool that models theoretical FLOPs and execution efficiency for different model architectures (DeepSeek-V3, Qwen3) on various hardware platforms.

## Common Commands

### Setup
```bash
pip install -e .
# Or with dev dependencies:
pip install -e ".[dev]"
```

### Running the Tool
```bash
# DeepSeek V3 Prefill example
python3 src/main.py \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --max_seqlen 4096 --batch_size 1 --mode extend \
    --tp_size 4 --dp_size 4 --ep_size 16 \
    --enable_deepep --hardware klx_p800

# Qwen3 example
python3 src/main.py \
    --model_path hf_config/qwen3-32B_config.json \
    --max_seqlen 8192 --batch_size 16 --mode extend \
    --tp_size 8 --dp_size 2 --hardware h800

# With Excel output
python3 src/main.py ... --output_format excel --output_file result.xlsx

# Config-driven auto adapter example (recommended)
python3 examples/autogen_qwen.py
```

### Code Quality
```bash
# Run all linting/formatting (must pass before committing)
pre-commit run --all-files

# Individual tools
ruff check src/
isort src/
black src/
```

### Testing
```bash
pytest
pytest -v
```

## Architecture

### High-Level Flow
```
main.py → ModelConfig.from_json() → create_model_arch() → PerformanceCalculator → Output Report
```

### Key Components

**Model Architecture Layer** (`src/arch/models_arch/`)
- `model_arch.py`: Factory that creates architecture instances based on `model_type` (deepseek_v3, qwen3, qwen3_moe)
- `base_model_arch.py`: Abstract base defining the interface for all model architectures
- `deepseek_v3_model_arch.py`: DeepSeek V3 with MLA + MoE
- `qwen3_moe_model_arch.py`: Qwen3 MoE variant
- `simple_model_arch.py`: Standard dense Transformer (used for Qwen3 dense models)

**Config-driven Auto Adapter** (`src/arch/models_arch/auto/`)
- `adapter.py`: High-level API `auto_adapter_from_config()` for building ModelArch from HF config
- `config_builder.py`: Registry-based IR builders for each model type (qwen3_moe, deepseek_v3, qwen3)
- `ir.py`: Intermediate Representation (IR) - ComputationalGraph, OpNode, ShapeSpec
- `transformer.py`: Converts IR to LLMSim ModelArch

Benefits of config-driven approach:
- No SGLang model instantiation required
- No mocking of distributed environment needed
- Faster and more stable
- Easy to extend for new model types via `@register_config_builder`

Example usage:
```python
from src.arch.models_arch.auto import auto_adapter_from_config
model_arch = auto_adapter_from_config(config, schedule_config)
```

**Operator Layer** (`src/arch/op/`)
Each operator implements `BaseOperator` and provides:
- `get_compute_complexity()`: Returns FLOPs count
- `get_io_volume()`: Returns memory access bytes
- `get_params()`: Returns parameter count

Operators:
- `attn_op.py`: Multi-Head Attention (MHA)
- `mla_attn_op.py`: Multi-Head Latent Attention (MLA) - DeepSeek's efficient attention
- `ffn_op.py`: Feed-Forward Network and MoE FFN
- `matmul_op.py`: Matrix multiplication operations
- `transfer_op.py`: Communication operations (all-reduce, all-to-all)

**Performance Calculation** (`src/arch/perf_calculator.py`)
- Calculates compute time: `FLOPs / (MAC_GFLOPS * 1e6)` → microseconds
- Calculates memory time: `bytes / (HBM_bandwidth_GB_s * 1e6)` → microseconds
- Roofline model: `total_time = max(compute_time, memory_time)` for each operator

**Hardware Modeling** (`src/hardware/hardware_config.py`)
- `HardwareConfig` dataclass with memory, bandwidth, and compute specs
- Predefined configs: h20, h800, gb200, klx_p800
- Supports custom JSON/JSON5 hardware configs

### Configuration System

**Model Configs** (`hf_config/`): JSON files defining model architecture parameters
- Loaded via `ModelConfig.from_json()` which auto-detects model type
- Supports loading from HuggingFace hub if path doesn't exist locally

**Schedule Config** (`src/arch/config.py`): Runtime parameters
- `mode`: "extend" (prefill) or "decode"
- Parallelism: `tp_size`, `dp_size`, `ep_size`
- Features: `is_mtp`, `deepep`, `enable_moe_dense_fully_dp`

### Data Flow

1. **ModelConfig.from_json()** loads model architecture parameters
2. **create_model_arch()** instantiates the appropriate architecture class
3. **ModelArch.build_model()** constructs the computation graph by creating operators layer by layer
4. **PerformanceCalculator.calculate_model_performance()** iterates through all operators, computing:
   - FLOPs and compute time
   - Memory access volume and time
   - Communication time for transfer ops
5. Results aggregated into `ModelInfo` and output via console or Excel

### Key Design Patterns

- **Factory Pattern**: `create_model_arch()` creates appropriate architecture based on `model_type`
- **Strategy Pattern**: Different operator classes implement `BaseOperator` with standard interfaces
- **Dataclass Configs**: `ModelConfig`, `ScheduleConfig`, `HardwareConfig` are dataclasses with `from_dict()` methods
- **Roofline Model**: Performance is bound by `max(compute_time, memory_time)` for each operator

## File Organization

```
src/
├── main.py                    # CLI entry point
├── arch/
│   ├── config.py              # ModelConfig, ScheduleConfig dataclasses
│   ├── model_type.py          # Enums (AttentionType, ForwardMode, etc.)
│   ├── perf_calculator.py     # Core performance calculation engine
│   ├── models_arch/           # Model architecture implementations
│   │   └── auto/              # Config-driven auto adapter
│   │       ├── adapter.py     # auto_adapter_from_config() API
│   │       ├── config_builder.py  # IR builders for each model type
│   │       ├── ir.py          # IR data structures
│   │       └── transformer.py # IR to ModelArch converter
│   ├── op/                    # Operator implementations with FLOP calculations
│   └── perf/                  # Performance data structures
├── hardware/                  # Hardware configuration
└── visual/                    # Output formatting (console, Excel)

hf_config/                     # Pre-configured model configs
hardware_config/               # Pre-configured hardware configs
```
