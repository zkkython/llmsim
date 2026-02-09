# LLMSim - LLM Inference Performance Analysis Tool

LLMSim is a **computational performance modeling and hardware adaptation analysis tool** for Large Language Models, focusing on quantifying theoretical FLOPs and actual execution efficiency of different model architectures (such as Qwen3, DeepSeek) on various hardware platforms.

**Core Value**: Provides data-driven decision support for model selection, hardware deployment, and inference optimization.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                      │
│                    (Command Line / CLI Arguments)                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      Core Computation Engine                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Model Arch    │  │  Perf Calculator│  │   Schedule      │  │
│  │   Parsing       │  │                 │  │   Config        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│   Operator     │  │   Hardware      │  │   Output        │
│   Layer (Ops)  │  │   Modeling      │  │   Report        │
│  ├─ Attention  │  │  ├─ GPU/XPU     │  │  ├─ Console     │
│  ├─ FFN/MoE    │  │  ├─ Bandwidth   │  │  ├─ Excel       │
│  ├─ MatMul     │  │  ├─ Compute     │  │  └─ Report      │
│  └─ Transfer   │  │  └─ Memory      │  │                 │
└────────────────┘  └─────────────────┘  └─────────────────┘
```

### Module Responsibilities

| Module | Path | Responsibility |
|--------|------|----------------|
| **Model Architecture** | `src/arch/models_arch/` | Parse model configs, build computation graph structure |
| **Operator Layer** | `src/arch/op/` | Implement FLOPs calculation for Attention, FFN, MoE operators |
| **Hardware Modeling** | `src/hardware/` | Chip characteristic modeling, supports JSON/JSON5 configuration |
| **Performance Calc** | `src/arch/perf_calculator.py` | Comprehensive theoretical latency and throughput calculation |
| **Output Reports** | `src/visual/` | Console tables and Excel report generation |

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd llmsim

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### 1. DeepSeek V3 Prefill Performance Analysis

```bash
python3 src/main.py \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --max_seqlen 4096 \
    --batch_size 1 \
    --mode extend \
    --tp_size 4 \
    --dp_size 4 \
    --ep_size 16 \
    --enable_deepep \
    --enable_moe_dense_fully_dp \
    --hardware klx_p800
```

### 2. DeepSeek V3 Decode Performance Analysis

```bash
python3 src/main.py \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --max_seqlen 4096 \
    --batch_size 128 \
    --mode decode \
    --tp_size 4 \
    --dp_size 4 \
    --ep_size 16 \
    --enable_deepep \
    --hardware klx_p800
```

### 3. Qwen3 Model Analysis

```bash
python3 src/main.py \
    --model_path hf_config/qwen3-32B_config.json \
    --max_seqlen 8192 \
    --batch_size 16 \
    --mode extend \
    --tp_size 8 \
    --dp_size 2 \
    --hardware h800
```


### 4. Qwen3 MoE Model Analysis

```bash
python3 src/main.py \
    --model_path hf_config/qwen3-235B-A22B_config.json \
    --max_seqlen 4096 \
    --batch_size 1 \
    --mode extend \
    --tp_size 4 \
    --dp_size 4 \
    --ep_size 16 \
    --enable_deepep \
    --hardware h800
```


---

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | string | **Required** | Path to model configuration file |
| `--batch_size` | int | **Required** | Batch size |
| `--max_seqlen` | int | 4096 | Maximum sequence length |
| `--mode` | string | extend | Forward mode: `extend` (Prefill) / `decode` |
| `--tp_size` | int | 4 | Tensor Parallel size |
| `--dp_size` | int | 4 | Data Parallel size |
| `--ep_size` | int | 16 | Expert Parallel size |
| `--hardware` | string | default | Hardware preset: `default`, `h20`, `h800`, `gb200`, `klx_p800`, `custom` |
| `--hardware_config` | string | None | Custom hardware configuration file path |
| `--enable_mtp` | flag | False | Enable Multi-Token Prediction |
| `--enable_deepep` | flag | False | Enable Deep Expert Parallel |
| `--enable_moe_dense_fully_dp` | flag | False | MoE dense layers fully data parallel |
| `--output_format` | string | console | Output format: `console` / `excel` |
| `--output_file` | string | None | Excel output file path |

---

## Supported Models

| Model | Config File | Architecture Features |
|-------|-------------|----------------------|
| DeepSeek-V3/R1 671B | `deepseek_671b_r1_config.json` | MLA + MoE, 256 experts |
| DeepSeek-V3.2 | `deepseek_v3.2_config.json` | MLA + MoE |
| Qwen3-235B-A22B | `qwen3-235B-A22B_config.json` | MHA + MoE |
| Qwen3-Next-80B-A3B | `qwen3-next-80B-A3B_config.json` | MHA + MoE |
| Qwen3-32B | `qwen3-32B_config.json` | Dense model |
| Qwen3-8B | `qwen3-8B_config.json` | Dense model |

---

## Supported Hardware

| Hardware | Config Name | Features |
|----------|-------------|----------|
| Default GPU | `default` | General configuration |
| NVIDIA H20 | `h20` | HBM3, high bandwidth |
| NVIDIA H800 | `h800` | China-specific H100 variant |
| NVIDIA GB200 | `gb200` | Grace-Blackwell superchip |
| Kunlunxin P800 | `klx_p800` | Domestic AI chip |
| Custom | `custom` | Specify via `--hardware_config` |

---

## Hardware Configuration Fields

Hardware configuration files support JSON/JSON5 format. Field descriptions:

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `device_type` | string | - | Device type (`gpu`/`xpu`/`accelerator`) |
| `name` | string | - | Hardware name |
| `memory.hbm_size_gb` | int | GB | HBM memory size |
| `memory.cache_line_size` | int | Bytes | Cache line size |
| `bandwidth.hbm_bandwidth_gb_s` | float | TB/s | HBM bandwidth |
| `bandwidth.dma_bandwidth_gb_s` | float | GB/s | DMA bandwidth (extend mode) |
| `bandwidth.dma_bandwidth_decode_gb_s` | float | GB/s | DMA bandwidth (decode mode) |
| `bandwidth.link_bandwidth_gb_s` | float | GB/s | Intra-node bandwidth (NVLink) |
| `bandwidth.rdma_bandwidth_gb_s` | float | GB/s | Inter-node bandwidth (RDMA) |
| `compute.mac_int8_gflops` | float | TFLOPS | INT8 MAC performance |
| `compute.mac_fp32_gflops` | float | TFLOPS | FP32 MAC performance |
| `compute.mac_bf16_gflops` | float | TFLOPS | BF16 MAC performance |

---

## Output Examples

### Console Output

```bash
python3 src/main.py \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --batch_size 1 --max_seqlen 4096 --mode extend \
    --tp_size 4 --dp_size 4 --ep_size 16 \
    --enable_deepep --hardware klx_p800 \
    --output_format console
```

### Excel Output

```bash
python3 src/main.py \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --batch_size 1 --max_seqlen 4096 --mode extend \
    --tp_size 4 --dp_size 4 --ep_size 16 \
    --enable_deepep --hardware klx_p800 \
    --output_format excel \
    --output_file prefill_result.xlsx
```

---

## Custom Hardware Configuration

Create a custom hardware configuration file `my_gpu.json5`:

```json5
{
  device_type: "gpu",
  name: "My Custom GPU",
  memory: {
    hbm_size_gb: 80,
    cache_line_size: 128,
  },
  bandwidth: {
    hbm_bandwidth_gb_s: 2.0,
    dma_bandwidth_gb_s: 100.0,
    link_bandwidth_gb_s: 100.0,
    rdma_bandwidth_gb_s: 50.0,
  },
  compute: {
    mac_int8_gflops: 1000.0,
    mac_fp32_gflops: 250.0,
    mac_bf16_gflops: 500.0,
  },
}
```

Use custom configuration:

```bash
python3 src/main.py \
    --model_path hf_config/qwen3-32B_config.json \
    --hardware custom \
    --hardware_config my_gpu.json5 \
    ...
```

---

## Parameter Optimization

LLMSim now includes an **automatic parameter optimization module** that searches for the best TP (Tensor Parallel), DP (Data Parallel), EP (Expert Parallel), and batch size configurations to optimize TTFT (Time To First Token) and TPS (Tokens Per Second).

### Quick Start with Optimization

#### 1. Quick Recommendation Mode

Get recommended configuration based on priority (latency/throughput/balanced):

```bash
# Optimize for throughput
python -m src.optimization.cli \
    --model_path hf_config/qwen3-32B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --recommend throughput

# Optimize for low latency
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --recommend latency

# Balanced optimization
python -m src.optimization.cli \
    --model_path hf_config/qwen3-32B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --recommend balanced
```

#### 2. Full Grid Search Optimization

Search across custom parameter ranges:

```bash
python -m src.optimization.cli \
    --model_path hf_config/qwen3-32B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --tp_range "1,2,4,8" \
    --dp_range "1,2,4,8" \
    --batch_range "1-128" \
    --objective maximize_tps \
    --output optimization_result.json
```

#### 3. Optimization with World Size Constraint

When total GPU count is fixed (constrains TP * DP):

```bash
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --tp_range "1,2,4,8" \
    --dp_range "1,2,4,8" \
    --ep_range "1,2,4,8,16" \
    --batch_range "1-32" \
    --world_size 16 \
    --objective maximize_tps
```

### Optimization CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | string | **Required** | Path to model configuration file |
| `--hardware` | string | h800 | Hardware preset: `h20`, `h800`, `gb200`, `klx_p800` |
| `--max_seqlen` | int | **Required** | Maximum sequence length (fixed parameter) |
| `--mode` | string | extend | Forward mode: `extend` (Prefill) / `decode` |
| `--tp_range` | string | "1,2,4,8" | Tensor Parallel range (e.g., "1,2,4,8" or "1-8") |
| `--dp_range` | string | "1,2,4,8" | Data Parallel range |
| `--ep_range` | string | Auto | Expert Parallel range (auto-detected for MoE) |
| `--batch_range` | string | "1-128" | Batch size range |
| `--world_size` | int | None | Total GPU count (constrains TP * DP) |
| `--objective` | string | maximize_tps | Optimization goal: `minimize_ttft`, `maximize_tps`, `balanced` |
| `--optimizer` | string | grid_search | Search algorithm (currently only grid_search) |
| `--max_evaluations` | int | None | Limit number of evaluations |
| `--recommend` | string | None | Quick mode: `latency`, `throughput`, `balanced` |
| `--output` | string | None | Output file path (JSON format) |
| `--verbose` | flag | False | Verbose output |

### Output Example

```
============================================================
OPTIMIZATION RESULTS
============================================================

Best Configuration:
  Tensor Parallel (TP): 8
  Data Parallel (DP): 1
  Expert Parallel (EP): 1
  Batch Size: 128
  Mode: EXTEND

Performance Metrics:
  ttft_ms: 107.47
  throughput_tps: 4878268.83
  throughput_per_gpu: 609783.60
  total_time_ms: 3624.74

Optimization Statistics:
  Total evaluations: 80
  Search space size: 80
  Total time: 0.05s
```

---

## Project Structure

```
llmsim/
├── src/
│   ├── main.py                 # Entry point
│   ├── arch/                   # Architecture layer
│   │   ├── config.py           # Model/schedule configuration
│   │   ├── model_type.py       # Model type definitions
│   │   ├── perf_calculator.py  # Performance calculator
│   │   ├── models_arch/        # Model architecture implementations
│   │   ├── op/                 # Operator implementations
│   │   └── perf/               # Performance models
│   ├── hardware/               # Hardware modeling
│   │   └── hardware_config.py
│   ├── optimization/           # Parameter optimization module
│   │   ├── cli.py              # Optimization CLI
│   │   ├── service.py          # Optimization service
│   │   ├── search_space.py     # Search space definition
│   │   ├── objective.py        # Objective functions
│   │   ├── evaluator.py        # Performance evaluator
│   │   ├── constraints.py      # Constraint validation
│   │   ├── config.py           # Search space config
│   │   ├── results.py          # Result data structures
│   │   └── optimizers/         # Optimization algorithms
│   │       ├── base.py
│   │       └── grid_search.py
│   └── visual/                 # Output reports
│       ├── console_report.py
│       └── excel_report.py
├── hf_config/                  # Pre-configured model configs
├── hardware_config/            # Pre-configured hardware configs
├── pyproject.toml              # Project configuration
├── README.md                   # This document
├── ds_prefill.sh               # DeepSeek prefill examples
├── ds_decode.sh                # DeepSeek decode examples
├── optimize_ds_prefill.sh      # DeepSeek prefill optimization
├── optimize_ds_decode.sh       # DeepSeek decode optimization
└── optimize_qwen3.sh           # Qwen3 optimization examples
```

---

## Contributing

Issues and PRs are welcome! Before contributing code, please ensure:

1. Code follows project conventions
2. All tests pass
3. Relevant documentation is updated

---

## License

MIT License
