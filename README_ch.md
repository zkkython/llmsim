# LLMSim - LLM 推理性能分析工具

LLMSim 是一个面向大语言模型的**计算性能建模与硬件适配分析工具**，专注于量化不同模型架构（如 Qwen3、DeepSeek）在各类芯片上的理论计算量（FLOPs）与实际执行效率。

**核心价值**：为模型选型、硬件部署、推理优化提供数据驱动决策依据。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户接口层                                │
│                    (命令行 / CLI Arguments)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      核心计算引擎                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   模型架构解析   │  │   性能计算器    │  │   调度配置      │  │
│  │  (Model Arch)   │  │ (Perf Calculator)│  │ (Schedule)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│   算子层 (Ops)  │  │   硬件建模层     │  │   输出报告层    │
│  ├─ Attention  │  │  ├─ GPU/XPU     │  │  ├─ Console    │
│  ├─ FFN/MoE    │  │  ├─ 带宽建模    │  │  ├─ Excel      │
│  ├─ MatMul     │  │  ├─ 计算能力    │  │  └─ Report     │
│  └─ Transfer   │  │  └─ 显存容量    │  │                │
└────────────────┘  └─────────────────┘  └─────────────────┘
```

### 模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| **模型架构** | `src/arch/models_arch/` | 解析模型配置，构建计算图结构 |
| **算子层** | `src/arch/op/` | 实现 Attention、FFN、MoE 等算子的 FLOPs 计算 |
| **硬件建模** | `src/hardware/` | 芯片特性建模，支持 JSON/JSON5 配置 |
| **性能计算** | `src/arch/perf_calculator.py` | 综合计算理论耗时与吞吐量 |
| **输出报告** | `src/visual/` | Console 表格与 Excel 报表生成 |

---

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd llmsim

# 安装依赖
pip install -e .

# 或安装开发依赖
pip install -e ".[dev]"
```

---

## 快速开始

### 1. DeepSeek V3 Prefill 性能分析

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

### 2. DeepSeek V3 Decode 性能分析

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

### 3. Qwen3 模型分析

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

---

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | string | **必填** | 模型配置文件路径 |
| `--batch_size` | int | **必填** | 批次大小 |
| `--max_seqlen` | int | 4096 | 最大序列长度 |
| `--mode` | string | extend | 前向模式: `extend`(Prefill) / `decode` |
| `--tp_size` | int | 4 | 张量并行度 (Tensor Parallel) |
| `--dp_size` | int | 4 | 数据并行度 (Data Parallel) |
| `--ep_size` | int | 16 | 专家并行度 (Expert Parallel) |
| `--hardware` | string | default | 硬件预设: `default`, `h20`, `h800`, `gb200`, `klx_p800`, `custom` |
| `--hardware_config` | string | None | 自定义硬件配置文件路径 |
| `--enable_mtp` | flag | False | 启用多 Token 预测 |
| `--enable_deepep` | flag | False | 启用深度专家并行 |
| `--enable_moe_dense_fully_dp` | flag | False | MoE 密集层完全数据并行 |
| `--output_format` | string | console | 输出格式: `console` / `excel` |
| `--output_file` | string | None | Excel 输出文件路径 |

---

## 支持的模型

| 模型 | 配置文件 | 架构特点 |
|------|----------|----------|
| DeepSeek-V3/R1 671B | `deepseek_671b_r1_config.json` | MLA + MoE, 256 专家 |
| DeepSeek-V3.2 | `deepseek_v3.2_config.json` | MLA + MoE |
| Qwen3-235B-A22B | `qwen3-235B-A22B_config.json` | MHA + MoE |
| Qwen3-Next-80B-A3B | `qwen3-next-80B-A3B_config.json` | MHA + MoE |
| Qwen3-32B | `qwen3-32B_config.json` | Dense 模型 |
| Qwen3-8B | `qwen3-8B_config.json` | Dense 模型 |

---

## 支持的硬件

| 硬件 | 配置名称 | 特点 |
|------|----------|------|
| 默认 GPU | `default` | 通用配置 |
| NVIDIA H20 | `h20` | HBM3, 高带宽 |
| NVIDIA H800 | `h800` | 中国特供版 H100 |
| NVIDIA GB200 | `gb200` | Grace-Blackwell 超级芯片 |
| 昆仑芯 P800 | `klx_p800` | 国产 AI 芯片 |
| 自定义 | `custom` | 通过 `--hardware_config` 指定 |

---

## 硬件配置字段说明

硬件配置文件支持 JSON/JSON5 格式，字段说明如下：

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `device_type` | string | - | 设备类型 (`gpu`/`xpu`/`accelerator`) |
| `name` | string | - | 硬件名称 |
| `memory.hbm_size_gb` | int | GB | HBM 显存大小 |
| `memory.cache_line_size` | int | Bytes | Cache 行大小 |
| `bandwidth.hbm_bandwidth_gb_s` | float | TB/s | HBM 带宽 |
| `bandwidth.dma_bandwidth_gb_s` | float | GB/s | DMA 带宽 (扩展模式) |
| `bandwidth.dma_bandwidth_decode_gb_s` | float | GB/s | DMA 带宽 (解码模式) |
| `bandwidth.link_bandwidth_gb_s` | float | GB/s | 机内通信带宽 (NVLink) |
| `bandwidth.rdma_bandwidth_gb_s` | float | GB/s | 机间通信带宽 (RDMA) |
| `compute.mac_int8_gflops` | float | TFLOPS | INT8 MAC 性能 |
| `compute.mac_fp32_gflops` | float | TFLOPS | FP32 MAC 性能 |
| `compute.mac_bf16_gflops` | float | TFLOPS | BF16 MAC 性能 |

---

## 输出示例

### Console 输出

```bash
python3 src/main.py \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --batch_size 1 --max_seqlen 4096 --mode extend \
    --tp_size 4 --dp_size 4 --ep_size 16 \
    --enable_deepep --hardware klx_p800 \
    --output_format console
```

### Excel 输出

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

## 自定义硬件配置

创建自定义硬件配置文件 `my_gpu.json5`：

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

使用自定义配置：

```bash
python3 src/main.py \
    --model_path hf_config/qwen3-32B_config.json \
    --hardware custom \
    --hardware_config my_gpu.json5 \
    ...
```

---

## 项目结构

```
llmsim/
├── src/
│   ├── main.py                 # 入口程序
│   ├── arch/                   # 架构层
│   │   ├── config.py           # 模型/调度配置
│   │   ├── model_type.py       # 模型类型定义
│   │   ├── perf_calculator.py  # 性能计算器
│   │   ├── models_arch/        # 模型架构实现
│   │   ├── op/                 # 算子实现
│   │   └── perf/               # 性能模型
│   ├── hardware/               # 硬件建模
│   │   └── hardware_config.py
│   └── visual/                 # 输出报告
│       ├── console_report.py
│       └── excel_report.py
├── hf_config/                  # 预置模型配置
├── hardware_config/            # 预置硬件配置
├── pyproject.toml              # 项目配置
└── README.md                   # 本文档
```

---

## 贡献指南

欢迎提交 Issue 和 PR！在贡献代码前，请确保：

1. 代码符合项目规范
2. 通过所有测试
3. 更新相关文档

---

## License

MIT License
