# Config-Driven Auto Adapter 设计文档


---

## 背景与目标

原始 `auto_generator.py` 的实现存在以下问题：
1. **代码生成方式难以维护**：生成 Python 代码字符串，类型安全差、调试困难
2. **紧耦合的 Shape 提取逻辑**：硬编码的 `LAYER_PATTERNS`，不易扩展
3. **依赖 SGLang 运行时**：需要实例化 PyTorch 模型，导致复杂的 mock 需求

本重构实现了**纯 Config-driven 架构**，直接从 HuggingFace Config 构建 IR，无需 SGLang 模型实例化。

---

## 架构设计

### 核心分层

```
┌─────────────────────────────────────────────────────────────────┐
│  HuggingFace Config                                             │
│  - Config 对象或字典                                             │
│  - 包含所有模型架构参数                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │ Config-driven Builder
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Model IR (中间表示层)                                           │
│  - ComputationalGraph: 有序的 OpNode 列表                        │
│  - OpNode: 算子类型 + 形状 + 参数 + 并行策略                     │
│  - 与框架无关，可序列化、可分析                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ Transformation
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLMSim ModelArch (直接实例化)                                   │
│  - 无需代码生成，内存中直接构建                                   │
│  - 动态创建 OperatorMetadata 并注册到 Arch                       │
└─────────────────────────────────────────────────────────────────┘
```

### 关键优势

1. **无需 SGLang 依赖**：直接从 HF Config 构建，无需 PyTorch 模型
2. **无需 Mock**：不依赖分布式环境初始化
3. **更快**：避免模型实例化开销
4. **更稳定**：不受 SGLang 内部 API 变化影响
5. **可扩展性强**：通过注册表添加新模型类型
6. **易于调试**：IR 层可独立检查、可视化、验证

---

## 文件结构

```
src/arch/models_arch/
├── base_model_arch.py          # 保持不变
├── model_arch.py               # 保持不变
└── auto/
    ├── __init__.py             # 导出: auto_adapter_from_config
    ├── adapter.py              # 主入口: auto_adapter_from_config()
    ├── config_builder.py       # Config-driven IR 构建器
    ├── ir.py                   # IR 定义: ComputationalGraph, OpNode
    └── transformer.py          # IR → ModelArch 转换器
```

---

## 核心实现细节

### 1. Config Builder 注册表 (`config_builder.py`)

```python
# 注册表存储所有模型类型的构建器
_CONFIG_BUILDERS: Dict[str, Callable[[Any], ComputationalGraph]] = {}

def register_config_builder(model_type: str):
    """装饰器：注册模型类型的 IR 构建器"""
    def decorator(builder_fn):
        _CONFIG_BUILDERS[model_type] = builder_fn
        return builder_fn
    return decorator

# 使用示例
@register_config_builder("qwen3_moe")
def build_qwen3_moe_ir(config: Any) -> ComputationalGraph:
    """从 Qwen3 MoE Config 构建 IR"""
    graph = ComputationalGraph(
        model_name="Qwen3MoeForCausalLM",
        model_type="moe",
        config=_config_to_dict(config),
        has_moe=True,
        ...
    )
    # 添加算子节点
    graph.add_node(OpNode(...))
    return graph
```

### 2. 模型类型推断

```python
def infer_model_type(config: Any) -> Optional[str]:
    """从 Config 推断模型类型"""
    model_type = _get_config_value(config, "model_type", "")

    # 检查注册表
    if model_type in _CONFIG_BUILDERS:
        return model_type

    # 从 architectures 字段推断
    architectures = _get_config_value(config, "architectures", [])
    if architectures:
        arch = architectures[0].lower()
        if "qwen3moe" in arch:
            return "qwen3_moe"
        elif "deepseekv3" in arch:
            return "deepseek_v3"
        elif "qwen3" in arch:
            return "qwen3"

    return None
```

### 3. IR 层 (`ir.py`)

保持不变，支持：
- `OpNode`: 算子节点（名称、类型、形状、并行策略等）
- `ComputationalGraph`: 计算图（节点列表、模型元数据）
- `ShapeSpec`: 形状规范（支持符号表达式）
- `ParallelStrategy`: 并行策略（TP/EP）
- 序列化/反序列化 (`to_dict()`, `from_dict()`)

### 4. 主适配器 (`adapter.py`)

简化后的单一入口：

```python
def auto_adapter_from_config(
    config: Any,
    schedule_config: ScheduleConfig,
    model_type: Optional[str] = None,
) -> BaseModelArch:
    """
    从 HuggingFace Config 构建 ModelArch。

    这是唯一的公开 API，无需 SGLang 模型类。
    """
    # 1. 构建 IR
    ir_graph = build_ir_from_config(config, model_type)

    # 2. 转换为 ModelArch
    transformer = IRToModelArchTransformer(ir_graph, schedule_config)
    return transformer.transform()
```

### 5. IR 到 ModelArch 转换器 (`transformer.py`)

将 IR 节点分组并构建对应算子：

```python
class IRToModelArchTransformer:
    def transform(self) -> BaseModelArch:
        # 创建 ModelConfig
        model_config = self._create_model_config()

        # 创建 ModelArch
        arch = create_model_arch(model_config, self.schedule_config)

        # 构建算子
        self._build_operators(arch)

        return arch

    def _build_operators(self, arch: BaseModelArch):
        """从 IR 节点构建算子"""
        # 按类型分组
        attention_nodes = []
        moe_nodes = []
        ...

        for node in self.ir_graph.nodes:
            if node.op_type == "matmul":
                if node.extra_attrs.get("is_attention"):
                    attention_nodes.append(node)
                elif node.extra_attrs.get("is_moe"):
                    moe_nodes.append(node)

        # 构建各类算子
        self._build_attention_operators(arch, attention_nodes)
        self._build_moe_operators(arch, moe_nodes)
```

---

## 扩展机制

### 添加新模型类型支持

```python
from src.arch.models_arch.auto import register_config_builder
from src.arch.models_arch.auto.ir import (
    ComputationalGraph, OpNode, ShapeSpec, DataType, ParallelStrategy
)

@register_config_builder("my_model")
def build_my_model_ir(config: Any) -> ComputationalGraph:
    """
    为新模型类型创建 IR 构建器。

    只需实现此函数，无需修改其他代码。
    """
    hidden_size = _get_config_value(config, "hidden_size")
    num_layers = _get_config_value(config, "num_hidden_layers")

    graph = ComputationalGraph(
        model_name="MyModel",
        model_type="dense",  # 或 "moe", "mla"
        config=_config_to_dict(config),
        has_moe=False,
        has_mla=False,
    )

    # 添加 Attention 投影
    graph.add_node(OpNode(
        name="qkv_proj",
        op_type="matmul",
        input_shape=ShapeSpec("seq_len", hidden_size),
        output_shape=ShapeSpec("seq_len", "qkv_out_dim"),
        weight_shape=ShapeSpec(hidden_size, "qkv_out_dim"),
        dtype=DataType.BF16,
        parallel_strategy=ParallelStrategy(tp_dim=1),
        num_layers=num_layers,
        extra_attrs={"is_attention": True},
    ))

    # 添加更多算子...

    return graph
```

---

## 使用示例

### 基本用法（推荐）

```python
from transformers import AutoConfig
from src.arch.config import ScheduleConfig, ForwardMode
from src.arch.models_arch.auto import auto_adapter_from_config

# 1. 从 HuggingFace 加载配置
config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")

# 2. 创建调度配置
schedule_config = ScheduleConfig(
    mode=ForwardMode.EXTEND,
    tp_size=4,
    dp_size=2,
    ep_size=8,
)

# 3. 一键适配（无需 SGLang 模型类！）
model_arch = auto_adapter_from_config(config, schedule_config)

# 4. 直接使用进行性能计算
from src.arch.perf_calculator import PerformanceCalculator
from src.hardware import hardware_config

calculator = PerformanceCalculator(hardware_config.DEFAULT_HARDWARE)
result = calculator.calculate_model_performance(model_arch)
```

### 指定模型类型

```python
# 当 auto-infer 失败时，可手动指定
model_arch = auto_adapter_from_config(
    config,
    schedule_config,
    model_type="qwen3_moe"
)
```

### IR 序列化（用于缓存/调试）

```python
from src.arch.models_arch.auto.config_builder import build_ir_from_config
import json

# 构建 IR
ir_graph = build_ir_from_config(config)

# 保存到文件
with open("model_ir.json", "w") as f:
    json.dump(ir_graph.to_dict(), f, indent=2)

# 从文件加载
from src.arch.models_arch.auto.ir import ComputationalGraph
with open("model_ir.json", "r") as f:
    data = json.load(f)
restored_graph = ComputationalGraph.from_dict(data)
```

---

## 测试

测试文件：
- `tests/test_auto_adapter.py` - IR 层测试（19 个测试）
- `tests/test_config_builder.py` - Config Builder 测试（22 个测试）

运行测试：
```bash
pytest tests/test_auto_adapter.py tests/test_config_builder.py -v
```

---

## 已注册构建器列表

当前共 **3 个** 已注册构建器：

| 模型类型 | 构建器函数 | 支持模型 |
|----------|-----------|----------|
| qwen3_moe | build_qwen3_moe_ir | Qwen3-30B-A3B, Qwen3-235B-A22B |
| deepseek_v3 | build_deepseek_v3_ir | DeepSeek-V3 |
| qwen3 | build_qwen3_ir | Qwen3 密集模型 |

---

## 与旧架构对比

| 特性 | 旧架构 | 新架构 |
|------|--------|--------|
| 输入 | SGLang Model Class + HF Config | 仅 HF Config |
| 依赖 | 需要 SGLang, PyTorch | 仅 transformers |
| Mock | 需要复杂的分布式 Mock | 无需 Mock |
| 性能 | 慢（需要模型实例化） | 快（纯 Python） |
| 稳定性 | 易受 SGLang 变化影响 | 稳定 |
| 扩展性 | 需添加 Layer Parser | 只需注册 Builder |

---

## 迁移指南

### 旧代码
```python
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from src.arch.models_arch.auto import auto_adapter

model_arch = auto_adapter(Qwen3MoeForCausalLM, config, schedule_config)
```

### 新代码
```python
from src.arch.models_arch.auto import auto_adapter_from_config

model_arch = auto_adapter_from_config(config, schedule_config)
```

只需删除 SGLang 导入，更改函数名即可。
