# SGLang 自动适配重构设计文档


---

## 背景与目标

当前 `auto_generator.py` 的实现存在以下问题：
1. **代码生成方式难以维护**：生成 Python 代码字符串，类型安全差、调试困难
2. **紧耦合的 Shape 提取逻辑**：硬编码的 `LAYER_PATTERNS`，不易扩展
3. **缺乏中间抽象层**：直接从 SGLang 模型跳到代码生成，没有清晰的 IR 层

本计划目标是实现一个**分层 IR 架构**，使自动适配更加优雅、可扩展、易调试。

---

## 架构设计

### 核心分层

```
┌─────────────────────────────────────────────────────────────────┐
│  SGLang Model (PyTorch nn.Module)                               │
│  - Qwen3MoeForCausalLM / DeepseekV3ForCausalLM / LlamaForCausalLM│
└────────────────────────────┬────────────────────────────────────┘
                             │ Parsing
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

1. **无需代码生成**：直接在内存中构建 ModelArch，避免字符串拼接和文件IO
2. **可扩展性强**：通过插件机制注册新的 Layer Parser，符合开闭原则
3. **易于调试**：IR 层可独立检查、可视化、验证
4. **职责分离**：每层只关心自己的转换逻辑

---

## 文件结构

```
src/arch/models_arch/
├── base_model_arch.py          # 保持不变
├── model_arch.py               # 保持不变
├── auto/
│   ├── __init__.py             # 导出: auto_adapter, register_layer_parser
│   ├── adapter.py              # 主入口: SglangAutoAdapter
│   ├── ir.py                   # IR 定义: ComputationalGraph, OpNode
│   ├── parser.py               # SGLang 模型解析器
│   ├── transformer.py          # IR → ModelArch 转换器
│   └── layer_parsers/          # 层解析器插件
│       ├── __init__.py
│       ├── base.py             # BaseLayerParser 抽象类
│       ├── linear.py           # Linear 层解析 (QKV, Row, Column)
│       ├── attention.py        # Attention 层解析
│       ├── moe.py              # MoE 层解析
│       ├── norm.py             # Normalization 层解析
│       └── registry.py         # 解析器注册表
```

---

## 核心实现细节

### 1. IR 层 (`ir.py`)

```python
@dataclass
class OpNode:
    """计算图节点 - 与框架无关的中间表示"""
    name: str
    op_type: str           # matmul, attention, transfer, norm
    input_shape: ShapeSpec # 支持表达式如 "seq_len", "hidden_size"
    output_shape: ShapeSpec
    weight_shape: ShapeSpec
    dtype: DataType
    parallel_strategy: ParallelStrategy  # TP/EP 切分策略
    num_layers: Union[int, str]  # 层数或引用如 "num_layers"
    extra_attrs: Dict[str, Any] = field(default_factory=dict)
    attention_type: Optional[str] = None  # mha, gqa, mla

@dataclass
class ComputationalGraph:
    """完整计算图"""
    model_name: str
    model_type: str        # dense, moe, mla
    config: Dict[str, Any] # 原始配置引用
    nodes: List[OpNode]
    kv_cache_type: str     # mha, gqa, mla

    # 模型特性标志
    has_moe: bool = False
    has_mla: bool = False
    has_dense_layers: bool = False
    first_k_dense_replace: int = 0

    def get_nodes_by_type(self, op_type: str) -> List[OpNode]: ...
    def get_attention_nodes(self) -> List[OpNode]: ...
    def get_matmul_nodes(self) -> List[OpNode]: ...
    def get_transfer_nodes(self) -> List[OpNode]: ...
    def get_node_by_name(self, name: str) -> Optional[OpNode]: ...
    def validate(self) -> List[str]: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ComputationalGraph": ...

@dataclass
class ShapeSpec:
    """形状规范，支持具体值和符号表达式"""
    m: Union[int, str] = 0
    n: Union[int, str] = 0

    @property
    def is_symbolic(self) -> bool: ...
    def resolve(self, context: Dict[str, Any]) -> Tuple[int, int]: ...

@dataclass
class ParallelStrategy:
    """并行化策略"""
    tp_dim: Optional[int] = None  # 0=row-wise, 1=column-wise
    ep_size: int = 1
    replicated: bool = False
```

### 2. 解析器插件系统 (`layer_parsers/`)

```python
# base.py
class BaseLayerParser(ABC):
    """层解析器基类"""

    @property
    @abstractmethod
    def layer_types(self) -> List[str]:
        """返回支持的 PyTorch 层类型名列表"""
        pass

    @abstractmethod
    def parse(self, name: str, module: nn.Module, config: Any) -> Optional[OpNode]:
        """解析层并返回 OpNode，返回 None 表示跳过"""
        pass

    def can_parse(self, module: nn.Module) -> bool: ...

# registry.py
_layer_parsers: Dict[str, BaseLayerParser] = {}

def register_layer_parser(*layer_types: str):
    """装饰器：注册层解析器"""
    def decorator(parser_class: Type[BaseLayerParser]):
        parser = parser_class()
        for lt in layer_types:
            _layer_parsers[lt] = parser
        return parser_class
    return decorator

def get_parser(layer_type: str) -> Optional[BaseLayerParser]: ...
def list_registered_parsers() -> Dict[str, str]: ...
def unregister_parser(layer_type: str) -> bool: ...
```

### 3. 具体解析器实现

#### 3.1 Linear 层 (`layer_parsers/linear.py`)

```python
@register_layer_parser("QKVParallelLinear", "QKVSepParallelLinear")
class QKVLinearParser(BaseLayerParser):
    """解析 QKV 投影层"""

    @property
    def layer_types(self) -> List[str]:
        return ["QKVParallelLinear", "QKVSepParallelLinear"]

    def parse(self, name: str, module: nn.Module, config: Any) -> OpNode:
        # 处理 MLA 风格的分离 QKV
        if "QKVSepParallelLinear" in type(module).__name__:
            return self._parse_mla_qkv(name, module, config)

        return OpNode(
            name=name,
            op_type="matmul",
            input_shape=ShapeSpec("seq_len", "hidden_size"),
            output_shape=ShapeSpec(
                "seq_len", "(num_heads_per_rank + kv_heads_per_rank * 2) * head_dim"
            ),
            weight_shape=ShapeSpec(
                "hidden_size", "(num_heads_per_rank + kv_heads_per_rank * 2) * head_dim"
            ),
            dtype=DataType.BF16,
            parallel_strategy=ParallelStrategy(tp_dim=1),
            num_layers="num_layers",
            extra_attrs={"is_attention": True, "is_qkv": True},
        )

@register_layer_parser("RowParallelLinear")
class RowLinearParser(BaseLayerParser):
    """解析 RowParallelLinear (通常是 o_proj, down_proj)"""
    ...

@register_layer_parser("ColumnParallelLinear")
class ColumnLinearParser(BaseLayerParser):
    """解析 ColumnParallelLinear (通常是 gate_up_proj)"""
    ...

@register_layer_parser("ReplicatedLinear")
class ReplicatedLinearParser(BaseLayerParser):
    """解析 ReplicatedLinear (通常是 MoE gate)"""
    ...
```

#### 3.2 Attention 层 (`layer_parsers/attention.py`)

```python
@register_layer_parser("RadixAttention")
class RadixAttentionParser(BaseLayerParser):
    """解析 RadixAttention (标准 MHA/GQA)"""
    ...

@register_layer_parser("MLAAttention")
class MLAAttentionParser(BaseLayerParser):
    """解析 MLAAttention (DeepSeek 风格)"""
    ...

@register_layer_parser("LlamaAttention", "Qwen2Attention")
class StandardAttentionParser(BaseLayerParser):
    """解析标准 Attention (Llama, Qwen2, Qwen3)"""
    ...
```

#### 3.3 MoE 层 (`layer_parsers/moe.py`)

```python
@register_layer_parser("FusedMoE")
class FusedMoEParser(BaseLayerParser):
    """解析 FusedMoE"""
    ...

@register_layer_parser("Qwen3MoeSparseMoeBlock", "DeepseekV3MoE")
class MoEBlockParser(BaseLayerParser):
    """解析 MoE Block (复合层，返回 None 处理子层)"""
    ...

@register_layer_parser("MoEGate", "TopKGate")
class MoEGateParser(BaseLayerParser):
    """解析 MoE Gate"""
    ...
```

#### 3.4 Normalization 层 (`layer_parsers/norm.py`)

```python
@register_layer_parser("RMSNorm", "LayerNorm", "FusedRMSNorm")
class NormParser(BaseLayerParser):
    """解析 Normalization 层 (跳过，不生成算子)"""

    def parse(self, name: str, module: nn.Module, config: Any) -> Optional[OpNode]:
        return None  # 跳过
```

### 4. 主适配器 (`adapter.py`)

```python
class SglangAutoAdapter:
    """SGLang 模型自动适配器主入口"""

    def __init__(self, model_class: Type, config: Any):
        self.model_class = model_class
        self.config = config
        self._ir_graph: Optional[ComputationalGraph] = None
        self._model_arch: Optional[BaseModelArch] = None

    @property
    def ir_graph(self) -> Optional[ComputationalGraph]: ...

    @property
    def model_arch(self) -> Optional[BaseModelArch]: ...

    def parse(self) -> ComputationalGraph:
        """第一步：解析 SGLang 模型为 IR"""
        parser = ModelParser(self.config)
        self._ir_graph = parser.parse(self.model_class)
        return self._ir_graph

    def transform(self, schedule_config: ScheduleConfig) -> BaseModelArch:
        """第二步：将 IR 转换为 ModelArch"""
        if self._ir_graph is None:
            self.parse()

        transformer = IRToModelArchTransformer(self._ir_graph, schedule_config)
        self._model_arch = transformer.transform()
        return self._model_arch

    def adapt(self, schedule_config: ScheduleConfig) -> BaseModelArch:
        """一键适配：parse + transform"""
        return self.transform(schedule_config)

    def get_ir_summary(self) -> dict:
        """获取 IR 摘要用于调试"""
        ...


def auto_adapter(
    model_class: Type, config: Any, schedule_config: ScheduleConfig
) -> BaseModelArch:
    """便捷函数"""
    adapter = SglangAutoAdapter(model_class, config)
    return adapter.adapt(schedule_config)


def parse_model(model_class: Type, config: Any) -> ComputationalGraph:
    """仅解析到 IR，不转换为 ModelArch"""
    adapter = SglangAutoAdapter(model_class, config)
    return adapter.parse()
```

### 5. SGLang 模型解析器 (`parser.py`)

```python
class ModelParser:
    """解析 SGLang 模型并提取 ComputationalGraph IR"""

    SKIP_LAYER_TYPES: Set[str] = {
        "Embedding",
        "RotaryEmbedding",
        "LlamaRotaryEmbedding",
        "Qwen2RotaryEmbedding",
    }

    def __init__(self, config: Any):
        self.config = config
        self._unsupported_layers: List[Dict[str, str]] = []

    def parse(self, model_class: Type) -> ComputationalGraph:
        """解析模型类并返回 ComputationalGraph"""
        with mock_sglang_environment():
            model = self._instantiate_model(model_class)

        graph = self._create_graph(model_class)
        self._traverse_model(model, graph)
        return graph

    def _infer_model_type(self) -> str: ...
    def _has_moe(self) -> bool: ...
    def _has_mla(self) -> bool: ...
    def _infer_kv_cache_type(self) -> str: ...
    def _traverse_model(self, model: Any, graph: ComputationalGraph): ...
    def _parse_layer(self, name: str, module: nn.Module) -> Optional[OpNode]: ...


@contextmanager
def mock_sglang_environment():
    """模拟 SGLang 环境以绕过分布式依赖"""
    mock_modules = create_mock_environment()
    # 替换 sys.modules
    ...
    try:
        yield
    finally:
        # 恢复原始模块
        ...
```

### 6. IR 到 ModelArch 转换器 (`transformer.py`)

```python
class IRToModelArchTransformer:
    """将 IR 计算图转换为 LLMSim ModelArch"""

    def __init__(self, ir_graph: ComputationalGraph, schedule_config: ScheduleConfig):
        self.ir_graph = ir_graph
        self.schedule_config = schedule_config
        self._shape_context: Dict[str, Any] = {}

    def transform(self) -> BaseModelArch:
        """转换入口"""
        model_config = self._create_model_config()
        arch = create_model_arch(model_config, self.schedule_config)

        self._build_shape_context(model_config)
        self._build_operators(arch)

        return arch

    def _build_operators(self, arch: BaseModelArch):
        """从 IR 节点构建算子"""
        # 分组处理不同类型的节点
        attention_proj_nodes = []
        attention_core_nodes = []
        ffn_nodes = []
        moe_nodes = []
        transfer_nodes = []

        for node in self.ir_graph.nodes:
            if node.op_type == "matmul":
                if node.extra_attrs.get("is_attention"):
                    attention_proj_nodes.append(node)
                elif node.extra_attrs.get("is_moe"):
                    moe_nodes.append(node)
                else:
                    ffn_nodes.append(node)
            elif node.op_type == "attention":
                attention_core_nodes.append(node)
            elif node.op_type == "transfer":
                transfer_nodes.append(node)

        # 构建各类算子
        self._build_attention_operators(arch, attention_proj_nodes, attention_core_nodes)
        self._build_ffn_operators(arch, ffn_nodes)
        self._build_moe_operators(arch, moe_nodes)
        self._build_transfer_operators(arch)

    def _resolve_shape(self, shape_spec: ShapeSpec) -> tuple:
        """解析 ShapeSpec 到具体维度"""
        return shape_spec.resolve(self._shape_context)
```

---

## 扩展机制

### 添加新层类型支持

```python
# 1. 创建解析器
from src.arch.models_arch.auto.layer_parsers import register_layer_parser, BaseLayerParser

@register_layer_parser("FlashAttention", "PagedAttention")
class FlashAttentionParser(BaseLayerParser):
    @property
    def layer_types(self) -> List[str]:
        return ["FlashAttention", "PagedAttention"]

    def parse(self, name: str, module: nn.Module, config: Any) -> OpNode:
        return OpNode(
            name=name,
            op_type="attention",
            # ... 形状计算
        )

# 2. 自动注册，无需修改其他代码
```

---

## 使用示例

### 基本用法

```python
# 使用新的自动适配 API
from src.arch.models_arch.auto import auto_adapter
from src.arch.config import ScheduleConfig

# 1. 从 HuggingFace 加载配置
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")

# 2. 导入 SGLang 模型类
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM

# 3. 创建调度配置
schedule_config = ScheduleConfig(
    mode="extend",
    tp_size=4,
    dp_size=2,
    ep_size=8,
)

# 4. 一键适配
model_arch = auto_adapter(Qwen3MoeForCausalLM, config, schedule_config)

# 5. 直接使用进行性能计算
from src.arch.perf_calculator import PerformanceCalculator
calculator = PerformanceCalculator(hardware_config)
result = calculator.calculate_model_performance(model_arch)
```

### 分步用法（用于调试）

```python
from src.arch.models_arch.auto import SglangAutoAdapter

# 创建适配器
adapter = SglangAutoAdapter(Qwen3MoeForCausalLM, config)

# 第一步：解析到 IR（可检查中间结果）
ir_graph = adapter.parse()
print(f"Model has {len(ir_graph.nodes)} operators")
print(f"Model type: {ir_graph.model_type}")

# 获取 IR 摘要
summary = adapter.get_ir_summary()
print(f"Matmul nodes: {summary['matmul_nodes']}")
print(f"Attention nodes: {summary['attention_nodes']}")

# 第二步：转换为 ModelArch
model_arch = adapter.transform(schedule_config)
```

### IR 序列化（用于缓存/调试）

```python
from src.arch.models_arch.auto import parse_model
import json

# 解析到 IR
ir_graph = parse_model(Qwen3MoeForCausalLM, config)

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

测试文件位于 `tests/test_auto_adapter.py`，包含：

- **28 个通过测试**：覆盖 IR、解析器注册表、数据类型、Mock 环境、适配器 API
- **11 个跳过测试**：需要 PyTorch（ModelParser 相关）

运行测试：
```bash
pytest tests/test_auto_adapter.py -v
```

---

## 关键文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/arch/models_arch/auto/__init__.py` | ✅ 已实现 | 包入口 |
| `src/arch/models_arch/auto/ir.py` | ✅ 已实现 | IR 定义 |
| `src/arch/models_arch/auto/parser.py` | ✅ 已实现 | SGLang 模型解析 |
| `src/arch/models_arch/auto/transformer.py` | ✅ 已实现 | IR → ModelArch |
| `src/arch/models_arch/auto/adapter.py` | ✅ 已实现 | 主适配器 |
| `src/arch/models_arch/auto/layer_parsers/base.py` | ✅ 已实现 | 解析器基类 |
| `src/arch/models_arch/auto/layer_parsers/registry.py` | ✅ 已实现 | 解析器注册表 |
| `src/arch/models_arch/auto/layer_parsers/linear.py` | ✅ 已实现 | Linear 层解析 |
| `src/arch/models_arch/auto/layer_parsers/attention.py` | ✅ 已实现 | Attention 层解析 |
| `src/arch/models_arch/auto/layer_parsers/moe.py` | ✅ 已实现 | MoE 层解析 |
| `src/arch/models_arch/auto/layer_parsers/norm.py` | ✅ 已实现 | Normalization 层解析 |
| `src/arch/models_arch/auto_generator.py` | 📝 保留 | 标记 deprecated |
| `examples/generate_from_sglang.py` | 📝 保留 | 标记 deprecated |
| `tests/test_auto_adapter.py` | ✅ 已实现 | 单元测试 |

---

## 已注册解析器列表

当前共 **19 个** 已注册解析器：

| 层类型 | 解析器类 |
|--------|----------|
| QKVParallelLinear, QKVSepParallelLinear | QKVLinearParser |
| RowParallelLinear | RowLinearParser |
| ColumnParallelLinear | ColumnLinearParser |
| ReplicatedLinear | ReplicatedLinearParser |
| MergedColumnParallelLinear | MergedColumnLinearParser |
| RadixAttention | RadixAttentionParser |
| MLAAttention | MLAAttentionParser |
| LlamaAttention, Qwen2Attention | StandardAttentionParser |
| FusedMoE | FusedMoEParser |
| Qwen3MoeSparseMoeBlock | Qwen3MoEBlockParser |
| DeepseekV3MoE | DeepSeekV3MoEParser |
| MoEGate, TopKGate | MoEGateParser |
| Expert | ExpertParser |
| RMSNorm, LayerNorm, FusedRMSNorm | NormParser |
