# SGLang Model Architecture Auto-Generator 设计方案

## 概述

本方案提供一种**运行时动态分析**方法，自动从 SGLang 模型文件生成对应的 model_arch.py，解决模板方案覆盖不全的问题。

## 核心设计

### 1. 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│                    用户接口层 (CLI/API)                       │
│  generate_model_arch(model_class, config, output_path)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   模型结构提取器                              │
│  ModelStructureExtractor                                     │
│  ├── 识别模型类型 (MoE/Dense/MLA)                            │
│  ├── 遍历 nn.Module 结构                                     │
│  └── 生成 Transfer Operators                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   层类型识别器                                │
│  LayerTypeRecognizer                                         │
│  ├── 映射 SGLang 层 → Operator 类型                          │
│  ├── 提取形状信息 (动态表达式)                                │
│  └── 处理特殊结构 (MoE Block, MLA)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   代码生成器                                  │
│  ModelArchCodeGenerator                                      │
│  ├── 生成 import 语句                                        │
│  ├── 生成 build_operators 方法                               │
│  ├── 生成辅助方法 (_build_attention_operators, etc.)         │
│  └── 生成 KV Cache 方法                                      │
└─────────────────────────────────────────────────────────────┘
```

### 2. 关键技术点

#### 2.1 Mock 环境

SGLang 模型依赖分布式组件（TP/EP），直接实例化会失败。通过 Mock 机制绕过：

```python
@contextmanager
def mock_sglang_environment():
    """临时替换 SGLang 依赖为 Mock 对象"""
    mock_modules = create_mock_environment()
    # ... 替换 sys.modules
    yield
    # ... 恢复原模块
```

Mock 范围：
- `sglang.srt.distributed.*` - 分布式通信
- `sglang.srt.layers.communicator.*` - 层间通信
- `sglang.srt.server_args.*` - 服务参数
- `sgl_kernel.*` - CUDA 内核

#### 2.2 层类型映射

| SGLang 层类型 | Operator 类别 | 生成代码示例 |
|--------------|--------------|-------------|
| `QKVParallelLinear` | matmul | `qkv_proj` |
| `RowParallelLinear` | matmul | `o_proj` / `down_proj` |
| `ColumnParallelLinear` | matmul | `gate_up_proj` |
| `ReplicatedLinear` | matmul | `moe_gate` |
| `FusedMoE` | matmul (composite) | `moe_up` / `moe_down` |
| `RadixAttention` | attention | `qk`, `qkv` |
| `RMSNorm` | - | 不生成 operator |

#### 2.3 形状表达式

使用字符串表达式而非固定值，在生成的代码中动态计算：

```python
# 生成代码中的形状计算
num_heads_per_rank = mc.num_attention_heads // sc.tp_size
kv_heads_per_rank = max(1, mc.num_key_value_heads // sc.tp_size)

# OperatorMetadata 中的形状
input_shape=Tensor(seq_len, mc.hidden_size)
output_shape=Tensor(seq_len, (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim)
```

### 3. 处理流程

```python
def generate_model_arch(model_class, config, output_path):
    # 1. 提取模型结构
    extractor = ModelStructureExtractor(config, model_class)
    structure = extractor.extract()

    # 2. 识别模型特性
    #    - has_moe: 是否有 MoE 层
    #    - has_mla: 是否使用 MLA Attention
    #    - has_dense_layers: DeepSeek 风格的前 K 层 Dense
    #    - kv_cache_type: mha_gqa / mla

    # 3. 生成代码
    generator = ModelArchCodeGenerator(structure)
    code = generator.generate()

    # 4. 保存文件
    with open(output_path, 'w') as f:
        f.write(code)
```

### 4. 支持的模型类型

| 模型 | 特性 | 生成的辅助方法 |
|-----|------|--------------|
| Dense (Qwen3, Llama) | MHA/GQA + Dense FFN | `_build_attention_operators`, `_build_ffn_operators` |
| MoE (Qwen3-MoE) | MHA/GQA + MoE | `_build_attention_operators`, `_build_moe_operators`, `_build_deepep_operators` |
| DeepSeek V3 | MLA + Dense + MoE | `_build_attention_operators`, `_build_dense_operators`, `_build_moe_operators`, `_build_deepep_operators` |

### 5. 使用方式

#### 方式一：从 HuggingFace 模型生成

```python
from transformers import AutoConfig
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from src.arch.models_arch.auto_generator import generate_model_arch

config = AutoConfig.from_pretrained("Qwen/Qwen3-235B-A22B")
code = generate_model_arch(
    Qwen3MoeForCausalLM,
    config,
    "src/arch/models_arch/qwen3_moe_auto_arch.py"
)
```

#### 方式二：命令行工具

```bash
python examples/generate_from_sglang.py \
    --model Qwen/Qwen3-235B-A22B \
    --output src/arch/models_arch/qwen3_moe_auto_arch.py
```

#### 方式三：从本地 SGLang 文件生成

```bash
python examples/generate_from_sglang.py \
    --file /path/to/sglang_qwen3_moe.py \
    --output src/arch/models_arch/qwen3_moe_auto_arch.py
```

### 6. 扩展性设计

#### 6.1 添加新的层类型映射

在 `LayerTypeRecognizer.LAYER_PATTERNS` 中添加：

```python
LAYER_PATTERNS = {
    # ... 现有映射
    'NewLayerType': {
        'category': 'matmul',
        'op_name': 'new_op',
        'shape_extractor': '_extract_new_layer_shape',
    },
}
```

然后实现对应的形状提取方法：

```python
def _extract_new_layer_shape(self, name: str, module: Any) -> Dict:
    return {
        'input_shape': ('seq_len', 'mc.hidden_size'),
        'output_shape': ('seq_len', 'mc.hidden_size'),
        'weight_shape': ('mc.hidden_size', 'mc.hidden_size'),
        'dtype': 'BF16',
        'batch_size': 1,
    }
```

#### 6.2 支持新的模型架构

如果新架构有独特结构，可以继承 `ModelStructureExtractor`：

```python
class CustomModelExtractor(ModelStructureExtractor):
    def _identify_model_features(self, structure: ModelStructure):
        super()._identify_model_features(structure)
        # 添加自定义特征识别
        structure.has_custom_feature = hasattr(self.config, 'custom_attr')
```

### 7. 与模板方案的对比

| 维度 | 模板方案 | 运行时动态分析方案 (本方案) |
|-----|---------|------------------------|
| **覆盖度** | 低（需维护模板） | 高（自动识别所有层） |
| **维护成本** | 高（每个模型需模板） | 低（通用逻辑） |
| **准确性** | 中（静态值） | 高（运行时提取真实维度） |
| **适应性** | 差（模型更新需改模板） | 好（重新运行即可） |
| **依赖要求** | 无 | 需要 torch + sglang |
| **执行速度** | 快 | 较慢（需实例化模型） |

### 8. 注意事项

1. **依赖安装**：使用前需安装 `torch` 和 `sglang`
2. **Mock 完整性**：如果 SGLang 新增依赖，可能需要更新 mock 列表
3. **自定义层**：非标准层需要手动添加映射规则
4. **配置对齐**：确保 SGLang config 与项目 config 字段名一致

### 9. 文件结构

```
src/arch/models_arch/
├── auto_generator.py      # 核心生成器
├── base_model_arch.py     # 基类（已有）
├── model_arch.py          # 工厂函数（已有）
└── *_auto_arch.py         # 生成的文件

examples/
└── generate_from_sglang.py  # 使用示例
```

### 10. 后续优化方向

1. **AST 备用方案**：当运行时分析失败时，回退到 AST 静态分析
2. **缓存机制**：缓存已分析的模型结构，避免重复实例化
3. **增量更新**：检测模型变化，只更新变更部分
4. **可视化工具**：生成模型结构图，便于验证
