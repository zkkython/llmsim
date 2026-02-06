"""
示例：使用 Auto-Generator 从 SGLang 模型生成 model_arch

这个示例展示了如何使用 auto_generator 模块自动生成 model_arch 代码。

前置要求:
    pip install torch transformers sglang

Usage:
    python examples/generate_from_sglang.py --model Qwen/Qwen3-235B-A22B --output src/arch/models_arch/qwen3_moe_auto_arch.py
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_from_huggingface(model_name: str, output_path: str):
    """
    从 HuggingFace 模型配置生成 model_arch

    Args:
        model_name: HuggingFace 模型名称，如 "Qwen/Qwen3-235B-A22B"
        output_path: 输出文件路径
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers")
        return

    # 1. 加载配置
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # 2. 根据配置推断模型类
    model_class = infer_model_class(config)
    if model_class is None:
        print(f"Error: Cannot infer model class for {model_name}")
        print("Trying fallback: AST-based generation...")
        return generate_from_ast(model_name, output_path)

    # 3. 生成代码
    from src.arch.models_arch.auto_generator import generate_model_arch

    print(f"Generating model arch for {model_class.__name__}...")
    code = generate_model_arch(model_class, config, output_path)

    print(f"Successfully generated: {output_path}")
    return code


def infer_model_class(config):
    """
    根据配置推断对应的 SGLang 模型类

    映射规则基于 model_type 和 architectures
    """
    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", [])

    # 模型类型到 SGLang 模块的映射
    SGLANG_MODEL_MAP = {
        "qwen3_moe": ("sglang.srt.models.qwen3_moe", "Qwen3MoeForCausalLM"),
        "qwen3": ("sglang.srt.models.qwen3", "Qwen3ForCausalLM"),
        "qwen2_moe": ("sglang.srt.models.qwen2_moe", "Qwen2MoeForCausalLM"),
        "qwen2": ("sglang.srt.models.qwen2", "Qwen2ForCausalLM"),
        "deepseek_v3": ("sglang.srt.models.deepseek_v3", "DeepseekV3ForCausalLM"),
        "deepseek_v2": ("sglang.srt.models.deepseek_v2", "DeepseekV2ForCausalLM"),
        "llama": ("sglang.srt.models.llama", "LlamaForCausalLM"),
        "mistral": ("sglang.srt.models.llama", "LlamaForCausalLM"),
    }

    # 尝试从 architectures 推断
    for arch in architectures:
        arch_lower = arch.lower()
        if "qwen3moe" in arch_lower or "qwen3_moe" in arch_lower:
            model_type = "qwen3_moe"
            break
        elif "qwen3" in arch_lower:
            model_type = "qwen3"
            break
        elif "deepseek" in arch_lower and "v3" in arch_lower:
            model_type = "deepseek_v3"
            break
        elif "deepseek" in arch_lower and "v2" in arch_lower:
            model_type = "deepseek_v2"
            break

    if model_type not in SGLANG_MODEL_MAP:
        print(f"Warning: Unknown model type '{model_type}'")
        return None

    module_path, class_name = SGLANG_MODEL_MAP[model_type]

    try:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing {module_path}.{class_name}: {e}")
        return None


def generate_from_ast(model_name: str, output_path: str):
    """
    备用方案：从 HuggingFace 的 modeling 文件通过 AST 分析生成

    当 SGLang 模型类不可用时使用
    """
    print("AST-based generation not yet implemented")
    print("Please ensure sglang is installed: pip install sglang")
    return None


def generate_from_local_sglang_file(file_path: str, output_path: str):
    """
    从本地 SGLang 模型文件生成 model_arch

    Args:
        file_path: SGLang 模型文件路径
        output_path: 输出文件路径
    """
    # 1. 动态加载模型文件
    import importlib.util

    spec = importlib.util.spec_from_file_location("sglang_model", file_path)
    if spec is None or spec.loader is None:
        print(f"Error: Cannot load module from {file_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 2. 查找模型类和配置类
    model_class = None
    config_class_name = None

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            # 查找模型类（通常继承自 nn.Module 且有 forward 方法）
            if hasattr(obj, "forward") and "model" in name.lower():
                model_class = obj
            # 查找 Config 类
            if "config" in name.lower() and name.endswith("Config"):
                config_class_name = name

    if model_class is None:
        print(f"Error: Cannot find model class in {file_path}")
        return None

    print(f"Found model class: {model_class.__name__}")

    # 3. 创建最小配置
    # 从文件中提取默认配置值
    config = extract_config_from_file(file_path)

    # 4. 生成代码
    from src.arch.models_arch.auto_generator import generate_model_arch

    code = generate_model_arch(model_class, config, output_path)
    return code


def extract_config_from_file(file_path: str):
    """
    从 SGLang 模型文件中提取配置默认值

    通过 AST 分析 __init__ 方法中的 config 属性访问
    """
    import ast

    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    # 创建一个简单的配置对象
    class SimpleConfig:
        def __init__(self):
            # 默认值
            self.hidden_size = 4096
            self.num_hidden_layers = 32
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
            self.intermediate_size = 11008
            self.vocab_size = 32000

        def __getattr__(self, name):
            # 返回默认值而不是报错
            return 0

    config = SimpleConfig()

    # AST 分析：查找 config 的属性访问
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "config":
                attr_name = node.attr
                # 这里可以进一步分析赋值语句来推断类型

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate model_arch.py from SGLang model"
    )
    parser.add_argument(
        "--model", type=str, help="HuggingFace model name (e.g., Qwen/Qwen3-235B-A22B)"
    )
    parser.add_argument("--file", type=str, help="Local SGLang model file path")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for generated model_arch.py",
    )
    parser.add_argument(
        "--config", type=str, help="Optional: JSON config file with model parameters"
    )

    args = parser.parse_args()

    if args.config:
        # 从 JSON 配置生成
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        print(f"Loading config from {args.config}")
        # TODO: 实现从 JSON 直接生成

    elif args.model:
        # 从 HuggingFace 生成
        generate_from_huggingface(args.model, args.output)

    elif args.file:
        # 从本地文件生成
        generate_from_local_sglang_file(args.file, args.output)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
