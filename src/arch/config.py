import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from src.arch.model_type import AttentionType, ForwardMode
from src.llmsim.lib.configs import huggingface_configs_loader


@dataclass
class ModelConfig:
    """模型配置基类"""

    model_type: str = ""
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 11008
    attention_type: AttentionType = AttentionType.MHA  # 默认使用 MHA

    # 从 JSON 配置文件加载
    @staticmethod
    def from_json(config_path: str) -> "ModelConfig":
        """从 JSON 文件加载配置"""
        if not os.path.exists(config_path):
            # 如果文件不存在的时候，尝试自动从huggingface下载文件
            # 使用方法：
            #  --model_path zai-org/GLM-4.7-Flash
            config_path = huggingface_configs_loader.download_configs_from_hugging_face(config_path)
        if not os.path.exists(config_path):
            # 进行二次确认，文件不存在再报个错
            raise RuntimeError(f"Model config not found: {config_path}")


        with open(config_path, "r") as f:
            data = json.load(f)

        # 根据 model_type 选择合适的配置类
        model_type = data.get("model_type", "")

        if model_type == "deepseek_v3":
            return DeepSeekV3Config.from_dict(data)
        elif model_type == "qwen3":
            return Qwen3Config.from_dict(data)
        else:
            return ModelConfig.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置"""
        config = ModelConfig()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        config.model_type = data.get("model_type", "")
        return config


@dataclass
class DeepSeekV3Config(ModelConfig):
    """DeepSeek V3 模型配置"""

    model_type: str = "deepseek_v3"

    # 特定参数
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512

    first_k_dense_replace: int = 3
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    num_experts_per_tok: int = 8

    attention_type: AttentionType = AttentionType.MLA

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DeepSeekV3Config":
        """从字典创建配置"""
        config = DeepSeekV3Config()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        return config


@dataclass
class Qwen3Config(ModelConfig):
    """Qwen3 模型配置"""

    model_type: str = "qwen3"

    # 特定参数
    head_dim: int = 128
    attention_type: AttentionType = AttentionType.MHA

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Qwen3Config":
        """从字典创建配置"""
        config = Qwen3Config()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        return config


@dataclass
class ScheduleConfig:
    """调度配置"""

    batch_size: int = 64
    max_seqlen: int = 4096
    mode: ForwardMode = ForwardMode.EXTEND

    num_nodes: int = 2
    world_size: int = 16

    # 并行化配置
    tp_size: int = 1  # Tensor Parallel
    dp_size: int = 16  # Data Parallel
    ep_size: int = 16  # Expert Parallel

    # 特殊功能开关
    is_mtp: bool = True  # Multi-Token Prediction
    deepep: bool = True  # Deep Expert Parallel
    enable_moe_dense_fully_dp: bool = False
