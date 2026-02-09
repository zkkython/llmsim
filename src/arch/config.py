import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from src.arch.configs_remote_loader import RemoteConfigsLoader
from src.arch.model_type import AttentionType, ForwardMode


@dataclass
class ModelConfig:
    """Base class for model configuration"""

    model_type: str = ""
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 11008
    attention_type: AttentionType = AttentionType.MHA  # Default to MHA

    # Load from JSON configuration file
    @staticmethod
    def from_json(config_path: str) -> "ModelConfig":
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            # when the file not exists, try to download from hugging-face
            #  --model_path zai-org/GLM-4.7-Flash
            # and model_path is mandatory to download from which site, like:
            #  --model_path huggingface.co/zai-org/GLM-4.7-Flash
            #  --model_path modelscope.cn/zai-org/GLM-4.7-Flash
            config_path = RemoteConfigsLoader.load_configs_from_remote(
                config_path
            )
        if not os.path.exists(config_path):
            # when the file still not exists, raise an error
            raise RuntimeError(f"Model config not found: {config_path}")

        with open(config_path, "r") as f:
            data = json.load(f)

        # Select appropriate config class based on model_type
        model_type = data.get("model_type", "")

        if model_type == "deepseek_v3":
            return DeepSeekV3Config.from_dict(data)
        elif model_type == "qwen3":
            return Qwen3Config.from_dict(data)
        elif model_type == "qwen3_moe":
            return Qwen3MoEConfig.from_dict(data)
        else:
            return ModelConfig.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary"""
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
    """DeepSeek V3 model configuration"""

    model_type: str = "deepseek_v3"

    # Specific parameters
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
        """Create configuration from dictionary"""
        config = DeepSeekV3Config()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        return config


@dataclass
class Qwen3Config(ModelConfig):
    """Qwen3 model configuration"""

    model_type: str = "qwen3"

    # Specific parameters
    head_dim: int = 128
    attention_type: AttentionType = AttentionType.MHA

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Qwen3Config":
        """Create configuration from dictionary"""
        config = Qwen3Config()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        return config


@dataclass
class Qwen3MoEConfig(Qwen3Config):
    """Qwen3 MoE configuration"""

    model_type: str = "qwen3_moe"

    # Specific parameters
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Qwen3MoEConfig":
        config = Qwen3MoEConfig()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        return config


@dataclass
class ScheduleConfig:
    """Scheduling configuration"""

    batch_size: int = 64
    max_seqlen: int = 4096
    mode: ForwardMode = ForwardMode.EXTEND

    num_nodes: int = 2
    world_size: int = 16

    # Parallelism configuration
    tp_size: int = 1  # Tensor Parallel
    dp_size: int = 16  # Data Parallel
    ep_size: int = 16  # Expert Parallel

    # Special feature switches
    is_mtp: bool = True  # Multi-Token Prediction
    deepep: bool = True  # Deep Expert Parallel
    enable_moe_dense_fully_dp: bool = False
