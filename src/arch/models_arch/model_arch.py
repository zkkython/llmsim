"""
模型架构抽象层 - 定义通用的模型架构接口和基类
"""
from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.config import ModelConfig, ScheduleConfig
from src.arch.models_arch.simple_model_arch import SimpleTransformerArch
from src.arch.models_arch.deepseek_v3_model_arch import DeepSeekV3Arch

def create_model_arch(model_config: ModelConfig, schedule_config: ScheduleConfig) -> BaseModelArch:
    """
    工厂函数 - 创建合适的模型架构
    
    Args:
        model_config: 模型配置
        schedule_config: 调度配置
        
    Returns:
        模型架构实例
    """
    model_type = model_config.model_type.lower()
    
    if model_type in ('deepseek_v3', 'deepseek_r1'):
        return DeepSeekV3Arch(model_config, schedule_config)
    elif model_type == 'qwen3_moe':
        raise NotImplementedError
    elif model_type == 'qwen3':
        return SimpleTransformerArch(model_config, schedule_config)
    else:
        # 默认使用标准 Transformer 架构
        return SimpleTransformerArch(model_config, schedule_config)