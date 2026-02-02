from enum import Enum
class AttentionType(Enum):
    """注意力类型"""
    MHA = "mha"  # Multi-Head Attention
    MLA = "mla"  # Multi-Head Latent Attention
    HYBRID = "hybrid"  # Hybrid Attention
    LINEAR = "linear"  # Linear Attention


class ForwardMode(Enum):
    """前向传递模式"""
    EXTEND = 0  # 序列扩展模式
    DECODE = 1  # 解码模式
