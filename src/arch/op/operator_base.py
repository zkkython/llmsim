"""
算子抽象层 - 提供通用的算子定义和接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class DataType(Enum):
    """数据类型"""
    INT8 = 1
    FP16 = 2
    BF16 = 2
    FP32 = 4
    FP64 = 8


@dataclass
class Tensor:
    """张量定义"""
    m: int = 0
    n: int = 0
    
    def size(self) -> int:
        """计算张量元素数量"""
        return self.m * self.n


@dataclass
class OperatorIO:
    """算子输入输出定义"""
    input_shape: Tensor = field(default_factory=Tensor)
    output_shape: Tensor = field(default_factory=Tensor)
    weight_shape: Tensor = field(default_factory=Tensor)
    
    input_dtype: DataType = DataType.BF16
    output_dtype: DataType = DataType.BF16
    weight_dtype: DataType = DataType.BF16


@dataclass
class OperatorMetadata:
    """算子元数据"""
    name: str = ""
    op_type: str = ""  # 算子类型: matmul, conv, etc
    description: str = ""
    
    io_config: OperatorIO = field(default_factory=OperatorIO)
    
    # 执行参数
    batch_size: int = 1
    num_layers: int = 1
    
    # 并行化参数
    parallelization_dim: Optional[str] = None  # 并行维度标记


class BaseOperator(ABC):
    """基础算子抽象类"""
    
    def __init__(self, metadata: OperatorMetadata):
        """
        初始化算子
        
        Args:
            metadata: 算子元数据
        """
        self.metadata = metadata
    
    @abstractmethod
    def get_compute_complexity(self) -> float:
        """
        获取计算复杂度 (FLOPs)
        
        Returns:
            FLOPs数量
        """
        pass
    
    @abstractmethod
    def get_memory_requirement(self) -> Dict[str, int]:
        """
        获取内存需求
        
        Returns:
            内存需求字典 {
                'input': 输入内存大小(字节),
                'output': 输出内存大小(字节),
                'weight': 权重内存大小(字节)
            }
        """
        pass
    
    def get_io_volume(self) -> Dict[str, int]:
        """
        获取 I/O 数据量
        
        Returns:
            I/O 数据量字典 {
                'load': 加载数据量(字节),
                'store': 存储数据量(字节)
            }
        """
        io = self.metadata.io_config
        input_size = io.input_shape.size() * self.metadata.batch_size * io.input_dtype.value
        output_size = io.output_shape.size() * self.metadata.batch_size * io.output_dtype.value
        weight_size = io.weight_shape.size() * self.metadata.batch_size * io.weight_dtype.value
        
        return {
            'load': input_size + weight_size,
            'store': output_size,
        }