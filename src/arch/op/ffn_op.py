from src.arch.op.operator_base import BaseOperator, OperatorMetadata
from typing import Dict

class FFNOperator(BaseOperator):
    """前馈网络算子"""
    
    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)
    
    def get_compute_complexity(self) -> float:
        """计算 FFN 的 FLOPs"""
        io = self.metadata.io_config
        m = io.input_shape.m
        k = io.input_shape.n
        n = io.output_shape.n
        batch = self.metadata.batch_size
        
        # FFN: input * w1 + w1_out * w2
        # 通常 w1 是 gate+up projection，计算两次矩阵乘法
        return 2.0 * 2.0 * m * k * n * batch
    
    def get_memory_requirement(self) -> Dict[str, int]:
        """获取 FFN 的内存需求"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size
        
        input_mem = io.input_shape.size() * batch * io.input_dtype.value
        output_mem = io.output_shape.size() * batch * io.output_dtype.value
        weight_mem = io.weight_shape.size() * io.weight_dtype.value
        
        return {
            'input': input_mem,
            'output': output_mem,
            'weight': weight_mem,
        }