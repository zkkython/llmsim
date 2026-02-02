from src.arch.op.operator_base import BaseOperator,OperatorMetadata, DataType
from typing import Dict

class AttentionOperator(BaseOperator):
    """注意力算子基类"""
    
    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)
    
    def get_compute_complexity(self) -> float:
        """计算注意力的 FLOPs"""
        io = self.metadata.io_config
        # Q-K 矩阵乘法
        seq_len = io.input_shape.m
        head_dim = io.input_shape.n
        
        # Q*K^T: seq_len * head_dim * seq_len
        # Softmax(Q*K^T) * V: seq_len * seq_len * head_dim
        # 总计: 2 * seq_len^2 * head_dim
        # print(f'Use here Attention FLOPs: {2.0 * seq_len * seq_len * head_dim}')
        return 4.0 * seq_len * seq_len * head_dim
    
    def get_memory_requirement(self) -> Dict[str, int]:
        """获取注意力的内存需求"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size
        
        input_mem = io.input_shape.size() * batch * io.input_dtype.value
        output_mem = io.output_shape.size() * batch * io.output_dtype.value
        # 注意力中间结果（Q*K^T）
        intermediate_mem = io.input_shape.m * io.input_shape.m * batch * DataType.FP32.value
        
        return {
            'input': input_mem,
            'output': output_mem,
            'intermediate': intermediate_mem,
        }