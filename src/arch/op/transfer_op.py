from src.arch.op.operator_base import BaseOperator, OperatorMetadata
from typing import Dict

class TransferOperator(BaseOperator):
    """数据传输算子基类"""
    
    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)
    
    def get_compute_complexity(self) -> float:
        """传输算子没有计算复杂度"""
        return 0.0
    
    def get_memory_requirement(self) -> Dict[str, int]:
        """获取传输的数据量"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size
        
        transfer_size = io.input_shape.size() * batch * io.input_dtype.value
        
        return {
            'transfer': transfer_size,
        }