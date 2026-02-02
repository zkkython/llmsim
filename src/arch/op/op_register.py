from src.arch.op.attn_op import AttentionOperator
from src.arch.op.matmul_op import MatmulOperator
from src.arch.op.ffn_op import FFNOperator
from src.arch.op.transfer_op import TransferOperator
from typing import Dict
from src.arch.op.operator_base import BaseOperator, OperatorMetadata

# 算子类型注册表
OPERATOR_REGISTRY: Dict[str, type] = {
    'matmul': MatmulOperator,
    'attention': AttentionOperator,
    'ffn': FFNOperator,
    'transfer': TransferOperator,
}


def create_operator(op_type: str, metadata: OperatorMetadata) -> BaseOperator:
    """
    工厂函数 - 创建算子实例
    
    Args:
        op_type: 算子类型
        metadata: 算子元数据
        
    Returns:
        算子实例
    """
    operator_class = OPERATOR_REGISTRY.get(op_type, MatmulOperator)
    return operator_class(metadata)
