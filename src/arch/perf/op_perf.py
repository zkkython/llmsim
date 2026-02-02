from dataclasses import dataclass, field

from src.arch.op.operator_base import OperatorMetadata


@dataclass
class OperatorPerformance:
    """单个算子的性能指标"""

    name: str = ""
    op_type: str = ""

    # 时间指标 (微秒)
    compute_time: float = 0.0
    memory_time: float = 0.0
    transfer_time: float = 0.0
    total_time: float = 0.0

    # 计算和内存指标
    flops: float = 0.0
    memory_volume: float = 0.0
    io_volume: float = 0.0

    # 记录这个算子的基本信息，用于可视化
    metadata: OperatorMetadata = field(default_factory=OperatorMetadata)

    def __str__(self) -> str:
        return (
            f"OperatorPerformance(name={self.name}, "
            f"compute={self.compute_time:.3f}us, "
            f"memory={self.memory_time:.3f}us, "
            f"transfer={self.transfer_time:.3f}us, "
            f"total={self.total_time:.3f}us)"
        )
