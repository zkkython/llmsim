from typing import Dict

from src.arch.op.operator_base import BaseOperator, OperatorMetadata
from src.hardware.hardware_config import HardwareConfig


class TransferOperator(BaseOperator):
    """Base class for data transfer operators"""

    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)

    def get_compute_complexity(self) -> float:
        """Transfer operators have no compute complexity"""
        return 0.0

    def get_memory_requirement(self) -> Dict[str, int]:
        """Get data volume for transfer"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size

        transfer_size = io.input_shape.size() * batch * io.input_dtype.value

        return {
            "transfer": transfer_size,
        }

    def get_hbm_time(self, hardware: HardwareConfig) -> float:
        return 0.0

    def get_weight_mem_occupy(self) -> float:
        return 0.0
