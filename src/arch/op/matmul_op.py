from typing import Dict

from hardware.hardware_config import HardwareConfig
from src.arch.op.operator_base import BaseOperator, OperatorMetadata


class MatmulOperator(BaseOperator):
    """Matrix multiplication operator"""

    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)

    def get_compute_complexity(self) -> float:
        """Calculate FLOPs for matrix multiplication"""
        io = self.metadata.io_config
        # FLOPs = 2 * m * n * k (matrix multiplication requires m*n*k multiplications and additions)
        m = io.input_shape.m
        k = io.input_shape.n
        n = io.output_shape.n
        batch = self.metadata.batch_size

        return 2.0 * m * k * n * batch

    def get_memory_requirement(self) -> Dict[str, int]:
        """Get memory requirements for matrix multiplication"""
        """Theoretically, all matrix multiplication memory usage follows the logic below"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size

        input_mem = io.input_shape.size() * batch * io.input_dtype.value
        output_mem = io.output_shape.size() * batch * io.output_dtype.value
        weight_mem = io.weight_shape.size() * io.weight_dtype.value

        return {
            "input": input_mem,
            "output": output_mem,
            "weight": weight_mem,
        }

    def get_hbm_time(self, hardware: HardwareConfig) -> float:
        pass

    def get_weight_mem_occupy(self) -> float:
        return (
            self.metadata.io_config.weight_shape.size()
            * self.metadata.io_config.weight_dtype.value
        )
