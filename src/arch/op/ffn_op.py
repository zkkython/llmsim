from typing import Dict

from src.arch.op.operator_base import BaseOperator, OperatorMetadata
from src.hardware.hardware_config import HardwareConfig


class FFNOperator(BaseOperator):
    """Feed-Forward Network operator"""

    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)

    def get_compute_complexity(self) -> float:
        """Calculate FLOPs for FFN"""
        io = self.metadata.io_config
        m = io.input_shape.m
        k = io.input_shape.n
        n = io.output_shape.n
        batch = self.metadata.batch_size

        # FFN: input * w1 + w1_out * w2
        # Usually w1 is gate+up projection, computing two matrix multiplications
        return 2.0 * 2.0 * m * k * n * batch

    def get_memory_requirement(self) -> Dict[str, int]:
        """Get memory requirements for FFN"""
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
