from typing import Dict

from src.arch.op.operator_base import BaseOperator, DataType, OperatorMetadata
from src.hardware.hardware_config import HardwareConfig


class AttentionOperator(BaseOperator):
    """Base class for attention operators"""

    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)

    def get_compute_complexity(self) -> float:
        """Calculate FLOPs for attention"""

        # Q*K^T: seq_len * head_dim * seq_len
        # Softmax(Q*K^T) * V: seq_len * seq_len * head_dim
        # Total: 2 * seq_len^2 * head_dim
        # print(f'Use here Attention FLOPs: {2.0 * seq_len * seq_len * head_dim}')
        # return 4.0 * seq_len * seq_len * head_dim

        def _legacy_cal_mac_time(
            cal_count: int, dtype: int, mac_int8: float = 500.0
        ) -> float:
            """
            old_main style MAC time calculation
            Returns: microseconds (us)
            """
            return 2 * cal_count / mac_int8 / 1000000.0 * dtype

        operate_io = self.metadata.io_config
        m, n, k = (
            operate_io.input_shape.m,
            operate_io.input_shape.n,
            operate_io.output_shape.n,
        )
        _count = m * n * k * self.metadata.batch_size
        compute_time = _legacy_cal_mac_time(_count, operate_io.weight_dtype.value)
        return compute_time

    def get_memory_requirement(self) -> Dict[str, int]:
        """Get memory requirements for attention"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size

        input_mem = io.input_shape.size() * batch * io.input_dtype.value
        output_mem = io.output_shape.size() * batch * io.output_dtype.value
        # Attention intermediate result (Q*K^T)
        intermediate_mem = (
            io.input_shape.m * io.input_shape.m * batch * DataType.FP32.value
        )

        return {
            "input": input_mem,
            "output": output_mem,
            "intermediate": intermediate_mem,
        }

    def _calculate_attention_hbm_time(
        self,
        load_count: int,
        load_dtype: int,
        store_count: int,
        store_dtype: int,
        dma: float,
    ) -> float:
        """
        Returns: microseconds (us)
        """
        return (load_count * load_dtype + store_count * store_dtype) / dma / 1000000.0

    def get_hbm_time(self, hardware: HardwareConfig) -> float:
        """Get HBM time for attention"""
        op_name = self.metadata.name
        if op_name == "qkv":
            load_count = (
                self.metadata.io_config.weight_shape.size() * self.metadata.batch_size
            )  # Right matrix case
            store_count = (
                self.metadata.io_config.input_shape.m
                * self.metadata.io_config.weight_shape.n
                * self.metadata.batch_size
            )  # Left matrix case
        else:
            load_count = (
                self.metadata.io_config.input_shape.size()
                + self.metadata.io_config.weight_shape.size()
            ) * self.metadata.batch_size
            store_count = 0
        memory_time = self._calculate_attention_hbm_time(
            load_count,
            self.metadata.io_config.input_dtype.value,
            store_count,
            self.metadata.io_config.output_dtype.value,
            hardware.bandwidth.hbm_bandwidth_gb_s,
        )
        # print(f"load_count={load_count}, store_count={store_count}, hbm={memory_time}")
        return memory_time

    def get_weight_mem_occupy(self) -> float:
        return 0
