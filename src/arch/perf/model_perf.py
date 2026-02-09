from dataclasses import dataclass, field
from typing import List

from src.arch.config import ScheduleConfig
from src.arch.model_type import ForwardMode
from src.arch.perf.layer_perf import LayerPerformance


@dataclass
class ModelPerformance:
    """Performance metrics for the entire model"""

    model_name: str = ""
    forward_mode: str = ""

    # Layer-level performance
    layer_performances: List[LayerPerformance] = field(default_factory=list)

    # Aggregated metrics
    total_compute_time: float = 0.0
    total_memory_time: float = 0.0
    total_transfer_time: float = 0.0
    total_time: float = 0.0
    ttft: float = 0.0
    throughput: float = 0.0

    schedule_config: ScheduleConfig = field(default_factory=ScheduleConfig)
    model_total_mem_occupy: float = 0.0

    def add_layer(self, layer_perf: LayerPerformance) -> None:
        """Add layer performance"""
        self.layer_performances.append(layer_perf)

    def finalize(self) -> None:
        """Calculate final metrics"""
        sum_full_time = 0.0  # Sum of full_time for all operators (microseconds), used for percentage calculation

        for layer_perf in self.layer_performances:
            layer_perf.finalize()
            # Total time takes the maximum of compute time and memory time (due to pipelining), plus transfer time
            layer_total = (
                max(layer_perf.total_compute_time, layer_perf.total_memory_time)
                + layer_perf.total_transfer_time
            )
            self.total_compute_time += layer_perf.total_compute_time
            self.total_memory_time += layer_perf.total_memory_time
            self.total_transfer_time += layer_perf.total_transfer_time
            # Total time is the sum of maximum values across all layers
            self.total_time += layer_total
            self.model_total_mem_occupy += layer_perf.layer_total_mem_occupy

        # Calculate sum of full_time for all operators (used for percentage calculation, consistent with old version)
        # full_time = per_layer_time * layers, unit is microseconds
        for layer_perf in self.layer_performances:
            for op_perf in layer_perf.operators:
                # Operator's total_time is already per_layer_time * layer_count (milliseconds)
                # Need to multiply by 1000 to convert to microseconds
                full_time_us = op_perf.total_time * 1000.0
                sum_full_time += full_time_us
                self.ttft += op_perf.total_time

        # If total time is 0, use sum_full_time
        if self.total_time == 0:
            self.total_time = sum_full_time / 1000.0  # Convert to milliseconds

        # Save sum_full_time for percentage calculation (microseconds)
        self._sum_full_time = sum_full_time

    def get_bottleneck_op(self) -> tuple:
        """Get performance bottleneck operator"""
        max_time = 0.0
        bottleneck_op = None

        for layer_perf in self.layer_performances:
            for op_perf in layer_perf.operators:
                if op_perf.total_time > max_time:
                    max_time = op_perf.total_time
                    bottleneck_op = (layer_perf.layer_name, op_perf.name, op_perf)

        return bottleneck_op

    def get_percentage(self, time_us: float) -> float:
        """Get percentage (based on sum of full_time for all operators)"""
        sum_full_time = getattr(self, "_sum_full_time", self.total_time)
        if sum_full_time == 0:
            return 0.0
        return time_us / sum_full_time * 100

    def get_ttft_or_tpot(self) -> float:
        """TTFT in ms, 0.02 is to account for framework overhead"""
        return self.ttft * 1.02

    def get_throughput(self) -> float:
        """Throughput TPS: tokens/second, need to distinguish between prefill and decode"""
        """
        Calculate throughput (tokens/second)
        Prefill mode: (batch_size * seq_len) / TTFT
        Decode mode: batch_size / time_per_token
        """
        mode = self.schedule_config.mode
        if mode == ForwardMode.EXTEND:  # Prefill
            total_tokens = (
                self.schedule_config.batch_size * self.schedule_config.max_seqlen
            )
            ttft_seconds = self.get_ttft_or_tpot() / 1000.0
            return total_tokens / ttft_seconds if ttft_seconds > 0 else 0.0
        else:  # Decode
            # Decode phase: generation time per token
            # TODO needs adjustment
            time_per_token_ms = (
                self.get_ttft_or_tpot()
            )  # Assume total_time is per-token time
            time_per_token_s = time_per_token_ms / 1000.0
            return (
                self.schedule_config.batch_size / time_per_token_s
                if time_per_token_s > 0
                else 0.0
            )

    def get_throughput_single_gpu(self) -> float:
        return self.get_throughput() / (
            self.schedule_config.tp_size * self.schedule_config.dp_size
        )
