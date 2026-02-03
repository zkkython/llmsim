"""
性能计算引擎 - 统一的性能计算和分析模块
"""

from src.arch.models_arch.model_arch import BaseModelArch
from src.arch.op.operator_base import BaseOperator
from src.arch.perf.layer_perf import LayerPerformance
from src.arch.perf.model_perf import ModelPerformance
from src.arch.perf.op_perf import OperatorPerformance
from src.hardware.hardware_config import HardwareConfig


class PerformanceCalculator:
    """性能计算引擎"""

    def __init__(self, hardware_config: HardwareConfig):
        """
        初始化性能计算器

        Args:
            hardware_config: 硬件配置
        """
        self.hardware = hardware_config

    def calculate_compute_time(self, operator: BaseOperator) -> float:
        """
        计算算子的计算时间

        Args:
            operator: 算子实例

        Returns:
            计算时间 (微秒)
        """
        flops = operator.get_compute_complexity()

        if flops == 0:
            return 0.0

        # 根据数据类型选择合适的 MAC 性能
        io_config = operator.metadata.io_config
        dtype_bytes = io_config.weight_dtype.value

        if dtype_bytes == 1:  # INT8
            mac_gflops = self.hardware.compute.mac_int8_gflops
        elif dtype_bytes == 4:  # FP32
            mac_gflops = self.hardware.compute.mac_fp32_gflops
        else:  # BF16/FP16 (默认)
            mac_gflops = self.hardware.compute.mac_bf16_gflops

        # 时间 = FLOPs / (GFLOPS * 10^9) * 10^6 = FLOPs / (GFLOPS * 1000 * 1000)
        compute_time_us = flops / (mac_gflops * 1e6)

        return compute_time_us

    def calculate_attention_hbm_time(
        self,
        load_count: int,
        load_dtype: int,
        store_count: int,
        store_dtype: int,
        dma: float,
    ) -> float:
        """
        返回: 微秒 (us)
        """
        return (load_count * load_dtype + store_count * store_dtype) / dma / 1000000.0

    def calculate_memory_time(self, operator: BaseOperator) -> float:
        """
        计算算子的内存访问时间

        Args:
            operator: 算子实例

        Returns:
            内存时间 (微秒)
        """
        io_volume = operator.get_io_volume()

        load_bytes = io_volume.get("load", 0)
        store_bytes = io_volume.get("store", 0)
        memory_time_us = (
            (load_bytes + store_bytes)
            / self.hardware.bandwidth.hbm_bandwidth_gb_s
            / 1e6
        )
        return memory_time_us

    def calculate_transfer_time(self, operator: BaseOperator) -> float:
        """
        计算传输算子的传输时间

        Args:
            operator: 算子实例
            bandwidth_gb_s: 带宽 (GB/s)，如果为 None 使用默认网络带宽

        Returns:
            传输时间 (微秒)
        """
        # 获取带宽（从算子的自定义属性中获取，或使用默认值）
        bandwidth_gb_s = getattr(operator, "_bandwidth_gb_s", None)
        if bandwidth_gb_s is None:
            # 根据算子名称选择合适的带宽
            if operator.metadata.name == "dispatch":
                # dispatch 使用 DeepSeek 老版本中的 nb 配置（解码模式下 18.58）
                bandwidth_gb_s = self.hardware.bandwidth.dma_bandwidth_decode_gb_s
            elif operator.metadata.name == "combine":
                # combine 使用 DeepSeek 老版本中的 nb 配置（解码模式下 22.64）
                bandwidth_gb_s = self.hardware.bandwidth.dma_bandwidth_decode_gb_s
            else:
                bandwidth_gb_s = self.hardware.bandwidth.network_bandwidth_gb_s

        io_volume = operator.get_io_volume()
        transfer_bytes = io_volume.get("transfer", io_volume.get("load", 0))

        # data.transfer = m * n * batch * dtype / nb / 1000.0
        # 这里 transfer_bytes = m * n * batch * dtype（字节数），nb = GB/s
        # 返回 "微秒/层"
        transfer_time_us = transfer_bytes / bandwidth_gb_s / 1000.0

        return transfer_time_us

    def calculate_operator_performance(
        self, operator: BaseOperator
    ) -> OperatorPerformance:
        """
        计算单个算子的性能指标

        Args:
            operator: 算子实例

        Returns:
            算子性能指标
        """
        metadata = operator.metadata

        # 计算不同类型的时间
        if metadata.op_type == "transfer":
            compute_time = 0.0
            memory_time = 0.0
            transfer_time = self.calculate_transfer_time(operator)
        elif metadata.op_type == "attention":
            transfer_time = 0.0
            compute_time = operator.get_compute_complexity()
            memory_time = operator.get_hbm_time(hardware=self.hardware)

        elif metadata.op_type == "matmul":
            compute_time = self.calculate_compute_time(operator)
            memory_time = self.calculate_memory_time(operator)
            # print(f'name = {operator.metadata.name}, compute_time = {compute_time}， memory_time={memory_time}')
            transfer_time = 0.0

        # 每层的时间 (乘以层数)
        layer_count = metadata.num_layers
        total_compute_time = compute_time * layer_count
        total_memory_time = memory_time * layer_count
        total_transfer_time = transfer_time * layer_count

        # 总时间取最大值
        total_time = max(total_compute_time, total_memory_time) + total_transfer_time

        op_perf = OperatorPerformance(
            name=metadata.name,
            op_type=metadata.op_type,
            compute_time=compute_time,
            memory_time=memory_time,
            transfer_time=transfer_time,
            op_time_single_layer=max(compute_time, memory_time) + transfer_time,
            total_time=total_time / 1000.0,  # 转换为毫秒
            flops=operator.get_compute_complexity() * layer_count,
            memory_volume=operator.get_memory_requirement().get("weight", 0),
            io_volume=operator.get_io_volume().get("load", 0) + operator.get_io_volume().get("store", 0),
            metadata=metadata,
        )

        return op_perf

    def calculate_model_performance(
        self, model_arch: BaseModelArch
    ) -> ModelPerformance:
        """
        计算整个模型的性能

        Args:
            model_arch: 模型架构实例

        Returns:
            模型性能指标
        """
        model_arch.build_operators()

        model_perf = ModelPerformance(
            model_name=model_arch.model_config.model_type,
            forward_mode=model_arch.schedule_config.mode.name,
            schedule_config=model_arch.schedule_config,
        )

        # 矩阵算子的大部分在这里，attention 在attention那，传输的在传输那里
        for operator in model_arch.operators:
            op_perf = self.calculate_operator_performance(operator)
            layer_perf = LayerPerformance(layer_name=op_perf.name, layer_type="compute")
            layer_perf.add_operator(op_perf)
            model_perf.add_layer(layer_perf)

        # 处理注意力算子
        for attn_key, operators in model_arch.attention_operators.items():
            for operator in operators:
                op_perf = self.calculate_operator_performance(operator)
                layer_perf = LayerPerformance(
                    layer_name=attn_key, layer_type="attention"
                )
                layer_perf.add_operator(op_perf)
                model_perf.add_layer(layer_perf)

        # 处理传输算子
        for operator in model_arch.transfer_operators:
            op_perf = self.calculate_operator_performance(operator)
            layer_perf = LayerPerformance(
                layer_name=op_perf.name, layer_type="transfer"
            )
            layer_perf.add_operator(op_perf)
            model_perf.add_layer(layer_perf)

        model_perf.finalize()

        return model_perf

    def print_performance_report(
        self,
        model_perf: ModelPerformance,
        output_format: str = "console",
        output_path: str = None,
    ) -> None:
        """
        打印性能报告

        Args:
            model_perf: 模型性能指标
            output_format: 输出格式 ('console' 或 'excel')
            output_path: 输出文件路径（可选，仅对某些格式有效）
        """
        from src.visual.report_formatter import create_formatter

        formatter = create_formatter(output_format)

        if output_path:
            formatter.save(model_perf, output_path)
        else:
            formatter.save(model_perf)
