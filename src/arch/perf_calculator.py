"""
性能计算引擎 - 统一的性能计算和分析模块
"""
from dataclasses import dataclass, field
from typing import List

from src.hardware.hardware_config import HardwareConfig
from src.arch.models_arch.model_arch import BaseModelArch
from src.arch.op.operator_base import BaseOperator
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


@dataclass
class LayerPerformance:
    """单层的性能指标"""
    layer_name: str = ""
    layer_type: str = ""
    
    # 算子列表
    operators: List[OperatorPerformance] = field(default_factory=list)
    
    # 聚合指标
    total_compute_time: float = 0.0
    total_memory_time: float = 0.0
    total_transfer_time: float = 0.0
    total_time: float = 0.0
    
    def add_operator(self, op_perf: OperatorPerformance) -> None:
        """添加算子性能"""
        self.operators.append(op_perf)
        self.total_compute_time += op_perf.compute_time
        self.total_memory_time += op_perf.memory_time
        self.total_transfer_time += op_perf.transfer_time
    
    def finalize(self) -> None:
        """计算最终的总时间"""
        # 总时间取计算和内存的最大值加上传输时间
        self.total_time = max(self.total_compute_time, self.total_memory_time) + self.total_transfer_time


@dataclass
class ModelPerformance:
    """整个模型的性能指标"""
    model_name: str = ""
    forward_mode: str = ""
    
    # 层级性能
    layer_performances: List[LayerPerformance] = field(default_factory=list)
    
    # 聚合指标
    total_compute_time: float = 0.0
    total_memory_time: float = 0.0
    total_transfer_time: float = 0.0
    total_time: float = 0.0
    
    def add_layer(self, layer_perf: LayerPerformance) -> None:
        """添加层性能"""
        self.layer_performances.append(layer_perf)
    
    def finalize(self) -> None:
        """计算最终指标"""
        sum_full_time = 0.0  # 所有算子的 full_time 之和（微秒），用于百分比计算
        
        for layer_perf in self.layer_performances:
            layer_perf.finalize()
            # 总时间取计算时间和内存时间的最大值（因为会流水线执行），加上传输时间
            layer_total = max(layer_perf.total_compute_time, layer_perf.total_memory_time) + layer_perf.total_transfer_time
            self.total_compute_time += layer_perf.total_compute_time
            self.total_memory_time += layer_perf.total_memory_time
            self.total_transfer_time += layer_perf.total_transfer_time
            # 总时间取所有层的最大值之和
            self.total_time += layer_total
        
        # 计算所有算子的 full_time 之和（用于百分比计算，与旧版本一致）
        # full_time = per_layer_time * layers，单位是微秒
        for layer_perf in self.layer_performances:
            for op_perf in layer_perf.operators:
                # 算子的 total_time 已经是 per_layer_time * layer_count 的结果（毫秒）
                # 需要乘以 1000 转换为微秒
                full_time_us = op_perf.total_time * 1000.0
                sum_full_time += full_time_us
        
        # 如果总时间为0，使用 sum_full_time
        if self.total_time == 0:
            self.total_time = sum_full_time / 1000.0  # 转换为毫秒
        
        # 保存 sum_full_time 用于百分比计算（微秒）
        self._sum_full_time = sum_full_time
    
    def get_bottleneck_op(self) -> tuple:
        """获取性能瓶颈算子"""
        max_time = 0.0
        bottleneck_op = None
        
        for layer_perf in self.layer_performances:
            for op_perf in layer_perf.operators:
                if op_perf.total_time > max_time:
                    max_time = op_perf.total_time
                    bottleneck_op = (layer_perf.layer_name, op_perf.name, op_perf)
        
        return bottleneck_op
    
    def get_percentage(self, time_us: float) -> float:
        """获取所占百分比（基于所有算子full_time之和）"""
        sum_full_time = getattr(self, '_sum_full_time', self.total_time)
        if sum_full_time == 0:
            return 0.0
        return time_us / sum_full_time * 100


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
        dma: float
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
        
        load_bytes = io_volume.get('load', 0)
        store_bytes = io_volume.get('store', 0)
        memory_time_us = (load_bytes + store_bytes) / self.hardware.bandwidth.hbm_bandwidth_gb_s / 1e6
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
        bandwidth_gb_s = getattr(operator, '_bandwidth_gb_s', None)
        if bandwidth_gb_s is None:
            # 根据算子名称选择合适的带宽
            if operator.metadata.name == 'dispatch':
                # dispatch 使用 DeepSeek 老版本中的 nb 配置（解码模式下 18.58）
                bandwidth_gb_s = self.hardware.bandwidth.dma_bandwidth_decode_gb_s
            elif operator.metadata.name == 'combine':
                # combine 使用 DeepSeek 老版本中的 nb 配置（解码模式下 22.64）
                bandwidth_gb_s = self.hardware.bandwidth.dma_bandwidth_decode_gb_s
            else:
                bandwidth_gb_s = self.hardware.bandwidth.network_bandwidth_gb_s
                
        io_volume = operator.get_io_volume()
        transfer_bytes = io_volume.get('transfer', io_volume.get('load', 0))
                
       
        # data.transfer = m * n * batch * dtype / nb / 1000.0
        # 这里 transfer_bytes = m * n * batch * dtype（字节数），nb = GB/s
        # 返回 "微秒/层"
        transfer_time_us = transfer_bytes / bandwidth_gb_s / 1000.0
                
        return transfer_time_us
    def calculate_operator_performance(self, operator: BaseOperator) -> OperatorPerformance:
        """
        计算单个算子的性能指标
        
        Args:
            operator: 算子实例
            
        Returns:
            算子性能指标
        """
        metadata = operator.metadata
        
        # 计算不同类型的时间
        if metadata.op_type == 'transfer':
            compute_time = 0.0
            memory_time = 0.0
            transfer_time = self.calculate_transfer_time(operator)
        elif metadata.op_type == 'attention':
            transfer_time = 0.0
            # compute_time = self.calculate_compute_time(operator)
            def _legacy_cal_mac_time(cal_count: int, dtype: int, mac_int8: float = 500.0) -> float:
                """
                old_main 风格的 MAC 时间计算
                返回: 微秒 (us)
                """
                return 2 * cal_count / mac_int8 / 1000000.0 * dtype
            operate_io = metadata.io_config
            m, n, k = operate_io.input_shape.m, operate_io.input_shape.n, operate_io.output_shape.n
            _count = m * n * k * metadata.batch_size
            compute_time = _legacy_cal_mac_time(_count, operate_io.weight_dtype.value)

            #print(f'op name = {metadata.name}, shape = {operate_io.input_shape}, {operate_io.output_shape}, cost {compute_time}')
            op_name = metadata.name
            if op_name == 'qkv':
                load_count = operate_io.weight_shape.size() * metadata.batch_size # 右边矩阵情况
                store_count = operate_io.input_shape.m * operate_io.weight_shape.n * metadata.batch_size # 左边矩阵情况
            else:
                load_count = (operate_io.input_shape.size() + operate_io.weight_shape.size()) * metadata.batch_size
                store_count = 0
            memory_time = self.calculate_attention_hbm_time(
                    load_count, 
                    operate_io.input_dtype.value, 
                    store_count, 
                    operate_io.output_dtype.value, 
                    self.hardware.bandwidth.hbm_bandwidth_gb_s
                )
            print(f'load_count={load_count}, store_count={store_count}, hbm={memory_time}')
            
        elif metadata.op_type == 'matmul':
            compute_time = self.calculate_compute_time(operator)
            
            memory_time = self.calculate_memory_time(operator)
            #print(f'name = {operator.metadata.name}, compute_time = {compute_time}， memory_time={memory_time}')
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
            total_time=total_time / 1000.0,  # 转换为毫秒
            flops=operator.get_compute_complexity() * layer_count,
            memory_volume=operator.get_memory_requirement().get('weight', 0),
            io_volume=operator.get_io_volume().get('load', 0) + operator.get_io_volume().get('store', 0),
            metadata=metadata,
        )
        
        return op_perf
    

    def calculate_model_performance(self, model_arch: BaseModelArch) -> ModelPerformance:
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
        )
        
        # 矩阵算子的大部分在这里，attention 在attention那，传输的在传输那里
        for operator in model_arch.operators:
            op_perf = self.calculate_operator_performance(operator)
            layer_perf = LayerPerformance(layer_name=op_perf.name, layer_type='compute')
            layer_perf.add_operator(op_perf)
            model_perf.add_layer(layer_perf)
        
        # 处理注意力算子
        for attn_key, operators in model_arch.attention_operators.items():
            for operator in operators:
                op_perf = self.calculate_operator_performance(operator)
                layer_perf = LayerPerformance(layer_name=attn_key, layer_type='attention')
                layer_perf.add_operator(op_perf)
                model_perf.add_layer(layer_perf)
        
        # 处理传输算子
        for operator in model_arch.transfer_operators:
            op_perf = self.calculate_operator_performance(operator)
            layer_perf = LayerPerformance(layer_name=op_perf.name, layer_type='transfer')
            layer_perf.add_operator(op_perf)
            model_perf.add_layer(layer_perf)
        
        model_perf.finalize()
        
        return model_perf
    
    def print_performance_report(self, model_perf: ModelPerformance, output_format: str = 'console', output_path: str = None) -> None:
        """
        打印性能报告
        
        Args:
            model_perf: 模型性能指标
            output_format: 输出格式 ('console' 或 'excel')
            output_path: 输出文件路径（可选，仅对某些格式有效）
        """
        from src.arch.report_formatter import create_formatter
        
        formatter = create_formatter(output_format)
        
        if output_path:
            formatter.save(model_perf, output_path)
        else:
            formatter.save(model_perf)