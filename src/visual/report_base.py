from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.arch.perf_calculator import ModelPerformance

class ReportFormatter(ABC):
    """性能报告格式化器的抽象基类"""
    
    @abstractmethod
    def format(self, model_perf: ModelPerformance) -> Any:
        """
        格式化性能报告
        
        Args:
            model_perf: 模型性能指标
            
        Returns:
            格式化后的输出（内容取决于具体实现）
        """
        pass
    
    @abstractmethod
    def save(self, model_perf: ModelPerformance, output_path: str = None) -> None:
        """
        保存性能报告到文件或输出
        
        Args:
            model_perf: 模型性能指标
            output_path: 输出文件路径（可选）
        """
        pass
    
    def _collect_data(self, model_perf: ModelPerformance) -> List[Dict[str, Any]]:
        """
        收集所有性能数据供格式化器使用
        
        Args:
            model_perf: 模型性能指标
            
        Returns:
            包含所有行数据的列表
        """
        all_rows = []
        for layer_perf in model_perf.layer_performances:
            for op_perf in layer_perf.operators:
                op_meta = op_perf.metadata
                time_us = op_perf.total_time * 1000.0
                percentage = model_perf.get_percentage(time_us)
                all_rows.append({
                    'name': op_perf.name,
                    'type': op_perf.op_type,
                    'm': op_meta.io_config.input_shape.m,
                    #'n': op_meta.io_config.output_shape.n,
                    'n': op_meta.io_config.weight_shape.n if op_meta.io_config.weight_shape.n is not None else 0,
                    'k': op_meta.io_config.input_shape.n,
                    'batch': op_meta.batch_size,
                    'layers': op_meta.num_layers,
                    'in_dtype': op_meta.io_config.input_dtype.name,
                    'out_dtype': op_meta.io_config.output_dtype.name,
                    'weight_dtype': op_meta.io_config.weight_dtype.name,
                    'compute': op_perf.compute_time,
                    'memory': op_perf.memory_time,
                    'transfer': op_perf.transfer_time,
                    'op_time_single_layer': op_perf.op_time_single_layer,
                    'total': op_perf.total_time,
                    'percent': percentage,
                })
        return all_rows
