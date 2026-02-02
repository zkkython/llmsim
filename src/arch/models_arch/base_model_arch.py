from abc import ABC, abstractmethod
from typing import List, Dict
from src.arch.op.operator_base import BaseOperator
from src.arch.config import ModelConfig, ScheduleConfig, ForwardMode

class BaseModelArch(ABC):
    """模型架构基类"""
    
    def __init__(self, model_config: ModelConfig, schedule_config: ScheduleConfig):
        """
        初始化模型架构, 一个模型结构至少需要模型配置，调度配置，算子，注意力算子，传输算子
        
        Args:
            model_config: 模型配置
            schedule_config: 调度配置
        """
        self.model_config = model_config
        self.schedule_config = schedule_config
        self.operators: List[BaseOperator] = []
        self.attention_operators: Dict[str, List[BaseOperator]] = {}
        self.transfer_operators: List[BaseOperator] = []
    
    @abstractmethod
    def build_operators(self) -> List[BaseOperator]:
        """构建模型的算子图"""
        pass
    
    def get_seq_length(self) -> int:
        """根据模式获取序列长度"""
        if self.schedule_config.mode == ForwardMode.EXTEND:
            return self.schedule_config.max_seqlen
        elif self.schedule_config.mode == ForwardMode.DECODE:
            return self.schedule_config.batch_size
        return self.schedule_config.max_seqlen
    
    def _add_operator(self, operator: BaseOperator) -> None:
        """添加算子到操作符列表"""
        self.operators.append(operator)
    
    def _add_attention_operator(self, key: str, operators: List[BaseOperator]) -> None:
        """添加注意力算子"""
        self.attention_operators[key] = operators
    
    def _add_transfer_operator(self, operator: BaseOperator) -> None:
        """添加传输算子"""
        self.transfer_operators.append(operator)
