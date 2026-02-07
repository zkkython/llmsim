"""
Operator abstraction layer - Provides generic operator definitions and interfaces
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from src.hardware.hardware_config import HardwareConfig


class DataType(Enum):
    """Data type"""

    INT8 = 1
    FP8 = 1
    FP16 = 2
    BF16 = 2
    FP32 = 4
    FP64 = 8


@dataclass
class Tensor:
    """Tensor definition"""

    m: int = 0
    n: int = 0

    @property
    def shape(self) -> tuple:
        return (self.m, self.n)

    def size(self) -> int:
        """Calculate number of tensor elements"""
        return self.m * self.n


@dataclass
class OperatorIO:
    """Operator input/output definition"""

    input_shape: Tensor = field(default_factory=Tensor)
    output_shape: Tensor = field(default_factory=Tensor)
    weight_shape: Tensor = field(default_factory=Tensor)

    input_dtype: DataType = DataType.BF16
    output_dtype: DataType = DataType.BF16
    weight_dtype: DataType = DataType.BF16


@dataclass
class OperatorMetadata:
    """Operator metadata"""

    name: str = ""
    op_type: str = ""  # Operator type: matmul, conv, etc
    description: str = ""

    io_config: OperatorIO = field(default_factory=OperatorIO)

    # Execution parameters
    batch_size: int = 1
    num_layers: int = 1

    # Parallelization parameters
    parallelization_dim: Optional[str] = None  # Parallel dimension marker


class BaseOperator(ABC):
    """Base operator abstract class"""

    def __init__(self, metadata: OperatorMetadata):
        """
        Initialize operator

        Args:
            metadata: Operator metadata
        """
        self.metadata = metadata

    @abstractmethod
    def get_compute_complexity(self) -> float:
        """
        Get compute complexity (FLOPs)

        Returns:
            Number of FLOPs
        """
        pass

    @abstractmethod
    def get_memory_requirement(self) -> Dict[str, int]:
        """
        Get memory requirements

        Returns:
            Memory requirements dictionary {
                'input': input memory size (bytes),
                'output': output memory size (bytes),
                'weight': weight memory size (bytes)
            }
        """
        pass

    def get_io_volume(self) -> Dict[str, int]:
        """
        Get memory access I/O data volume

        Returns:
            I/O data volume dictionary {
                'load': load data volume (bytes),
                'store': store data volume (bytes)
            }
        """
        io = self.metadata.io_config
        input_size = (
            io.input_shape.size() * self.metadata.batch_size * io.input_dtype.value
        )
        output_size = (
            io.output_shape.size() * self.metadata.batch_size * io.output_dtype.value
        )
        weight_size = (
            io.weight_shape.size() * self.metadata.batch_size * io.weight_dtype.value
        )

        return {
            "load": input_size + weight_size,
            "store": output_size,
        }

    @abstractmethod
    def get_hbm_time(self, hardware: HardwareConfig) -> float:
        pass

    @abstractmethod
    def get_weight_mem_occupy(self) -> float:
        """
        Get weight memory occupancy

        Args:
            hardware: Hardware configuration

        Returns:
            Weight memory occupancy

        Unit:
            bytes
        """
        pass
