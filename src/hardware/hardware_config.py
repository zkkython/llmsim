"""
硬件配置模块 - 支持从 JSON/JSON5 配置文件动态加载硬件参数
"""
import json
import os
from dataclasses import dataclass
from enum import Enum

class DeviceType(Enum):
    """设备类型"""
    UNKNOWN = "unknown"
    GPU = "gpu"
    XPU = "xpu"
    ACCELERATOR = "accelerator"


@dataclass
class MemoryConfig:
    """显存配置"""
    hbm_size_gb: int = 96  # HBM显存大小(GB)
    cache_line_size: int = 128  # Cache行大小


@dataclass
class BandwidthConfig:
    """带宽配置"""
    hbm_bandwidth_gb_s: float = 1.8  # HBM带宽(T/s)
    dma_bandwidth_gb_s: float = 85.0  # DMA带宽(GB/s) - 扩展模式
    dma_bandwidth_decode_gb_s: float = 22.64  # DMA带宽(GB/s) - 解码模式
    network_bandwidth_gb_s: float = 85.0  # 网络带宽(GB/s)
    network_bandwidth_decode_gb_s: float = 22.64  # 网络带宽(GB/s) - 解码模式


@dataclass
class ComputeConfig:
    """计算配置"""
    mac_int8_gflops: float = 500.0  # INT8 MAC性能(TFLOPS)
    mac_fp32_gflops: float = 125.0  # FP32 MAC性能(TFLOPS)
    mac_bf16_gflops: float = 250.0  # BF16 MAC性能(TFLOPS)


@dataclass
class HardwareConfig:
    """硬件配置容器"""
    device_type: DeviceType = DeviceType.GPU
    name: str = "Default GPU"
    
    memory: MemoryConfig = None
    bandwidth: BandwidthConfig = None
    compute: ComputeConfig = None
    
    def __post_init__(self):
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.bandwidth is None:
            self.bandwidth = BandwidthConfig()
        if self.compute is None:
            self.compute = ComputeConfig()
    
    @classmethod
    def from_json(cls, config_path: str) -> "HardwareConfig":
        """
        从 JSON/JSON5 配置文件加载硬件配置
        
        Args:
            config_path: JSON 或 JSON5 配置文件路径
            
        Returns:
            HardwareConfig 实例
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Hardware config not found: {config_path}")
        
        # 根据文件扩展名选择解析器
        if config_path.endswith('.json5'):
            import json5
            with open(config_path, 'r') as f:
                data = json5.load(f)
        else:
            with open(config_path, 'r') as f:
                data = json.load(f)
        
        # 解析嵌套配置
        memory_data = data.get('memory', {})
        bandwidth_data = data.get('bandwidth', {})
        compute_data = data.get('compute', {})
        
        return cls(
            device_type=DeviceType(data.get('device_type', 'gpu')),
            name=data.get('name', 'Unknown Hardware'),
            memory=MemoryConfig(
                hbm_size_gb=memory_data.get('hbm_size_gb', 96),
                cache_line_size=memory_data.get('cache_line_size', 128),
            ),
            bandwidth=BandwidthConfig(
                hbm_bandwidth_gb_s=bandwidth_data.get('hbm_bandwidth_gb_s', 1.8),
                dma_bandwidth_gb_s=bandwidth_data.get('dma_bandwidth_gb_s', 85.0),
                dma_bandwidth_decode_gb_s=bandwidth_data.get('dma_bandwidth_decode_gb_s', 22.64),
                network_bandwidth_gb_s=bandwidth_data.get('network_bandwidth_gb_s', 85.0),
                network_bandwidth_decode_gb_s=bandwidth_data.get('network_bandwidth_decode_gb_s', 22.64),
            ),
            compute=ComputeConfig(
                mac_int8_gflops=compute_data.get('mac_int8_gflops', 500.0),
                mac_fp32_gflops=compute_data.get('mac_fp32_gflops', 125.0),
                mac_bf16_gflops=compute_data.get('mac_bf16_gflops', 250.0),
            ),
        )




# 硬件配置注册表 - 支持通过名称加载预定义配置
_HARDWARE_REGISTRY: dict[str, str] = {
    "default": "hardware_config/default_gpu.json5",
    "h20": "hardware_config/h20.json5",
    "h800": "hardware_config/h800.json5",
    "gb200": "hardware_config/gb200.json5",
    "klx_p800": "hardware_config/klx_p800.json5",
}


def get_hardware_config(name: str = "default") -> HardwareConfig:
    """
    通过名称获取预定义的硬件配置
    
    Args:
        name: 硬件配置名称 (default, h20, h800, gb200)
        
    Returns:
        HardwareConfig 实例
    """
    config_path = _HARDWARE_REGISTRY.get(name.lower())
    if config_path is None:
        raise ValueError(f"Unknown hardware config: {name}. Available: {list(_HARDWARE_REGISTRY.keys())}")
    
    # 如果是相对路径，添加项目根目录
    if not os.path.isabs(config_path):
        # 获取当前文件所在目录的父目录（即项目根目录）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, config_path)
    
    return HardwareConfig.from_json(config_path)


# 默认硬件配置 (回退配置)
DEFAULT_HARDWARE = get_hardware_config("default")
