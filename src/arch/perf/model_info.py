from dataclasses import dataclass

from src.arch.perf.model_perf import ModelPerformance


@dataclass
class ModelInfo:
    model_perf: ModelPerformance
    # unit is bytes
    model_kv_cache_per_gpu: float
    # unit is bytes
    model_kv_cache_total: float

    @property
    def model_kv_cache_per_gpu_gb(self) -> float:
        """Get KV cache per GPU in GB"""
        return self.model_kv_cache_per_gpu / 1024 / 1024 / 1024

    @property
    def model_kv_cache_total_gb(self) -> float:
        """Get total KV cache in GB"""
        return self.model_kv_cache_total / 1024 / 1024 / 1024
