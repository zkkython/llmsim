from dataclasses import dataclass

from arch.perf.model_perf import ModelPerformance


@dataclass
class ModelInfo:
    model_perf: ModelPerformance
    model_kv_cache_per_gpu: float


