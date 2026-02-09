"""
Performance evaluator - bridges optimizer with performance calculator
"""

from typing import List, Optional

from src.arch.config import ModelConfig, ScheduleConfig
from src.arch.models_arch.model_arch import create_model_arch
from src.arch.perf.model_perf import ModelPerformance
from src.arch.perf_calculator import PerformanceCalculator
from src.hardware.hardware_config import HardwareConfig


class PerformanceEvaluator:
    """
    Evaluates performance for given schedule configurations

    Acts as a bridge between the optimizer and the performance calculator,
    handling configuration validation and error recovery.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hardware_config: HardwareConfig,
    ):
        """
        Initialize evaluator

        Args:
            model_config: Model configuration (fixed)
            hardware_config: Hardware configuration (fixed)
        """
        self.model_config = model_config
        self.hardware_config = hardware_config
        self.calculator = PerformanceCalculator(hardware_config)
        self._cache: dict = {}  # Simple cache for performance results

    def _get_cache_key(self, schedule_config: ScheduleConfig) -> str:
        """Generate cache key for a configuration"""
        return (
            f"tp{schedule_config.tp_size}_"
            f"dp{schedule_config.dp_size}_"
            f"ep{schedule_config.ep_size}_"
            f"bs{schedule_config.batch_size}_"
            f"seq{schedule_config.max_seqlen}_"
            f"mode{schedule_config.mode.name}"
        )

    def evaluate(self, schedule_config: ScheduleConfig) -> Optional[ModelPerformance]:
        """
        Evaluate performance for a schedule configuration

        Args:
            schedule_config: Schedule configuration to evaluate

        Returns:
            ModelPerformance if successful, None if evaluation failed
        """
        cache_key = self._get_cache_key(schedule_config)

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Create model architecture
            model_arch = create_model_arch(
                model_config=self.model_config, schedule_config=schedule_config
            )

            # Calculate performance
            perf = self.calculator.calculate_model_performance(model_arch)

            # Cache result
            self._cache[cache_key] = perf

            return perf

        except Exception as e:
            # Log error and return None to indicate failure
            # This allows optimizer to handle gracefully
            print(f"Warning: Evaluation failed for config {cache_key}: {e}")
            return None

    def evaluate_batch(
        self, schedule_configs: List[ScheduleConfig], parallel_workers: int = 1
    ) -> List[Optional[ModelPerformance]]:
        """
        Evaluate multiple configurations

        Args:
            schedule_configs: List of schedule configurations
            parallel_workers: Number of parallel workers (currently unused)

        Returns:
            List of ModelPerformance (or None for failed evaluations)
        """
        # Currently sequential; could be parallelized in the future
        results = []
        for config in schedule_configs:
            perf = self.evaluate(config)
            results.append(perf)
        return results

    def get_model_memory_estimate(self, schedule_config: ScheduleConfig) -> float:
        """
        Estimate model memory usage in GB

        Args:
            schedule_config: Schedule configuration

        Returns:
            Estimated memory in GB
        """
        try:
            model_arch = create_model_arch(
                model_config=self.model_config, schedule_config=schedule_config
            )
            model_arch.build_operators()

            # Calculate total parameter memory
            total_params = 0
            for operator in model_arch.operators:
                params = operator.get_params()
                total_params += params

            # Assume FP16/BF16 (2 bytes per parameter)
            memory_bytes = total_params * 2
            memory_gb = memory_bytes / (1024**3)

            return memory_gb

        except Exception as e:
            print(f"Warning: Memory estimation failed: {e}")
            return float("inf")

    def clear_cache(self) -> None:
        """Clear the evaluation cache"""
        self._cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {"cache_size": len(self._cache), "cache_keys": list(self._cache.keys())}
