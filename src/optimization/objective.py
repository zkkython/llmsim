"""
Objective functions for optimization

Defines different optimization goals: minimize TTFT, maximize throughput,
or multi-objective optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from src.arch.perf.model_perf import ModelPerformance


class BaseObjective(ABC):
    """Abstract base class for objective functions"""

    def __init__(self, normalize: bool = False):
        """
        Args:
            normalize: Whether to normalize metrics before evaluation
        """
        self.normalize = normalize
        self._normalization_bounds: Optional[Dict[str, tuple]] = None

    @abstractmethod
    def evaluate(self, perf: ModelPerformance) -> float:
        """
        Evaluate performance and return a score

        Lower scores are better (minimization convention)

        Args:
            perf: Model performance metrics

        Returns:
            Score (lower is better)
        """
        pass

    @abstractmethod
    def get_metrics(self, perf: ModelPerformance) -> Dict[str, float]:
        """
        Get relevant metrics from performance object

        Args:
            perf: Model performance metrics

        Returns:
            Dictionary of metric name -> value
        """
        pass

    def set_normalization_bounds(self, bounds: Dict[str, tuple]) -> None:
        """
        Set bounds for normalization

        Args:
            bounds: Dictionary of metric name -> (min, max) bounds
        """
        self._normalization_bounds = bounds

    def _normalize(self, value: float, metric_name: str) -> float:
        """Normalize a value using stored bounds"""
        if not self.normalize or self._normalization_bounds is None:
            return value

        bounds = self._normalization_bounds.get(metric_name)
        if bounds is None:
            return value

        min_val, max_val = bounds
        if max_val == min_val:
            return 0.0

        return (value - min_val) / (max_val - min_val)


class MinimizeTTFT(BaseObjective):
    """Objective to minimize Time To First Token (latency)"""

    def evaluate(self, perf: ModelPerformance) -> float:
        """Return TTFT in milliseconds (lower is better)"""
        ttft = perf.get_ttft_or_tpot()
        return self._normalize(ttft, "ttft")

    def get_metrics(self, perf: ModelPerformance) -> Dict[str, float]:
        return {
            "ttft_ms": perf.get_ttft_or_tpot(),
            "total_time_ms": perf.total_time,
        }


class MaximizeThroughput(BaseObjective):
    """Objective to maximize throughput (TPS - tokens per second)"""

    def evaluate(self, perf: ModelPerformance) -> float:
        """
        Return negative throughput (lower is better for minimization convention)

        We use negative because optimizers minimize by default.
        """
        throughput = perf.get_throughput()
        # Return negative for minimization convention, or large value if throughput is 0
        if throughput <= 0:
            return float("inf")
        normalized = self._normalize(throughput, "throughput")
        # For maximization, we negate the normalized value
        return -normalized if self.normalize else -throughput

    def get_metrics(self, perf: ModelPerformance) -> Dict[str, float]:
        throughput = perf.get_throughput()
        return {
            "throughput_tps": throughput,
            "throughput_per_gpu": perf.get_throughput_single_gpu(),
            "total_time_ms": perf.total_time,
        }


class MinimizeTotalTime(BaseObjective):
    """Objective to minimize total execution time"""

    def evaluate(self, perf: ModelPerformance) -> float:
        """Return total time in milliseconds"""
        return self._normalize(perf.total_time, "total_time")

    def get_metrics(self, perf: ModelPerformance) -> Dict[str, float]:
        return {
            "total_time_ms": perf.total_time,
            "compute_time_ms": perf.total_compute_time
            / 1000.0,  # Convert from us to ms
            "memory_time_ms": perf.total_memory_time / 1000.0,
        }


class MultiObjective(BaseObjective):
    """
    Multi-objective optimization combining multiple objectives with weights

    Example:
        objective = MultiObjective({
            "ttft": (MinimizeTTFT(), 0.4),
            "throughput": (MaximizeThroughput(), 0.6)
        })
    """

    def __init__(self, objectives: Dict[str, tuple], normalize: bool = True):
        """
        Args:
            objectives: Dictionary mapping names to (objective, weight) tuples
            normalize: Whether to normalize metrics before combining
        """
        super().__init__(normalize=normalize)
        self.objectives = objectives

    def evaluate(self, perf: ModelPerformance) -> float:
        """
        Compute weighted combination of objectives

        Returns:
            Weighted score (lower is better)
        """
        total_score = 0.0
        total_weight = 0.0

        for name, (objective, weight) in self.objectives.items():
            score = objective.evaluate(perf)
            if score == float("inf"):
                return float("inf")
            total_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return float("inf")

        return total_score / total_weight

    def get_metrics(self, perf: ModelPerformance) -> Dict[str, float]:
        """Get metrics from all sub-objectives"""
        metrics = {}
        for name, (objective, _) in self.objectives.items():
            sub_metrics = objective.get_metrics(perf)
            for key, value in sub_metrics.items():
                metrics[f"{name}_{key}"] = value
        return metrics

    def set_normalization_bounds(self, bounds: Dict[str, tuple]) -> None:
        """Set bounds for all sub-objectives"""
        super().set_normalization_bounds(bounds)
        for objective, _ in self.objectives.values():
            objective.set_normalization_bounds(bounds)


class BalancedObjective(MultiObjective):
    """
    Balanced objective that equally weights TTFT and throughput

    This is a convenience class for common use case.
    """

    def __init__(
        self,
        ttft_weight: float = 0.5,
        throughput_weight: float = 0.5,
        normalize: bool = True,
    ):
        """
        Args:
            ttft_weight: Weight for TTFT minimization
            throughput_weight: Weight for throughput maximization
            normalize: Whether to normalize metrics
        """
        objectives = {
            "ttft": (MinimizeTTFT(normalize=normalize), ttft_weight),
            "throughput": (MaximizeThroughput(normalize=normalize), throughput_weight),
        }
        super().__init__(objectives, normalize=normalize)


def create_objective(objective_type: str, **kwargs) -> BaseObjective:
    """
    Factory function to create objective functions by name

    Args:
        objective_type: Type of objective ("minimize_ttft", "maximize_tps", "balanced", etc.)
        **kwargs: Additional arguments for the objective constructor

    Returns:
        Objective function instance

    Raises:
        ValueError: If objective_type is unknown
    """
    objective_type = objective_type.lower().replace("-", "_")

    if objective_type in ("minimize_ttft", "minimize_ttft", "ttft", "latency"):
        return MinimizeTTFT(**kwargs)
    elif objective_type in ("maximize_throughput", "maximize_tps", "throughput", "tps"):
        return MaximizeThroughput(**kwargs)
    elif objective_type in ("minimize_total_time", "total_time"):
        return MinimizeTotalTime(**kwargs)
    elif objective_type in ("balanced", "balanced_objective"):
        return BalancedObjective(**kwargs)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
