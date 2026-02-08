"""
Optimization results data structures
"""

from dataclasses import dataclass, field
from typing import Dict, List

from src.arch.config import ScheduleConfig


@dataclass
class OptimizationStep:
    """Single step in the optimization process"""

    iteration: int
    config: ScheduleConfig
    metrics: Dict[str, float]
    score: float


@dataclass
class OptimizationResult:
    """Result of an optimization run"""

    best_config: ScheduleConfig
    best_metrics: Dict[str, float]
    optimization_history: List[OptimizationStep] = field(default_factory=list)
    total_evaluations: int = 0
    total_time_seconds: float = 0.0
    search_space_size: int = 0

    def get_best_score(self) -> float:
        """Get the best score achieved"""
        if self.optimization_history:
            return min(step.score for step in self.optimization_history)
        return float("inf")

    def get_improvement_history(self) -> List[tuple]:
        """Get list of (iteration, score) tuples showing improvement over time"""
        if not self.optimization_history:
            return []

        history = []
        best_so_far = float("inf")
        for step in self.optimization_history:
            best_so_far = min(best_so_far, step.score)
            history.append((step.iteration, best_so_far))
        return history


@dataclass
class SensitivityAnalysisResult:
    """Result of parameter sensitivity analysis"""

    param_name: str
    param_values: List
    scores: List[float]
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def get_most_sensitive_range(self) -> tuple:
        """Return the parameter range with highest score variance"""
        if len(self.scores) < 2:
            return None, None

        max_variance = 0
        best_start = 0
        best_end = 1

        for i in range(len(self.scores) - 1):
            variance = abs(self.scores[i + 1] - self.scores[i])
            if variance > max_variance:
                max_variance = variance
                best_start = i
                best_end = i + 1

        return self.param_values[best_start], self.param_values[best_end]


@dataclass
class RecommendedConfig:
    """Recommended configuration with explanation"""

    config: ScheduleConfig
    metrics: Dict[str, float]
    priority: str  # "latency", "throughput", "balanced"
    explanation: str = ""
