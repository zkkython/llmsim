"""
Base optimizer class for parameter optimization
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.arch.config import ScheduleConfig
from src.optimization.evaluator import PerformanceEvaluator
from src.optimization.objective import BaseObjective
from src.optimization.results import OptimizationResult, OptimizationStep
from src.optimization.search_space import SearchSpace


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms

    All optimizers should inherit from this class and implement the optimize method.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: BaseObjective,
        parallel_workers: int = 1,
        early_stop_patience: Optional[int] = None,
    ):
        """
        Initialize optimizer

        Args:
            search_space: Search space to explore
            objective: Objective function to optimize
            parallel_workers: Number of parallel evaluation workers
            early_stop_patience: Number of iterations without improvement before stopping
        """
        self.search_space = search_space
        self.objective = objective
        self.parallel_workers = parallel_workers
        self.early_stop_patience = early_stop_patience

        self._best_config: Optional[ScheduleConfig] = None
        self._best_score: float = float("inf")
        self._optimization_history: List[OptimizationStep] = []
        self._iteration: int = 0

    @abstractmethod
    def optimize(self, evaluator: PerformanceEvaluator) -> OptimizationResult:
        """
        Run the optimization process

        Args:
            evaluator: Performance evaluator

        Returns:
            Optimization result containing best config and history
        """
        pass

    def get_best_params(self) -> Optional[ScheduleConfig]:
        """Get the best configuration found so far"""
        return self._best_config

    def get_best_score(self) -> float:
        """Get the best score achieved so far"""
        return self._best_score

    def get_optimization_history(self) -> List[OptimizationStep]:
        """Get the history of optimization steps"""
        return self._optimization_history.copy()

    def _update_best(self, config: ScheduleConfig, score: float, metrics: dict) -> bool:
        """
        Update the best configuration if score is better

        Args:
            config: Configuration evaluated
            score: Score achieved (lower is better)
            metrics: Detailed metrics

        Returns:
            True if this is a new best score
        """
        if score < self._best_score:
            self._best_score = score
            self._best_config = config
            return True
        return False

    def _record_step(
        self, config: ScheduleConfig, score: float, metrics: dict
    ) -> OptimizationStep:
        """
        Record an optimization step

        Args:
            config: Configuration evaluated
            score: Score achieved
            metrics: Detailed metrics

        Returns:
            OptimizationStep recorded
        """
        self._iteration += 1
        step = OptimizationStep(
            iteration=self._iteration, config=config, metrics=metrics, score=score
        )
        self._optimization_history.append(step)
        return step

    def _should_stop_early(self) -> bool:
        """
        Check if optimization should stop early

        Returns True if no improvement for patience iterations
        """
        if self.early_stop_patience is None:
            return False

        if len(self._optimization_history) < self.early_stop_patience:
            return False

        # Check if there was improvement in the last patience iterations
        recent_steps = self._optimization_history[-self.early_stop_patience :]
        best_in_recent = min(step.score for step in recent_steps)

        # Compare with best before the recent window
        if len(self._optimization_history) > self.early_stop_patience:
            earlier_best = min(
                step.score
                for step in self._optimization_history[: -self.early_stop_patience]
            )
        else:
            earlier_best = self._best_score

        # Stop if no improvement over the patience window
        return best_in_recent >= earlier_best

    def _create_result(self, total_time_seconds: float) -> OptimizationResult:
        """
        Create an optimization result from the current state

        Args:
            total_time_seconds: Total optimization time

        Returns:
            OptimizationResult
        """
        best_metrics = {}
        if self._best_config and self._optimization_history:
            # Find the step with the best score
            for step in self._optimization_history:
                if step.score == self._best_score:
                    best_metrics = step.metrics
                    break

        return OptimizationResult(
            best_config=self._best_config or ScheduleConfig(),
            best_metrics=best_metrics,
            optimization_history=self._optimization_history,
            total_evaluations=self._iteration,
            total_time_seconds=total_time_seconds,
            search_space_size=self.search_space.get_search_space_size(),
        )

    def reset(self) -> None:
        """Reset the optimizer state"""
        self._best_config = None
        self._best_score = float("inf")
        self._optimization_history = []
        self._iteration = 0
