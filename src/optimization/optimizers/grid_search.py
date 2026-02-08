"""
Grid search optimizer - exhaustive search over all valid configurations
"""

import time
from typing import Optional

from src.optimization.evaluator import PerformanceEvaluator
from src.optimization.objective import BaseObjective
from src.optimization.optimizers.base import BaseOptimizer
from src.optimization.results import OptimizationResult
from src.optimization.search_space import SearchSpace


class GridSearchOptimizer(BaseOptimizer):
    """
    Grid search optimizer that evaluates all valid configurations

    This optimizer is exhaustive and deterministic, suitable for small
    to medium-sized search spaces.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: BaseObjective,
        parallel_workers: int = 1,
        max_evaluations: Optional[int] = None,
    ):
        """
        Initialize grid search optimizer

        Args:
            search_space: Search space to explore
            objective: Objective function to optimize
            parallel_workers: Number of parallel evaluation workers
            max_evaluations: Maximum number of configurations to evaluate
        """
        super().__init__(
            search_space=search_space,
            objective=objective,
            parallel_workers=parallel_workers,
        )
        self.max_evaluations = max_evaluations

    def optimize(self, evaluator: PerformanceEvaluator) -> OptimizationResult:
        """
        Run grid search optimization

        Args:
            evaluator: Performance evaluator

        Returns:
            Optimization result
        """
        start_time = time.time()
        self.reset()

        print(
            f"Starting grid search over {self.search_space.get_search_space_size()} configurations..."
        )

        evaluation_count = 0

        for config in self.search_space.iterate_all():
            if self.max_evaluations and evaluation_count >= self.max_evaluations:
                print(f"Reached max evaluations limit ({self.max_evaluations})")
                break

            evaluation_count += 1

            # Evaluate configuration
            perf = evaluator.evaluate(config)

            if perf is None:
                print(
                    f"  [{evaluation_count}] Config failed: TP={config.tp_size}, DP={config.dp_size}, "
                    f"EP={config.ep_size}, BS={config.batch_size}"
                )
                continue

            # Calculate score and metrics
            score = self.objective.evaluate(perf)
            metrics = self.objective.get_metrics(perf)

            # Add standard metrics
            metrics["ttft_ms"] = perf.get_ttft_or_tpot()
            metrics["throughput_tps"] = perf.get_throughput()
            metrics["total_time_ms"] = perf.total_time

            # Record step
            self._record_step(config, score, metrics)

            # Update best
            is_best = self._update_best(config, score, metrics)

            status = "*** BEST ***" if is_best else ""
            print(
                f"  [{evaluation_count}] TP={config.tp_size}, DP={config.dp_size}, "
                f"EP={config.ep_size}, BS={config.batch_size}, Mode={config.mode.name}, "
                f"Score={score:.4f}, TTFT={metrics['ttft_ms']:.2f}ms, "
                f"TPS={metrics['throughput_tps']:.2f} {status}"
            )

        total_time = time.time() - start_time
        print(f"\nGrid search completed in {total_time:.2f}s")
        print(f"Total evaluations: {evaluation_count}")
        print(f"Best score: {self._best_score:.4f}")

        return self._create_result(total_time)
