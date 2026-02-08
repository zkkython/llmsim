"""
LLMSim Parameter Optimization Module

This module provides automatic parameter optimization for LLM inference
to achieve optimal TTFT (Time To First Token) and TPS (Tokens Per Second).

Example usage:
    from src.optimization import OptimizationService, SearchSpaceConfig

    service = OptimizationService()
    result = service.optimize(
        model_config=model_config,
        hardware_config=hardware_config,
        max_seqlen=4096,
        search_space_config=SearchSpaceConfig(
            tp_size=[1, 2, 4, 8],
            dp_size=[1, 2, 4, 8],
            batch_size=[1, 2, 4, 8, 16],
            mode="extend"
        ),
        objective_type="maximize_tps"
    )

    print(f"Best config: TP={result.best_config.tp_size}, DP={result.best_config.dp_size}")
    print(f"Best TTFT: {result.best_metrics['ttft_ms']:.2f} ms")
    print(f"Best TPS: {result.best_metrics['throughput_tps']:.2f}")
"""

# Core classes
from src.optimization.config import SearchSpaceConfig
from src.optimization.constraints import (
    Constraint,
    DivisibilityConstraint,
    LessThanOrEqualConstraint,
    MemoryConstraint,
    ProductConstraint,
    RangeConstraint,
)
from src.optimization.evaluator import PerformanceEvaluator
from src.optimization.objective import (
    BalancedObjective,
    BaseObjective,
    MaximizeThroughput,
    MinimizeTTFT,
    MultiObjective,
    create_objective,
)
from src.optimization.optimizers import BaseOptimizer, GridSearchOptimizer
from src.optimization.results import (
    OptimizationResult,
    OptimizationStep,
    RecommendedConfig,
    SensitivityAnalysisResult,
)
from src.optimization.search_space import SearchSpace
from src.optimization.service import (
    OptimizationService,
    get_recommended_config,
)

__all__ = [
    # Core service
    "OptimizationService",
    "get_recommended_config",
    # Configuration
    "SearchSpaceConfig",
    "SearchSpace",
    # Objectives
    "BaseObjective",
    "MinimizeTTFT",
    "MaximizeThroughput",
    "MultiObjective",
    "BalancedObjective",
    "create_objective",
    # Optimizers
    "BaseOptimizer",
    "GridSearchOptimizer",
    # Evaluator
    "PerformanceEvaluator",
    # Constraints
    "Constraint",
    "DivisibilityConstraint",
    "ProductConstraint",
    "LessThanOrEqualConstraint",
    "MemoryConstraint",
    "RangeConstraint",
    # Results
    "OptimizationResult",
    "OptimizationStep",
    "RecommendedConfig",
    "SensitivityAnalysisResult",
]
