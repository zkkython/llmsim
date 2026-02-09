"""
Optimization service - main interface for parameter optimization
"""

from typing import List, Optional, Union

from src.arch.config import ModelConfig, ScheduleConfig
from src.hardware.hardware_config import HardwareConfig
from src.optimization.config import SearchSpaceConfig
from src.optimization.evaluator import PerformanceEvaluator
from src.optimization.objective import (
    BaseObjective,
    create_objective,
)
from src.optimization.optimizers.base import BaseOptimizer
from src.optimization.optimizers.grid_search import GridSearchOptimizer
from src.optimization.results import (
    OptimizationResult,
    RecommendedConfig,
    SensitivityAnalysisResult,
)
from src.optimization.search_space import SearchSpace


class OptimizationService:
    """
    Main service for parameter optimization

    Provides a unified interface for optimizing LLM inference parameters.
    """

    def __init__(self):
        """Initialize optimization service"""
        pass

    def optimize(
        self,
        model_config: ModelConfig,
        hardware_config: HardwareConfig,
        max_seqlen: int,
        search_space_config: SearchSpaceConfig,
        objective_type: str = "maximize_tps",
        optimizer_type: str = "grid_search",
        parallel_workers: int = 1,
        max_evaluations: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Run parameter optimization

        Args:
            model_config: Model configuration
            hardware_config: Hardware configuration
            max_seqlen: Fixed maximum sequence length
            search_space_config: Search space configuration
            objective_type: Type of objective ("minimize_ttft", "maximize_tps", "balanced")
            optimizer_type: Type of optimizer ("grid_search")
            parallel_workers: Number of parallel workers
            max_evaluations: Maximum number of evaluations (None for unlimited)

        Returns:
            Optimization result
        """
        # Determine if model uses MoE
        is_moe_model = self._is_moe_model(model_config)

        # Create search space
        search_space = SearchSpace(
            config=search_space_config, max_seqlen=max_seqlen, is_moe_model=is_moe_model
        )

        # Create objective function
        objective = create_objective(objective_type)

        # Create evaluator
        evaluator = PerformanceEvaluator(
            model_config=model_config, hardware_config=hardware_config
        )

        # Create optimizer
        optimizer = self._create_optimizer(
            optimizer_type=optimizer_type,
            search_space=search_space,
            objective=objective,
            parallel_workers=parallel_workers,
            max_evaluations=max_evaluations,
        )

        # Run optimization
        return optimizer.optimize(evaluator)

    def get_recommended_config(
        self,
        model_config: ModelConfig,
        hardware_config: HardwareConfig,
        max_seqlen: int,
        priority: str = "balanced",
        world_size: Optional[int] = None,
    ) -> RecommendedConfig:
        """
        Get recommended configuration for a given priority

        Args:
            model_config: Model configuration
            hardware_config: Hardware configuration
            max_seqlen: Fixed maximum sequence length
            priority: Priority mode ("latency", "throughput", "balanced")
            world_size: Total number of GPUs (constrains TP * DP)

        Returns:
            Recommended configuration with metrics
        """
        # Define search space based on priority
        if priority == "latency":
            # For low latency: smaller batch sizes, higher TP
            search_config = SearchSpaceConfig(
                tp_size=[1, 2, 4, 8],
                dp_size=[1, 2, 4, 8],
                ep_size=[1, 2, 4, 8] if self._is_moe_model(model_config) else [1],
                batch_size=[1, 2, 4, 8],
                mode="extend",
                world_size=world_size,
            )
            objective_type = "minimize_ttft"
        elif priority == "throughput":
            # For high throughput: larger batch sizes, balanced TP/DP
            search_config = SearchSpaceConfig(
                tp_size=[1, 2, 4, 8],
                dp_size=[1, 2, 4, 8],
                ep_size=[1, 2, 4, 8, 16] if self._is_moe_model(model_config) else [1],
                batch_size=[8, 16, 32, 64, 128],
                mode="extend",
                world_size=world_size,
            )
            objective_type = "maximize_tps"
        else:  # balanced
            # Balanced approach
            search_config = SearchSpaceConfig(
                tp_size=[1, 2, 4, 8],
                dp_size=[1, 2, 4, 8],
                ep_size=[1, 2, 4, 8] if self._is_moe_model(model_config) else [1],
                batch_size=[1, 2, 4, 8, 16, 32, 64],
                mode="extend",
                world_size=world_size,
            )
            objective_type = "balanced"

        # Run optimization
        result = self.optimize(
            model_config=model_config,
            hardware_config=hardware_config,
            max_seqlen=max_seqlen,
            search_space_config=search_config,
            objective_type=objective_type,
        )

        # Generate explanation
        explanation = self._generate_explanation(result, priority)

        return RecommendedConfig(
            config=result.best_config,
            metrics=result.best_metrics,
            priority=priority,
            explanation=explanation,
        )

    def analyze_sensitivity(
        self,
        model_config: ModelConfig,
        hardware_config: HardwareConfig,
        base_config: ScheduleConfig,
        param_name: str,
        param_range: Union[List, tuple],
        objective_type: str = "balanced",
    ) -> SensitivityAnalysisResult:
        """
        Analyze sensitivity of a parameter

        Args:
            model_config: Model configuration
            hardware_config: Hardware configuration
            base_config: Base configuration to vary from
            param_name: Name of parameter to analyze (e.g., "tp_size", "batch_size")
            param_range: Range of values to test
            objective_type: Type of objective

        Returns:
            Sensitivity analysis result
        """
        evaluator = PerformanceEvaluator(
            model_config=model_config, hardware_config=hardware_config
        )

        objective = create_objective(objective_type)

        # Convert range to list
        if isinstance(param_range, tuple):
            param_values = list(range(param_range[0], param_range[1] + 1))
        else:
            param_values = list(param_range)

        scores = []
        metrics = {"ttft": [], "throughput": [], "total_time": []}

        print(
            f"Analyzing sensitivity of {param_name} over {len(param_values)} values..."
        )

        for value in param_values:
            # Create modified config
            config = ScheduleConfig(
                batch_size=base_config.batch_size,
                max_seqlen=base_config.max_seqlen,
                mode=base_config.mode,
                tp_size=base_config.tp_size,
                dp_size=base_config.dp_size,
                ep_size=base_config.ep_size,
            )
            setattr(config, param_name, value)

            # Evaluate
            perf = evaluator.evaluate(config)
            if perf is not None:
                score = objective.evaluate(perf)
                scores.append(score)
                metrics["ttft"].append(perf.get_ttft_or_tpot())
                metrics["throughput"].append(perf.get_throughput())
                metrics["total_time"].append(perf.total_time)
            else:
                scores.append(float("inf"))
                metrics["ttft"].append(float("inf"))
                metrics["throughput"].append(0)
                metrics["total_time"].append(float("inf"))

        return SensitivityAnalysisResult(
            param_name=param_name,
            param_values=param_values,
            scores=scores,
            metrics=metrics,
        )

    def _is_moe_model(self, model_config: ModelConfig) -> bool:
        """Check if model uses Mixture of Experts"""
        model_type = getattr(model_config, "model_type", "")
        return "moe" in model_type.lower() or model_type == "deepseek_v3"

    def _create_optimizer(
        self,
        optimizer_type: str,
        search_space: SearchSpace,
        objective: BaseObjective,
        parallel_workers: int,
        max_evaluations: Optional[int] = None,
    ) -> BaseOptimizer:
        """Create optimizer instance"""
        optimizer_type = optimizer_type.lower()

        if optimizer_type == "grid_search":
            return GridSearchOptimizer(
                search_space=search_space,
                objective=objective,
                parallel_workers=parallel_workers,
                max_evaluations=max_evaluations,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _generate_explanation(self, result: OptimizationResult, priority: str) -> str:
        """Generate human-readable explanation of recommendation"""
        if not result.best_config:
            return "No valid configuration found."

        config = result.best_config
        metrics = result.best_metrics

        explanations = []
        explanations.append(f"Recommended configuration for '{priority}' priority:")
        explanations.append(
            f"  Tensor Parallel (TP): {config.tp_size} - "
            f"Distributes attention computation across {config.tp_size} GPUs"
        )
        explanations.append(
            f"  Data Parallel (DP): {config.dp_size} - "
            f"Processes {config.dp_size} batches in parallel"
        )
        if config.ep_size > 1:
            explanations.append(
                f"  Expert Parallel (EP): {config.ep_size} - "
                f"Distributes MoE experts across {config.ep_size} partitions"
            )
        explanations.append(
            f"  Batch Size: {config.batch_size} - " f"Optimal for {priority} workload"
        )

        if "ttft_ms" in metrics:
            explanations.append("\nExpected Performance:")
            explanations.append(
                f"  Time To First Token (TTFT): {metrics['ttft_ms']:.2f} ms"
            )
            explanations.append(
                f"  Throughput: {metrics.get('throughput_tps', 0):.2f} tokens/sec"
            )

        return "\n".join(explanations)


def get_recommended_config(
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    max_seqlen: int,
    priority: str = "balanced",
    world_size: Optional[int] = None,
) -> RecommendedConfig:
    """
    Convenience function to get recommended configuration

    Args:
        model_config: Model configuration
        hardware_config: Hardware configuration
        max_seqlen: Fixed maximum sequence length
        priority: Priority mode ("latency", "throughput", "balanced")
        world_size: Total number of GPUs

    Returns:
        Recommended configuration
    """
    service = OptimizationService()
    return service.get_recommended_config(
        model_config=model_config,
        hardware_config=hardware_config,
        max_seqlen=max_seqlen,
        priority=priority,
        world_size=world_size,
    )
