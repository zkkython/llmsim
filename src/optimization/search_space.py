"""
Search space definition and parameter generation for optimization
"""

from typing import Iterator, List

from src.arch.config import ScheduleConfig
from src.arch.model_type import ForwardMode
from src.optimization.config import SearchSpaceConfig
from src.optimization.constraints import (
    Constraint,
    DivisibilityConstraint,
    LessThanOrEqualConstraint,
    ProductConstraint,
)


class SearchSpace:
    """
    Defines the search space for parameter optimization.

    Manages parameter ranges and validates configurations against constraints.
    """

    def __init__(
        self,
        config: SearchSpaceConfig,
        max_seqlen: int,
        is_moe_model: bool = False,
    ):
        """
        Initialize search space

        Args:
            config: Search space configuration
            max_seqlen: Fixed maximum sequence length (used for constraint checking)
            is_moe_model: Whether the model uses MoE (affects EP constraints)
        """
        self.config = config
        self.max_seqlen = max_seqlen
        self.is_moe_model = is_moe_model
        self._constraints: List[Constraint] = []
        self._build_constraints()

    def _build_constraints(self) -> None:
        """Build the list of constraints for this search space"""
        # max_seqlen must be divisible by tp_size
        self._constraints.append(
            DivisibilityConstraint(
                dividend_attr="max_seqlen",
                divisor_attr="tp_size",
                description="max_seqlen % tp_size == 0",
            )
        )

        # batch_size must be divisible by tp_size when batch_size > tp_size
        self._constraints.append(
            DivisibilityConstraint(
                dividend_attr="batch_size",
                divisor_attr="tp_size",
                description="batch_size % tp_size == 0",
            )
        )

        # world_size constraint: tp_size * dp_size = world_size (if specified)
        if self.config.world_size is not None:
            self._constraints.append(
                ProductConstraint(
                    factor_attrs=["tp_size", "dp_size"],
                    target_attr=f"={self.config.world_size}",
                    description=f"tp_size * dp_size = {self.config.world_size}",
                )
            )

        # EP constraint: ep_size <= dp_size (for MoE models)
        if self.is_moe_model:
            self._constraints.append(
                LessThanOrEqualConstraint(
                    left_attr="ep_size",
                    right_attr="dp_size",
                    description="ep_size <= dp_size",
                )
            )

    def get_constraints(self) -> List[Constraint]:
        """Get all constraints"""
        return self._constraints.copy()

    def validate(self, schedule_config: ScheduleConfig) -> tuple[bool, List[str]]:
        """
        Validate a configuration against all constraints

        Args:
            schedule_config: Configuration to validate

        Returns:
            Tuple of (is_valid, list of violation messages)
        """
        violations = []
        context = {"max_seqlen": self.max_seqlen}

        for constraint in self._constraints:
            if not constraint.check(schedule_config, **context):
                violations.append(
                    constraint.get_violation_message(schedule_config, **context)
                )

        return len(violations) == 0, violations

    def is_valid(self, schedule_config: ScheduleConfig) -> bool:
        """Quick check if configuration is valid"""
        valid, _ = self.validate(schedule_config)
        return valid

    def iterate_all(self) -> Iterator[ScheduleConfig]:
        """
        Iterate over all valid configurations in the search space

        Yields:
            ScheduleConfig: Valid configuration
        """
        tp_values = self.config.get_tp_values()
        dp_values = self.config.get_dp_values()
        ep_values = self.config.get_ep_values()
        batch_values = self.config.get_batch_size_values()
        mode_values = self.config.get_mode_values()

        iteration = 0
        for tp in tp_values:
            for dp in dp_values:
                for ep in ep_values:
                    for batch in batch_values:
                        for mode_str in mode_values:
                            # Skip EP > 1 for non-MoE models
                            if not self.is_moe_model and ep > 1:
                                continue

                            mode = (
                                ForwardMode.EXTEND
                                if mode_str == "extend"
                                else ForwardMode.DECODE
                            )

                            config = ScheduleConfig(
                                tp_size=tp,
                                dp_size=dp,
                                ep_size=ep,
                                batch_size=batch,
                                max_seqlen=self.max_seqlen,
                                mode=mode,
                            )

                            if self.is_valid(config):
                                iteration += 1
                                yield config

    def get_valid_configs(self) -> List[ScheduleConfig]:
        """Get all valid configurations as a list"""
        return list(self.iterate_all())

    def get_search_space_size(self) -> int:
        """Get the total number of valid configurations"""
        return len(self.get_valid_configs())

    def filter_by_memory(
        self,
        model_memory_gb: float,
        gpu_memory_gb: float,
        num_gpus: int,
        safety_factor: float = 0.9,
    ) -> List[ScheduleConfig]:
        """
        Filter configurations by memory constraint

        Args:
            model_memory_gb: Estimated model memory in GB
            gpu_memory_gb: GPU memory per device in GB
            num_gpus: Number of GPUs
            safety_factor: Safety factor for memory usage

        Returns:
            List of valid configurations that fit in memory
        """
        available_memory = gpu_memory_gb * num_gpus * safety_factor

        valid_configs = []
        for config in self.iterate_all():
            # Adjust memory estimate based on parallelism
            adjusted_memory = model_memory_gb / (config.tp_size * config.dp_size)
            if adjusted_memory <= available_memory:
                valid_configs.append(config)

        return valid_configs
