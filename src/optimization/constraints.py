"""
Constraint classes for parameter validation
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.arch.config import ScheduleConfig


class Constraint(ABC):
    """Abstract base class for constraints"""

    @abstractmethod
    def check(self, config: ScheduleConfig, **context) -> bool:
        """Check if the configuration satisfies this constraint"""
        pass

    @abstractmethod
    def get_violation_message(self, config: ScheduleConfig, **context) -> str:
        """Get a human-readable message explaining the violation"""
        pass


class DivisibilityConstraint(Constraint):
    """Constraint that checks if one value is divisible by another"""

    def __init__(
        self, dividend_attr: str, divisor_attr: str, description: Optional[str] = None
    ):
        """
        Args:
            dividend_attr: Attribute name of the dividend (e.g., "max_seqlen")
            divisor_attr: Attribute name of the divisor (e.g., "tp_size")
            description: Optional description for error messages
        """
        self.dividend_attr = dividend_attr
        self.divisor_attr = divisor_attr
        self.description = description or f"{dividend_attr} % {divisor_attr} == 0"

    def check(self, config: ScheduleConfig, **context) -> bool:
        dividend = getattr(config, self.dividend_attr, None)
        divisor = getattr(config, self.divisor_attr, None)

        if dividend is None:
            dividend = context.get(self.dividend_attr)
        if divisor is None:
            divisor = context.get(self.divisor_attr)

        if dividend is None or divisor is None:
            return True  # Cannot check, assume valid

        if divisor == 0:
            return False

        return dividend % divisor == 0

    def get_violation_message(self, config: ScheduleConfig, **context) -> str:
        dividend = getattr(config, self.dividend_attr, None)
        divisor = getattr(config, self.divisor_attr, None)

        if dividend is None:
            dividend = context.get(self.dividend_attr, "unknown")
        if divisor is None:
            divisor = context.get(self.divisor_attr, "unknown")

        return f"{self.dividend_attr} ({dividend}) must be divisible by {self.divisor_attr} ({divisor})"


class ProductConstraint(Constraint):
    """Constraint that checks if the product of factors equals a target value"""

    def __init__(
        self,
        factor_attrs: List[str],
        target_attr: str,
        description: Optional[str] = None,
    ):
        """
        Args:
            factor_attrs: List of attribute names to multiply
            target_attr: Attribute name of the target product (or fixed value if starts with "=")
            description: Optional description for error messages
        """
        self.factor_attrs = factor_attrs
        self.target_attr = target_attr
        self.description = description

    def check(self, config: ScheduleConfig, **context) -> bool:
        product = 1
        for attr in self.factor_attrs:
            value = getattr(config, attr, None)
            if value is None:
                value = context.get(attr, 1)
            product *= value

        if self.target_attr.startswith("="):
            target = int(self.target_attr[1:])
        else:
            target = getattr(config, self.target_attr, None)
            if target is None:
                target = context.get(self.target_attr)

        if target is None:
            return True

        return product == target

    def get_violation_message(self, config: ScheduleConfig, **context) -> str:
        product = 1
        factor_values = []
        for attr in self.factor_attrs:
            value = getattr(config, attr, None)
            if value is None:
                value = context.get(attr, 1)
            factor_values.append(f"{attr}={value}")
            product *= value

        if self.target_attr.startswith("="):
            target = int(self.target_attr[1:])
            target_str = str(target)
        else:
            target = getattr(config, self.target_attr, None)
            if target is None:
                target = context.get(self.target_attr, "unknown")
            target_str = f"{self.target_attr}={target}"

        factors_str = " * ".join(factor_values)
        return f"Product constraint violated: {factors_str} = {product}, expected {target_str}"


class LessThanOrEqualConstraint(Constraint):
    """Constraint that checks if one value is less than or equal to another"""

    def __init__(
        self, left_attr: str, right_attr: str, description: Optional[str] = None
    ):
        """
        Args:
            left_attr: Attribute name of the left side value
            right_attr: Attribute name of the right side value
            description: Optional description for error messages
        """
        self.left_attr = left_attr
        self.right_attr = right_attr
        self.description = description or f"{left_attr} <= {right_attr}"

    def check(self, config: ScheduleConfig, **context) -> bool:
        left = getattr(config, self.left_attr, None)
        right = getattr(config, self.right_attr, None)

        if left is None:
            left = context.get(self.left_attr)
        if right is None:
            right = context.get(self.right_attr)

        if left is None or right is None:
            return True

        return left <= right

    def get_violation_message(self, config: ScheduleConfig, **context) -> str:
        left = getattr(config, self.left_attr, None)
        right = getattr(config, self.right_attr, None)

        if left is None:
            left = context.get(self.left_attr, "unknown")
        if right is None:
            right = context.get(self.right_attr, "unknown")

        return f"{self.left_attr} ({left}) must be <= {self.right_attr} ({right})"


class MemoryConstraint(Constraint):
    """Constraint that checks if model fits in GPU memory"""

    def __init__(self, max_memory_gb: float, num_gpus: int, safety_factor: float = 0.9):
        """
        Args:
            max_memory_gb: Maximum memory per GPU in GB
            num_gpus: Number of GPUs
            safety_factor: Safety factor for memory usage (default 0.9 = 90%)
        """
        self.max_memory_gb = max_memory_gb
        self.num_gpus = num_gpus
        self.safety_factor = safety_factor

    def check(self, config: ScheduleConfig, **context) -> bool:
        model_memory_gb = context.get("model_memory_gb")
        if model_memory_gb is None:
            return True  # Cannot check without model memory info

        available_memory = self.max_memory_gb * self.num_gpus * self.safety_factor
        return model_memory_gb <= available_memory

    def get_violation_message(self, config: ScheduleConfig, **context) -> str:
        model_memory_gb = context.get("model_memory_gb", "unknown")
        available = self.max_memory_gb * self.num_gpus * self.safety_factor
        return f"Model memory ({model_memory_gb} GB) exceeds available memory ({available:.1f} GB)"


class RangeConstraint(Constraint):
    """Constraint that checks if a value is within a range"""

    def __init__(
        self,
        attr: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ):
        """
        Args:
            attr: Attribute name to check
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
        """
        self.attr = attr
        self.min_value = min_value
        self.max_value = max_value

    def check(self, config: ScheduleConfig, **context) -> bool:
        value = getattr(config, self.attr, None)
        if value is None:
            value = context.get(self.attr)
        if value is None:
            return True

        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True

    def get_violation_message(self, config: ScheduleConfig, **context) -> str:
        value = getattr(config, self.attr, None)
        if value is None:
            value = context.get(self.attr, "unknown")

        if self.min_value is not None and self.max_value is not None:
            return f"{self.attr} ({value}) must be in range [{self.min_value}, {self.max_value}]"
        elif self.min_value is not None:
            return f"{self.attr} ({value}) must be >= {self.min_value}"
        else:
            return f"{self.attr} ({value}) must be <= {self.max_value}"
