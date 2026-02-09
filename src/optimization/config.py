"""
Search space configuration for optimization
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class SearchSpaceConfig:
    """
    Search space configuration for parameter optimization

    Note: max_seqlen is a fixed input parameter, not an optimization parameter.
    It is used for constraint checking (e.g., max_seqlen % tp_size == 0).
    """

    # Tensor Parallel size: single value, range (min, max), or list of values
    tp_size: Union[int, tuple, List[int]] = field(default_factory=lambda: [1, 2, 4, 8])

    # Data Parallel size: single value, range (min, max), or list of values
    dp_size: Union[int, tuple, List[int]] = field(default_factory=lambda: [1, 2, 4, 8])

    # Expert Parallel size: single value, range (min, max), list of values, or None (auto-infer)
    ep_size: Union[int, tuple, List[int], None] = None

    # Batch size: single value, range (min, max), or list of values
    batch_size: Union[int, tuple, List[int]] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64]
    )

    # Forward mode: single mode or list of modes
    mode: Union[str, List[str]] = "extend"

    # Fixed world size (total GPU count): if set, constrains tp_size * dp_size = world_size
    world_size: Optional[int] = None

    @staticmethod
    def from_dict(data: dict) -> "SearchSpaceConfig":
        """Create SearchSpaceConfig from a dictionary"""
        config = SearchSpaceConfig()

        if "tp_size" in data:
            config.tp_size = data["tp_size"]
        if "dp_size" in data:
            config.dp_size = data["dp_size"]
        if "ep_size" in data:
            config.ep_size = data["ep_size"]
        if "batch_size" in data:
            config.batch_size = data["batch_size"]
        if "mode" in data:
            config.mode = data["mode"]
        if "world_size" in data:
            config.world_size = data["world_size"]

        return config

    def get_tp_values(self) -> List[int]:
        """Get list of TP size values to search"""
        return self._normalize_to_list(self.tp_size)

    def get_dp_values(self) -> List[int]:
        """Get list of DP size values to search"""
        return self._normalize_to_list(self.dp_size)

    def get_ep_values(self) -> List[int]:
        """Get list of EP size values to search"""
        if self.ep_size is None:
            return [1]  # Default to 1 (no expert parallelism)
        return self._normalize_to_list(self.ep_size)

    def get_batch_size_values(self) -> List[int]:
        """Get list of batch size values to search"""
        return self._normalize_to_list(self.batch_size)

    def get_mode_values(self) -> List[str]:
        """Get list of mode values to search"""
        if isinstance(self.mode, str):
            return [self.mode]
        return self.mode

    @staticmethod
    def _normalize_to_list(value: Union[int, tuple, List[int]]) -> List[int]:
        """Normalize a value specification to a list of integers"""
        if isinstance(value, int):
            return [value]
        elif isinstance(value, tuple) and len(value) == 2:
            min_val, max_val = value
            # Generate powers of 2 within range
            result = []
            v = 1
            while v <= max_val:
                if v >= min_val:
                    result.append(v)
                v *= 2
            return result if result else [min_val]
        elif isinstance(value, list):
            return value
        else:
            raise ValueError(f"Cannot normalize value to list: {value}")

    def get_search_space_size(self) -> int:
        """Calculate the total size of the search space"""
        return (
            len(self.get_tp_values())
            * len(self.get_dp_values())
            * len(self.get_ep_values())
            * len(self.get_batch_size_values())
            * len(self.get_mode_values())
        )

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "ep_size": self.ep_size,
            "batch_size": self.batch_size,
            "mode": self.mode,
            "world_size": self.world_size,
        }
