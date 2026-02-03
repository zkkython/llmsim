from arch.config import ModelConfig, ScheduleConfig
from hardware.hardware_config import HardwareConfig


class NetworkComm:

    def __init__(
        self,
        hardware_config: "HardwareConfig",
        model_config: "ModelConfig",
        schedule_config: "ScheduleConfig",
    ):
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.schedule_config = schedule_config

    def size_of_bandwidth(self):

        return 0
