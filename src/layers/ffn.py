from src.config.model_config import ModelConfig
from src.server_args import ServerArgs


class FFN:
    def __init__(self, serverArgs: ServerArgs, config: ModelConfig, layer_idx: int):
        self.serverArgs = serverArgs
        self.config = config
        self.layer_idx = layer_idx

    @staticmethod
    def create(serverArgs: ServerArgs, config: ModelConfig, layer_idx: int) -> "FFN":
        # 1. 优先处理模型特定的 FFN
        if config.model_type == "qwen3_next":
            return QwenNextFFN(serverArgs, config, layer_idx)
        # DeepSeek consider first_k_dense_replace, qwen3 moe and qwen3 next not
        if config.model_type == "deepseek_v3":
            return DeepSeekV3FFN(serverArgs, config, layer_idx)

        # 否则根据 moe_config 决定
        if config.moe_config and config.moe_config.num_routed_experts > 1:
            return MoE(serverArgs, config, layer_idx)
        return DenseMLP(serverArgs, config, layer_idx)

    def weights_size(self):
        return 0

    def layer_idx_ffn_state(self) -> str:

        if self.config.moe_config and self.config.moe_config.num_routed_experts > 1:
            return "MOE"
        return "DenseMLP"


class DenseMLP(FFN):

    def __init__(self, serverArgs: ServerArgs, config: ModelConfig, layer_idx: int):
        super().__init__(serverArgs, config, layer_idx)

    def weights_size(self):
        cfg = self.config.moe_config
        if not cfg:
            return 0

        hidden_size = self.config.hidden_size
        tp_size = self.serverArgs.tp_size if self.serverArgs.tp_size > 0 else 1
        intermediate_size = cfg.intermediate_size

        # Dense MLP TP 切分策略：
        # gate_proj, up_proj 按列切分 (intermediate_size / tp_size)
        # down_proj 按行切分 (输入维度 intermediate_size / tp_size)
        w_gate_up = 2 * hidden_size * (intermediate_size // tp_size)
        w_down = (intermediate_size // tp_size) * hidden_size
        w = w_gate_up + w_down

        if self.serverArgs.use_fp8_gemm:
            return w
        return 2 * w


class MoE(FFN):

    def __init__(self, serverArgs: ServerArgs, config: ModelConfig, layer_idx: int):
        super().__init__(serverArgs, config, layer_idx)
        # whether has shared experts
        self.shared_experts = (
            config.moe_config.num_shared_experts if config.moe_config else 0
        )
        self.ep_size = (
            serverArgs.ep_size if serverArgs.ep_size > 0 else serverArgs.world_size
        )

    def single_expert_weights_size(self):
        cfg = self.config.moe_config
        if not cfg:
            return 0

        hidden_size = self.config.hidden_size
        ep_tp_size = (
            self.serverArgs.world_size // self.serverArgs.ep_size
            if self.serverArgs.ep_size > 0
            else 1
        )
        intermediate_size = cfg.intermediate_size

        # MoE 专家内部 TP 切分（如果专家很大，TP 会切分专家权重）
        w_gate_up = 2 * hidden_size * (intermediate_size // ep_tp_size)
        w_down = (intermediate_size // ep_tp_size) * hidden_size
        w = w_gate_up + w_down

        if self.serverArgs.use_fp8_gemm:
            return w
        return 2 * w

    def single_shared_mlp_size(self):
        cfg = self.config.moe_config
        if not cfg:
            return 0

        hidden_size = self.config.hidden_size
        tp_size = self.serverArgs.tp_size if self.serverArgs.tp_size > 0 else 1
        # if deepseek, put shared expert tp_size = 1
        tp_size = (
            1 if self.config.model_type in ["deepseek_v3", "deepseek_v32"] else tp_size
        )
        intermediate_size = cfg.intermediate_size

        # Shared MLP TP 切分（有些实现不切分共享专家，这里暂按 TP 切分统计）
        w_gate_up = 2 * hidden_size * (intermediate_size // tp_size)
        w_down = (intermediate_size // tp_size) * hidden_size
        w = w_gate_up + w_down

        if self.serverArgs.use_fp8_gemm:
            return w
        return 2 * w

    def weights_size(self):
        cfg = self.config.moe_config
        if not cfg:
            return 0
        # ep distributed, router expert will be divided by ep size, shared expert will copy on every gpu
        # num_experts = (cfg.num_routed_experts / self.ep_size) + self.shared_experts
        # print(
        #     f"num_experts: {num_experts}， single expert weights size: {self.single_expert_weights_size()/(1024**2)}MB"
        # )
        return (
            cfg.num_routed_experts / self.ep_size
        ) * self.single_expert_weights_size() + self.shared_experts * self.single_shared_mlp_size()


class QwenNextFFN(MoE):
    def __init__(self, serverArgs: ServerArgs, config: ModelConfig, layer_idx: int):
        super().__init__(serverArgs, config, layer_idx)
        # 这里可以解析 Qwen Next 特有的字段，比如 shared_expert_intermediate_size
        # 实际应用中可能需要从 config 中提取更多特定值

    def weights_size(self):
        # 实现 Qwen Next 特有的逻辑（如果需要的话）
        # 如果逻辑和 MoE 一样，可以直接复用父类，或者在这里修改公式
        return super().weights_size()


class DeepSeekV3FFN(FFN):
    def __init__(self, serverArgs: ServerArgs, config: ModelConfig, layer_idx: int):
        super().__init__(serverArgs, config, layer_idx)
        # 2. 逻辑：如果当前层索引小于 first_k_dense_replace，强制使用 DenseMLP
        if (
            config.first_k_dense_replace > 0
            and layer_idx < config.first_k_dense_replace
        ):
            self.ffn = DenseMLP(serverArgs, config, layer_idx)
        else:
            self.ffn = MoE(serverArgs, config, layer_idx)

    def weights_size(self):
        # 实现 DeepSeek V3 特有的逻辑（如果需要的话）
        return self.ffn.weights_size()

    def layer_idx_ffn_state(self) -> str:
        assert (
            self.config.first_k_dense_replace > 0
        ), "DeepSeek V3 requires first_k_dense_replace to be greater than 0"
        # print(
        #     f"DeepSeek V3 layer {self.layer_idx} first_k_dense_replace: {self.config.first_k_dense_replace}"
        # )
        # print(f"moe config: {self.config.moe_config}")
        if (
            self.config.first_k_dense_replace > 0
            and self.layer_idx < self.config.first_k_dense_replace
        ):
            return "DenseMLP"
        if self.config.moe_config and self.config.moe_config.num_routed_experts > 1:
            return "MOE"
        return "DenseMLP"
