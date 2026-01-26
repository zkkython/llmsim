from abc import abstractmethod
from typing import Optional

from src.config.model_config import ModelConfig
from src.server_args import ServerArgs


class Attn:

    def __init__(self, serverArgs: ServerArgs, config: ModelConfig):
        self.serverArgs = serverArgs
        self.config = config

    @staticmethod
    def create(
        serverArgs: ServerArgs, config: ModelConfig, layer_idx: Optional[int]
    ) -> "Attn":
        from src.config.model_config import (
            HybridAttnConfig,
            LinearAttnConfig,
            MHAConfig,
            MLAConfig,
        )

        actual_config = config.attn_config

        # 处理混合模式逻辑
        if isinstance(actual_config, HybridAttnConfig):
            assert layer_idx is not None
            if layer_idx % actual_config.full_attention_interval == 0:
                actual_config = actual_config.full_attn_config
            else:
                actual_config = actual_config.linear_attn_config

        if isinstance(actual_config, MHAConfig):
            return MHAAttn(serverArgs, config)
        elif isinstance(actual_config, MLAConfig):
            return MLAAttn(serverArgs, config)
        elif isinstance(actual_config, LinearAttnConfig):
            return LinearAttn(serverArgs, config)
        return Attn(serverArgs, config)

    def weights_size(self):
        return 0

    @abstractmethod
    def attn_type(self):
        raise NotImplementedError
    
    @abstractmethod
    def kv_cache_factors(self) -> tuple[int, int]:
        """返回 (static_bytes, per_token_bytes)"""
        raise NotImplementedError


class MHAAttn(Attn):
    def __init__(self, serverArgs: ServerArgs, config: ModelConfig):
        super().__init__(serverArgs, config)

    def attn_type(self):
        return "MHA"

    def weights_size(self):
        from src.config.model_config import HybridAttnConfig, MHAConfig

        cfg = self.config.attn_config
        if isinstance(cfg, HybridAttnConfig):
            cfg = cfg.full_attn_config

        if not isinstance(cfg, MHAConfig):
            return 0

        hidden_size = self.config.hidden_size
        tp_size = self.serverArgs.tp_size if self.serverArgs.tp_size > 0 else 1

        # TP 切分：Q, K, V 通常按列切分（Head 维度除以 TP）
        # Output 投影通常按行切分（输入维度被切分）
        wq = hidden_size * (cfg.num_attention_heads // tp_size) * cfg.head_dim
        wk = hidden_size * (cfg.num_key_value_heads // tp_size) * cfg.head_dim
        wv = hidden_size * (cfg.num_key_value_heads // tp_size) * cfg.head_dim
        wo = (cfg.num_attention_heads // tp_size) * cfg.head_dim * hidden_size

        total = wq + wk + wv + wo
        if self.serverArgs.use_fp8_gemm:
            return total
        return 2 * total

    def kv_cache_factors(self) -> tuple[int, int]:
        from src.config.model_config import HybridAttnConfig, MHAConfig
    
        tp_size = self.serverArgs.tp_size if self.serverArgs.tp_size > 0 else 1
        cfg = self.config.attn_config
            
        if isinstance(cfg, HybridAttnConfig):
            cfg = cfg.full_attn_config
    
        if not isinstance(cfg, MHAConfig):
            return 0, 0
                
        # MHA KV Cache: 每个 token 存储 2 * num_kv_heads * head_dim
        per_token_size = (
            2 * (cfg.num_key_value_heads // tp_size) * cfg.head_dim
        )
        if not self.serverArgs.use_fp8_kv:
            per_token_size *= 2
                
        return 0, per_token_size
    

class MLAAttn(Attn):
    def __init__(self, serverArgs: ServerArgs, config: ModelConfig):
        super().__init__(serverArgs, config)

    def attn_type(self):
        return "MLA"

    def weights_size(self):
        from src.config.model_config import HybridAttnConfig, MLAConfig

        cfg = self.config.attn_config
        if isinstance(cfg, HybridAttnConfig):
            cfg = cfg.full_attn_config

        if not isinstance(cfg, MLAConfig):
            return 0

        hidden_size = self.config.hidden_size
        tp_size = self.serverArgs.tp_size if self.serverArgs.tp_size > 0 else 1

        # MLA TP 策略：
        # 1. wq_down, wkv_down 通常不切分
        # 2. wq_up, wkv_up 按列切分（Head 维度除以 TP）
        # 3. wo 按行切分
        wq_down = hidden_size * cfg.q_lora_rank
        wq_up = cfg.q_lora_rank * (cfg.num_attention_heads // tp_size) * cfg.qk_head_dim
        wq = wq_down + wq_up

        wkv_down = hidden_size * cfg.kv_lora_rank
        wkv_up = (
            cfg.kv_lora_rank
            * (cfg.num_attention_heads // tp_size)
            * (cfg.qk_nope_head_dim + cfg.v_head_dim)
        )
        wkv = wkv_down + wkv_up
        wo = (cfg.num_attention_heads // tp_size) * cfg.v_head_dim * hidden_size

        total = wq + wkv + wo
        if self.serverArgs.use_fp8_gemm:
            return total
        return 2 * total


    def kv_cache_factors(self) -> tuple[int, int]:
        from src.config.model_config import HybridAttnConfig, MLAConfig

        cfg = self.config.attn_config
        if isinstance(cfg, HybridAttnConfig):
            cfg = cfg.full_attn_config

        if not isinstance(cfg, MLAConfig):
            return 0, 0

        # MLA KV Cache (单层每个 token): kv_lora_rank + qk_rope_head_dim
        per_token_size = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        
        if not self.serverArgs.use_fp8_kv:
            per_token_size *= 2
            
        return 0, per_token_size


class LinearAttn(Attn):
    def __init__(self, serverArgs: ServerArgs, config: ModelConfig):
        super().__init__(serverArgs, config)

    def attn_type(self):
        return "LinearAttn"

    def weights_size(self):
        from src.config.model_config import HybridAttnConfig, LinearAttnConfig

        cfg = self.config.attn_config
        if isinstance(cfg, HybridAttnConfig):
            cfg = cfg.linear_attn_config

        if not isinstance(cfg, LinearAttnConfig):
            return 0

        hidden_size = self.config.hidden_size
        tp_size = self.serverArgs.tp_size if self.serverArgs.tp_size > 0 else 1

        # LinearAttn TP 策略：
        # 1. 投影矩阵 (wq, wk, wv, wz, wa, wb) 按列切分 (Head 维度除以 TP)
        # 2. 卷积权重通常在单卡处理或有特殊切分，这里按 Head 维度分片
        wq = hidden_size * (cfg.num_key_heads // tp_size) * cfg.key_head_dim
        wk = wq
        wv = hidden_size * (cfg.num_value_heads // tp_size) * cfg.value_head_dim
        wz = wv
        wa = hidden_size * (cfg.num_value_heads // tp_size)
        wb = wa
        s = wq + wk + wv + wz + wa + wb

        # 卷积权重 wconv：按 Head 维度分片
        wconv = (cfg.num_key_heads // tp_size) * cfg.key_head_dim * cfg.conv_kernel_dim
        wconv += (cfg.num_key_heads // tp_size) * cfg.key_head_dim * cfg.conv_kernel_dim
        wconv += (
            (cfg.num_value_heads // tp_size) * cfg.value_head_dim * cfg.conv_kernel_dim
        )

        if self.serverArgs.use_fp8_gemm:
            return s + wconv
        return 2 * s + wconv

    def kv_cache_factors(self) -> tuple[int, int]:
        from src.config.model_config import HybridAttnConfig, LinearAttnConfig

        cfg = self.config.attn_config
        if isinstance(cfg, HybridAttnConfig):
            cfg = cfg.linear_attn_config

        if not isinstance(cfg, LinearAttnConfig):
            return 0, 0

        tp_size = self.serverArgs.tp_size if self.serverArgs.tp_size > 0 else 1

        # 1. Conv State
        num_v_heads = cfg.num_value_heads // tp_size
        num_k_heads = cfg.num_key_heads // tp_size
        conv_state_size = (
            (num_v_heads * cfg.value_head_dim + 2 * num_k_heads * cfg.key_head_dim)
            * (cfg.conv_kernel_dim - 1)
            * 2
        )
        
        # 2. SSM State (FP32)
        ssm_state_size = (
            num_v_heads * cfg.key_head_dim * cfg.value_head_dim * 4
        )

        return conv_state_size + ssm_state_size, 0
