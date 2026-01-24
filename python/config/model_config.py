import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    is_hybrid_linear: bool
    num_full_attn_layers: int
    num_linear_attn_layers: int
    linear_conv_kernel_dim: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_num_key_heads: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_num_value_heads: Optional[int] = None
    attn_type: str = "MHA/GQA"
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    v_head_dim: Optional[int] = None
    index_topk: Optional[int] = None
    qk_head_dim: Optional[int] = None
    is_moe: bool = True
    num_routed_experts: int = 1
    num_experts_per_tok: int = 1
    intermediate_size: Optional[int] = None
    num_shared_experts: int = 0
    first_k_dense_replace: Optional[int] = 0

    @classmethod
    def from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            d = json.load(f)

        # Determine if hybrid linear
        is_hybrid_linear = d.get("full_attention_interval") is not None
        if is_hybrid_linear:
            num_full_attn_layers = (
                d["num_hidden_layers"] // d["full_attention_interval"]
            )
            num_linear_attn_layers = d["num_hidden_layers"] - num_full_attn_layers
            linear_conv_kernel_dim = d["linear_conv_kernel_dim"]
            linear_key_head_dim = d["linear_key_head_dim"]
            linear_num_key_heads = d["linear_num_key_heads"]
            linear_value_head_dim = d["linear_value_head_dim"]
            linear_num_value_heads = d["linear_num_value_heads"]
        else:
            num_full_attn_layers = 0
            num_linear_attn_layers = 0
            linear_conv_kernel_dim = None
            linear_key_head_dim = None
            linear_num_key_heads = None
            linear_value_head_dim = None
            linear_num_value_heads = None

        # Determine attention type
        attn_type = "MHA/GQA" if "kv_lora_rank" not in d else "MLA"

        # Handle attention-specific fields
        if attn_type == "MHA/GQA":
            num_attention_heads = d["num_attention_heads"]
            num_key_value_heads = d["num_key_value_heads"]
            if "head_dim" in d:
                head_dim = d["head_dim"]
            else:
                head_dim = d["hidden_size"] // d["num_attention_heads"]

            # MLA-specific fields remain None
            q_lora_rank = None
            qk_nope_head_dim = None
            qk_rope_head_dim = None
            kv_lora_rank = None
            v_head_dim = None
            index_topk = d.get("index_topk")
            qk_head_dim = None
        elif attn_type == "MLA":
            q_lora_rank = d["q_lora_rank"]
            qk_nope_head_dim = d["qk_nope_head_dim"]
            qk_rope_head_dim = d["qk_rope_head_dim"]
            kv_lora_rank = d["kv_lora_rank"]
            num_attention_heads = d["num_attention_heads"]
            v_head_dim = d["v_head_dim"]
            index_topk = d.get("index_topk")
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            first_k_dense_replace = d["first_k_dense_replace"]

        # Handle MoE-specific fields, for ds use num_routed_experts, for qwen use num_experts
        if "num_routed_experts" in d:
            num_routed_experts = d["num_routed_experts"]
        elif "num_experts" in d:
            num_routed_experts = d["num_experts"]
        else:
            is_moe = False
            num_routed_experts = 1

        if is_moe:
            num_experts_per_tok = d["num_experts_per_tok"]
            intermediate_size = d["moe_intermediate_size"]
            num_shared_experts = d.get("num_shared_experts", 0)
        else:
            num_experts_per_tok = 1
            intermediate_size = d["intermediate_size"]
            num_shared_experts = 0

        return cls(
            hidden_size=d["hidden_size"],
            num_hidden_layers=d["num_hidden_layers"],
            is_hybrid_linear=is_hybrid_linear,
            num_full_attn_layers=num_full_attn_layers,
            num_linear_attn_layers=num_linear_attn_layers,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_value_heads=linear_num_value_heads,
            attn_type=attn_type,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            v_head_dim=v_head_dim,
            index_topk=index_topk,
            qk_head_dim=qk_head_dim,
            is_moe=is_moe,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            num_shared_experts=num_shared_experts,
            first_k_dense_replace=first_k_dense_replace,
        )
