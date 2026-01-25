import os

import pytest

from src.config.model_config import ModelConfig
from src.layers.decode_block import DecoderBlocks
from src.server_args import ServerArgs


def get_representative_configs():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 选择几个有代表性的模型
    configs = [
        "hf_config/qwen3-8B_config.json",  # MHA + DenseMLP
        "hf_config/deepseek_671b_r1_config.json",  # MLA + MoE (Hybrid)
        "hf_config/qwen3-next-80B-A3B_config.json",  # HybridAttn + MoE
    ]
    return [
        os.path.join(base_path, f)
        for f in configs
        if os.path.exists(os.path.join(base_path, f))
    ]


@pytest.mark.parametrize("config_path", get_representative_configs())
def test_tp_weight_comparison(config_path):
    print(f"\nTP Comparison for: {os.path.basename(config_path)}")
    config = ModelConfig.from_config_path(config_path)

    tp_values = [1, 2, 4, 8]
    results = {}

    print(
        f"{'TP Size':<10} | {'Attn (MB)':<12} | {'FFN (MB)':<12} | {'Total (MB)':<12} | {'Reduction'}"
    )
    print("-" * 70)

    baseline_total = 0

    for tp in tp_values:
        # 为了公平对比 MoE，我们假设 world_size 随 TP 增加，保持 ep_size=1 (或者保持 world_size = tp * ep)
        # 这里我们主要验证 TP 对 Dense 部分的削减效果，所以固定 ep_size=1
        server_args = ServerArgs(
            config_path=config_path,
            tp_size=tp,
            ep_size=1,  # 固定每个 GPU 负责全量专家，只看 TP 切分
            use_fp8_gemm=False,
        )

        blocks = DecoderBlocks(server_args, config)
        attn_mb = blocks.total_attn_weights() / (1024**2)
        ffn_mb = blocks.total_ffn_weights() / (1024**2)
        total_mb = blocks.weights_bytes() / (1024**2)

        if tp == 1:
            baseline_total = total_mb
            reduction = "1.00x"
        else:
            reduction = f"{baseline_total / total_mb:.2f}x"

        print(
            f"{tp:<10} | {attn_mb:<12.2f} | {ffn_mb:<12.2f} | {total_mb:<12.2f} | {reduction}"
        )

        results[tp] = total_mb

    # 断言：随着 TP 增加，权重应该减少（因为我们固定了 ep_size）
    assert results[2] < results[1]
    assert results[4] < results[2]


if __name__ == "__main__":
    configs = get_representative_configs()
    for cfg in configs:
        test_tp_weight_comparison(cfg)
