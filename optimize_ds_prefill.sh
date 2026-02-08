#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

echo "=========================================="
echo "DeepSeek V3/R1 Prefill Optimization"
echo "=========================================="

# 1. Quick recommendation for throughput (maximum TPS)
echo ""
echo "1. Optimizing for Maximum Throughput..."
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --recommend throughput \
    --output metrics/ds_prefill_throughput_recommendation.json

# 2. Quick recommendation for low latency (minimum TTFT)
echo ""
echo "2. Optimizing for Low Latency..."
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --recommend latency \
    --output metrics/ds_prefill_latency_recommendation.json

# 3. Full grid search with world_size=32 (e.g., 4 nodes x 8 GPUs)
echo ""
echo "3. Running Full Grid Search (world_size=32)..."
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --tp_range "1,2,4,8" \
    --dp_range "1,2,4,8" \
    --ep_range "1,2,4,8,16" \
    --batch_range "1,4,8,16,32,64,128" \
    --world_size 32 \
    --objective maximize_tps \
    --output metrics/ds_prefill_optimization_ws32.json

# 4. Full grid search with world_size=16 (e.g., 2 nodes x 8 GPUs)
echo ""
echo "4. Running Full Grid Search (world_size=16)..."
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --tp_range "1,2,4,8" \
    --dp_range "1,2,4,8" \
    --ep_range "1,2,4,8,16" \
    --batch_range "1,4,8,16,32,64" \
    --world_size 16 \
    --objective maximize_tps \
    --output metrics/ds_prefill_optimization_ws16.json

# 5. Balanced optimization with verbose output
echo ""
echo "5. Running Balanced Optimization (verbose)..."
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --tp_range "1,2,4" \
    --dp_range "1,2,4" \
    --ep_range "1,2,4,8" \
    --batch_range "1,8,16,32" \
    --world_size 16 \
    --objective balanced \
    --verbose \
    --output metrics/ds_prefill_balanced_optimization.json

echo ""
echo "=========================================="
echo "Optimization Complete!"
echo "Results saved to metrics/ directory"
echo "=========================================="
