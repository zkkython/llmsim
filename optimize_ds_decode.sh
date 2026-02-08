#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

echo "=========================================="
echo "DeepSeek V3/R1 Decode Optimization"
echo "=========================================="

# 1. Quick recommendation for throughput in decode mode
echo ""
echo "1. Optimizing for Maximum Throughput (Decode)..."
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --mode decode \
    --recommend throughput \
    --output metrics/ds_decode_throughput_recommendation.json

# 2. Quick recommendation for low latency in decode mode
echo ""
echo "2. Optimizing for Low Latency (Decode)..."
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --mode decode \
    --recommend latency \
    --output metrics/ds_decode_latency_recommendation.json

# 3. Full grid search for decode with different batch sizes
echo ""
echo "3. Running Full Grid Search for Decode (world_size=32)..."
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --mode decode \
    --tp_range "1,2,4,8" \
    --dp_range "1,2,4,8" \
    --ep_range "1,2,4,8,16" \
    --batch_range "8,16,32,64,128,256" \
    --world_size 32 \
    --objective maximize_tps \
    --output metrics/ds_decode_optimization_ws32.json

# 4. Decode optimization with minimize TTFT objective
echo ""
echo "4. Optimizing for Minimum TTFT (Decode)..."
python -m src.optimization.cli \
    --model_path hf_config/deepseek_671b_r1_config.json \
    --hardware klx_p800 \
    --max_seqlen 4096 \
    --mode decode \
    --tp_range "1,2,4,8" \
    --dp_range "1,2,4,8" \
    --ep_range "1,2,4,8,16" \
    --batch_range "1,4,8,16" \
    --world_size 16 \
    --objective minimize_ttft \
    --output metrics/ds_decode_minimize_ttft.json

echo ""
echo "=========================================="
echo "Decode Optimization Complete!"
echo "Results saved to metrics/ directory"
echo "=========================================="
