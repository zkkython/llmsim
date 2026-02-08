#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

echo "=========================================="
echo "Qwen3 Model Optimization"
echo "=========================================="

# Qwen3-32B Dense Model Optimization
echo ""
echo "=========================================="
echo "Qwen3-32B Dense Model"
echo "=========================================="

# 1. Throughput optimization
echo ""
echo "1. Qwen3-32B Throughput Optimization..."
python -m src.optimization.cli \
    --model_path hf_config/qwen3-32B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --recommend throughput \
    --output metrics/qwen3_32b_throughput_recommendation.json

# 2. Latency optimization
echo ""
echo "2. Qwen3-32B Latency Optimization..."
python -m src.optimization.cli \
    --model_path hf_config/qwen3-32B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --recommend latency \
    --output metrics/qwen3_32b_latency_recommendation.json

# 3. Full grid search for Qwen3-32B
echo ""
echo "3. Qwen3-32B Full Grid Search..."
python -m src.optimization.cli \
    --model_path hf_config/qwen3-32B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --tp_range "1,2,4,8" \
    --dp_range "1,2,4,8" \
    --batch_range "1,4,8,16,32,64,128" \
    --world_size 16 \
    --objective maximize_tps \
    --output metrics/qwen3_32b_optimization.json

# Qwen3-8B Dense Model Optimization
echo ""
echo "=========================================="
echo "Qwen3-8B Dense Model"
echo "=========================================="

echo ""
echo "4. Qwen3-8B Balanced Optimization..."
python -m src.optimization.cli \
    --model_path hf_config/qwen3-8B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --recommend balanced \
    --output metrics/qwen3_8b_recommendation.json

# Qwen3-235B-A22B MoE Model Optimization
echo ""
echo "=========================================="
echo "Qwen3-235B-A22B MoE Model"
echo "=========================================="

echo ""
echo "5. Qwen3-235B-A22B MoE Throughput Optimization..."
python -m src.optimization.cli \
    --model_path hf_config/qwen3-235B-A22B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --recommend throughput \
    --output metrics/qwen3_235b_recommendation.json

echo ""
echo "6. Qwen3-235B-A22B Full Optimization..."
python -m src.optimization.cli \
    --model_path hf_config/qwen3-235B-A22B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --tp_range "1,2,4,8" \
    --dp_range "1,2,4,8" \
    --ep_range "1,2,4,8,16" \
    --batch_range "1,4,8,16,32" \
    --world_size 32 \
    --objective maximize_tps \
    --output metrics/qwen3_235b_optimization.json

# Qwen3-Next-80B-A3B MoE Model Optimization
echo ""
echo "=========================================="
echo "Qwen3-Next-80B-A3B MoE Model"
echo "=========================================="

echo ""
echo "7. Qwen3-Next-80B-A3B Optimization..."
python -m src.optimization.cli \
    --model_path hf_config/qwen3-next-80B-A3B_config.json \
    --hardware h800 \
    --max_seqlen 4096 \
    --recommend balanced \
    --output metrics/qwen3_next_80b_recommendation.json

echo ""
echo "=========================================="
echo "Qwen3 Optimization Complete!"
echo "Results saved to metrics/ directory"
echo "=========================================="
