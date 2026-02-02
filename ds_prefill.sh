export PYTHONPATH=$PYTHONPATH:.

# 执行deepseek 的 prefill
python3 src/main.py --model_path hf_config/deepseek_671b_r1_config.json \
    --max_seqlen 4096 \
    --batch_size 1 \
    --mode extend \
    --tp_size 4 \
    --dp_size 4 \
    --ep_size 16 \
    --enable_deepep \
    --enable_moe_dense_fully_dp \
    --hardware klx_p800