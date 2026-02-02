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


字段说明
字段	类型	单位	说明
device_type	string	-	设备类型 (gpu/xpu/accelerator)
name	string	-	硬件名称
memory.hbm_size_gb	int	GB	HBM 显存大小
memory.cache_line_size	int	Bytes	Cache 行大小
bandwidth.hbm_bandwidth_gb_s	float	TB/s	HBM 带宽
bandwidth.dma_bandwidth_gb_s	float	GB/s	DMA 带宽 (扩展模式)
bandwidth.dma_bandwidth_decode_gb_s	float	GB/s	DMA 带宽 (解码模式)
compute.mac_int8_gflops	float	TFLOPS	INT8 MAC 性能
compute.mac_fp32_gflops	float	TFLOPS	FP32 MAC 性能
compute.mac_bf16_gflops	float	TFLOPS	BF16 MAC 性能

输出excel:
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
    --hardware klx_p800 \
    --output_format excel \
    --output_file prefill_result.xlsx


输出console
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
    --hardware klx_p800 \
    --output_format console