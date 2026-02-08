from src.arch.config import ForwardMode
from src.arch.kvcache.kvcache import mha_gqa_kvcache, mha_gqa_kvcache_per_gpu
from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.op.op_register import create_operator
from src.arch.op.operator_base import DataType, OperatorIO, OperatorMetadata, Tensor


class SimpleTransformerArch(BaseModelArch):
    """Simple Transformer model architecture (e.g., Qwen3)"""

    def build_operators(self) -> None:
        """Build standard Transformer operators"""
        mc = self.model_config
        sc = self.schedule_config

        assert mc.num_attention_heads % sc.tp_size == 0
        if mc.num_key_value_heads > sc.tp_size:
            assert mc.num_key_value_heads % sc.tp_size == 0
        else:
            assert sc.tp_size % mc.num_key_value_heads == 0

        # Calculate number of heads per rank
        num_heads_per_rank = mc.num_attention_heads // sc.tp_size
        kv_heads_per_rank = max(1, mc.num_key_value_heads // sc.tp_size)
        seq_len = self.get_seq_length()
        head_dim = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)

        # 1. QKV projection
        qkv_proj_metadata = OperatorMetadata(
            name="qkv_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(
                    seq_len, (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim
                ),
                weight_shape=Tensor(
                    mc.hidden_size,
                    (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim,
                ),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=1,
            num_layers=mc.num_hidden_layers,
        )
        self._add_operator(create_operator("matmul", qkv_proj_metadata))

        # 2. Output projection
        o_proj_metadata = OperatorMetadata(
            name="o_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, num_heads_per_rank * head_dim),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(num_heads_per_rank * head_dim, mc.hidden_size),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=1,
            num_layers=mc.num_hidden_layers,
        )
        self._add_operator(create_operator("matmul", o_proj_metadata))

        # 2.1. TP AllReduce (if TP > 1)
        if sc.tp_size > 1:
            # Select bandwidth based on mode
            if sc.mode == ForwardMode.EXTEND:
                reduce_bandwidth = 85.0  # GB/s
            else:  # DECODE
                reduce_bandwidth = 22.64  # GB/s

            all_reduce_metadata = OperatorMetadata(
                name="attn_all_reduce",
                op_type="transfer",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.hidden_size),
                    output_shape=Tensor(seq_len, mc.hidden_size),
                    weight_shape=Tensor(0, 0),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                ),
                batch_size=1,
                num_layers=mc.num_hidden_layers,
            )
            all_reduce_op = create_operator("transfer", all_reduce_metadata)
            all_reduce_op._bandwidth_gb_s = reduce_bandwidth
            self._add_transfer_operator(all_reduce_op)

        # 3. Attention core
        attn_operators = []

        # Q-K attention
        qk_metadata = OperatorMetadata(
            name="qk",
            op_type="attention",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, head_dim),
                output_shape=Tensor(seq_len, sc.max_seqlen),
                weight_shape=Tensor(head_dim, sc.max_seqlen),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=mc.num_hidden_layers,
        )
        attn_operators.append(
            create_operator("attention", qk_metadata, mc.attention_type)
        )

        # Q-K-V attention
        qkv_metadata = OperatorMetadata(
            name="qkv",
            op_type="attention",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, sc.max_seqlen),
                output_shape=Tensor(seq_len, head_dim),
                weight_shape=Tensor(sc.max_seqlen, head_dim),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=kv_heads_per_rank,
            num_layers=mc.num_hidden_layers,
        )
        attn_operators.append(
            create_operator("attention", qkv_metadata, mc.attention_type)
        )

        self._add_attention_operator("attention", attn_operators)

        # 4. Feed-Forward Network (FFN)
        assert mc.intermediate_size % sc.tp_size == 0
        intermediate_size = mc.intermediate_size // sc.tp_size

        # Gate-Up projection
        gate_up_metadata = OperatorMetadata(
            name="dense_gate_up_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, 2 * intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * intermediate_size),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=1,
            num_layers=mc.num_hidden_layers,
        )
        self._add_operator(create_operator("matmul", gate_up_metadata))

        # Down projection
        down_metadata = OperatorMetadata(
            name="dense_down_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, intermediate_size),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(intermediate_size, mc.hidden_size),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=1,
            num_layers=mc.num_hidden_layers,
        )
        self._add_operator(create_operator("matmul", down_metadata))

        # 4.1. TP AllReduce (如果 TP > 1)
        if sc.tp_size > 1:
            # 根据模式选择带宽
            if sc.mode == ForwardMode.EXTEND:
                reduce_bandwidth = 85.0  # GB/s
            else:  # DECODE
                reduce_bandwidth = 22.64  # GB/s

            all_reduce_metadata = OperatorMetadata(
                name="dense_all_reduce",
                op_type="transfer",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.hidden_size),
                    output_shape=Tensor(seq_len, mc.hidden_size),
                    weight_shape=Tensor(0, 0),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                ),
                batch_size=1,
                num_layers=mc.num_hidden_layers,
            )
            all_reduce_op = create_operator("transfer", all_reduce_metadata)
            all_reduce_op._bandwidth_gb_s = reduce_bandwidth
            self._add_transfer_operator(all_reduce_op)

    def get_kv_cache(self):
        return mha_gqa_kvcache(self.model_config, DataType.BF16)

    def get_kv_cache_per_gpu(self):
        return mha_gqa_kvcache_per_gpu(
            self.model_config, DataType.BF16, self.schedule_config.tp_size
        )
