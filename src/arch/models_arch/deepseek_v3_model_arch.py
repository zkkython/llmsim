from src.arch.config import DeepSeekV3Config, ForwardMode
from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.op.op_register import create_operator
from src.arch.op.operator_base import DataType, OperatorIO, OperatorMetadata, Tensor


class DeepSeekV3Arch(BaseModelArch):
    """混合专家模型架构（如 DeepSeek V3）"""

    def build_operators(self) -> None:
        """构建 MoE 模型的算子"""
        mc = self.model_config
        sc = self.schedule_config

        # 处理 MoE 配置的特殊情况
        if not isinstance(mc, DeepSeekV3Config):
            raise ValueError(f"MoE 架构需要 DeepSeekV3Config，但收到 {type(mc)}")

        # 1. 建立注意力层
        num_attn_layers = mc.num_hidden_layers + (1 if sc.is_mtp else 0)
        self._build_attention_operators(num_attn_layers)

        # 2. 建立密集层（前 K 层）
        self._build_dense_operators(mc.first_k_dense_replace)

        # 3. 建立 MoE 层
        moe_layers = (
            mc.num_hidden_layers - mc.first_k_dense_replace + (1 if sc.is_mtp else 0)
        )
        self._build_moe_operators(moe_layers)

        # 4. 建立 Deep-EP 传输算子（如果启用）
        if sc.deepep:
            self._build_deepep_operators(moe_layers)

    def _build_attention_operators(self, num_layers) -> None:
        # ====================
        # 1. QKV 投影算子
        # ====================
        """构建注意力算子"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, DeepSeekV3Config):
            raise ValueError("Expected DeepSeekV3Config")

        assert mc.num_attention_heads % sc.tp_size == 0
        num_heads_per_rank = mc.num_attention_heads // sc.tp_size
        seq_len = self.get_seq_length()
        qk_head_dim = mc.qk_nope_head_dim + mc.qk_rope_head_dim
        # QKV A 投影算子Meta信息
        q_a_kv_a_metadata = OperatorMetadata(
            name="q_a_kv_a",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(
                    seq_len, mc.q_lora_rank + mc.kv_lora_rank + mc.qk_rope_head_dim
                ),
                weight_shape=Tensor(
                    mc.hidden_size,
                    mc.q_lora_rank + mc.kv_lora_rank + mc.qk_rope_head_dim,
                ),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", q_a_kv_a_metadata))

        # Q B 投影算子Meta信息
        q_b_metadata = OperatorMetadata(
            name="q_b",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.q_lora_rank),
                output_shape=Tensor(seq_len, num_heads_per_rank * qk_head_dim),
                weight_shape=Tensor(mc.q_lora_rank, num_heads_per_rank * qk_head_dim),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", q_b_metadata))

        # KV B 投影
        if sc.mode == ForwardMode.EXTEND:
            kv_b_metadata = OperatorMetadata(
                name="kv_b",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.kv_lora_rank),
                    output_shape=Tensor(
                        seq_len, num_heads_per_rank * (mc.v_head_dim + mc.qk_nope_head_dim)
                    ),
                    weight_shape=Tensor(
                        mc.kv_lora_rank,
                        num_heads_per_rank * (mc.v_head_dim + mc.qk_nope_head_dim),
                    ),
                    input_dtype=DataType.INT8,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.INT8,
                ),
                batch_size=1,
                num_layers=num_layers,
            )
            self._add_operator(create_operator("matmul", kv_b_metadata))
        if sc.mode == ForwardMode.DECODE:
            q_absorb_metadata = OperatorMetadata(
                name="q_absorb",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.qk_nope_head_dim),
                    output_shape=Tensor(
                        seq_len, mc.kv_lora_rank
                    ),
                    weight_shape=Tensor(
                        mc.qk_nope_head_dim,
                        mc.kv_lora_rank,
                    ),
                    input_dtype=DataType.FP32,
                    output_dtype=DataType.FP32,
                    weight_dtype=DataType.FP32,
                ),
                batch_size=num_heads_per_rank,
                num_layers=num_layers,
            )
            self._add_operator(create_operator("matmul", q_absorb_metadata))

            o_absorb_metadata = OperatorMetadata(
                name="o_absorb",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.kv_lora_rank),
                    output_shape=Tensor(
                        seq_len, mc.v_head_dim
                    ),
                    weight_shape=Tensor(
                        mc.kv_lora_rank,
                        mc.v_head_dim,
                    ),
                    input_dtype=DataType.FP32,
                    output_dtype=DataType.FP32,
                    weight_dtype=DataType.FP32,
                ),
                batch_size=num_heads_per_rank,
                num_layers=num_layers,
            )
            self._add_operator(create_operator("matmul", o_absorb_metadata))

        # 输出投影
        o_proj_metadata = OperatorMetadata(
            name="o_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, num_heads_per_rank * mc.v_head_dim),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(num_heads_per_rank * mc.v_head_dim, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", o_proj_metadata))

        # 2. TP AllReduce (如果 TP > 1)
        if sc.tp_size > 1:
            # 根据模式选择带宽
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
                num_layers=num_layers,
            )
            all_reduce_op = create_operator("transfer", all_reduce_metadata)
            all_reduce_op._bandwidth_gb_s = reduce_bandwidth
            self._add_transfer_operator(all_reduce_op)

        # 3. 注意力核心
        attn_operators = []
        qk_nope_metadata = OperatorMetadata(
            name="qk_nope",
            op_type="attention",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.qk_nope_head_dim),
                output_shape=Tensor(seq_len, sc.max_seqlen),
                weight_shape=Tensor(mc.qk_nope_head_dim, sc.max_seqlen),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=num_layers,
        )
        # self._add_operator(create_operator('matmul', qk_nope_metadata))
        attn_operators.append(
            create_operator("attention", qk_nope_metadata, mc.attention_type)
        )

        qk_rope_metadata = OperatorMetadata(
            name="qk_rope",
            op_type="attention",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.qk_rope_head_dim),
                output_shape=Tensor(seq_len, sc.max_seqlen),
                weight_shape=Tensor(mc.qk_rope_head_dim, sc.max_seqlen),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=num_layers,
        )
        # self._add_operator(create_operator('matmul', qk_rope_metadata))
        attn_operators.append(
            create_operator("attention", qk_rope_metadata, mc.attention_type)
        )

        qkv_metadata = OperatorMetadata(
            name="qkv",
            op_type="attention",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, sc.max_seqlen),
                output_shape=Tensor(seq_len, mc.v_head_dim),
                weight_shape=Tensor(sc.max_seqlen, mc.v_head_dim),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=num_layers,
        )
        # self._add_operator(create_operator('matmul', qkv_metadata))
        attn_operators.append(
            create_operator("attention", qkv_metadata, mc.attention_type)
        )
        self._add_attention_operator("attention", attn_operators)

    def _build_dense_operators(self, num_dense_layers: int) -> None:
        # ====================
        # 2. Dense 层算子
        # ====================
        """构建密集层算子"""
        mc = self.model_config
        sc = self.schedule_config

        seq_len = self.get_seq_length()
        assert mc.intermediate_size % sc.tp_size == 0
        intermediate_size = mc.intermediate_size
        if not sc.enable_moe_dense_fully_dp:
            intermediate_size = intermediate_size // sc.tp_size

        # Gate-Up 投影
        gate_up_metadata = OperatorMetadata(
            name="dense_gate_up_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, 2 * intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * intermediate_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_dense_layers,
        )
        self._add_operator(create_operator("matmul", gate_up_metadata))

        # Down 投影
        down_metadata = OperatorMetadata(
            name="dense_down_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, intermediate_size),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(intermediate_size, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_dense_layers,
        )
        self._add_operator(create_operator("matmul", down_metadata))

        # 3. TP AllReduce (如果 TP > 1 且不是全 DP 模式)
        if sc.tp_size > 1 and not sc.enable_moe_dense_fully_dp:
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
                num_layers=num_dense_layers,
            )
            all_reduce_op = create_operator("transfer", all_reduce_metadata)
            all_reduce_op._bandwidth_gb_s = reduce_bandwidth
            self._add_transfer_operator(all_reduce_op)

    def _build_moe_operators(self, num_moe_layers: int) -> None:
        # ====================
        # 3. MoE 层算子
        # ====================
        """构建 MoE 算子"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, DeepSeekV3Config):
            raise ValueError("Expected DeepSeekV3Config")

        seq_len = self.get_seq_length()
        assert mc.n_routed_experts % sc.ep_size == 0
        experts_per_rank = mc.n_routed_experts // sc.ep_size
        if sc.mode == ForwardMode.EXTEND:
            L = sc.max_seqlen
        elif sc.mode == ForwardMode.DECODE:
            L = sc.batch_size
        assert L // sc.tp_size * mc.num_experts_per_tok % experts_per_rank == 0
        # 计算每个 rank 的 token 数量
        L_per_rank = (
            L // sc.tp_size * mc.num_experts_per_tok // experts_per_rank
        )
        
        # MoE 共享中间层大小
        _moe_intermediate_size = mc.moe_intermediate_size
        if not sc.deepep:
            _moe_intermediate_size = _moe_intermediate_size // sc.tp_size

        # MoE Gate 投影
        moe_gate_metadata = OperatorMetadata(
            name="moe_gate",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, mc.n_routed_experts),
                weight_shape=Tensor(mc.hidden_size, mc.n_routed_experts),
                input_dtype=DataType.FP32,
                output_dtype=DataType.FP32,
                weight_dtype=DataType.FP32,
            ),
            batch_size=1,
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator("matmul", moe_gate_metadata))

        # MoE Up 投影
        moe_up_metadata = OperatorMetadata(
            name="moe_up",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(L_per_rank, mc.hidden_size),
                output_shape=Tensor(L_per_rank, 2 * mc.moe_intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * mc.moe_intermediate_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=int(experts_per_rank),
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator("matmul", moe_up_metadata))

        # MoE Down 投影
        moe_down_metadata = OperatorMetadata(
            name="moe_down",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(L_per_rank, mc.moe_intermediate_size),
                output_shape=Tensor(L_per_rank, mc.hidden_size),
                weight_shape=Tensor(mc.moe_intermediate_size, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=int(experts_per_rank),
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator("matmul", moe_down_metadata))

        # 共享专家 Up 投影
        share_up_metadata = OperatorMetadata(
            name="share_up",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, 2 * _moe_intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * _moe_intermediate_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator("matmul", share_up_metadata))

        # 共享专家 Down 投影
        share_down_metadata = OperatorMetadata(
            name="share_down",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, _moe_intermediate_size),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(_moe_intermediate_size, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator("matmul", share_down_metadata))

    def _build_deepep_operators(self, num_moe_layers: int) -> None:
        # ====================
        # 4. Deep-EP 传输算子
        # ====================
        """构建 Deep-EP 传输算子"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, DeepSeekV3Config):
            raise ValueError("Expected DeepSeekV3Config")

        if sc.mode == ForwardMode.EXTEND:
            L = sc.max_seqlen // sc.tp_size
            dispatch_bandwidth = 85.0  # GB/s
            combine_bandwidth = 85.0
        else:  # DECODE
            L = sc.batch_size // sc.tp_size
            dispatch_bandwidth = 18.58  # GB/s
            combine_bandwidth = 22.64

        # 分发传输, prefill 或者 decode 下面的逻辑一致，只是dispatch_bandwidth 和 combine_bandwidth 的值不同
        dispatch_metadata = OperatorMetadata(
            name="dispatch",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(L, mc.hidden_size),
                output_shape=Tensor(L, mc.hidden_size),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.INT8,
            ),
            batch_size=mc.num_experts_per_tok,
            num_layers=num_moe_layers,
        )
        dispatch_op = create_operator("transfer", dispatch_metadata)
        # 设置带宽（用于传输时间计算）
        dispatch_op._bandwidth_gb_s = dispatch_bandwidth
        self._add_transfer_operator(dispatch_op)

        # 合并传输
        combine_metadata = OperatorMetadata(
            name="combine",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(L, mc.hidden_size),
                output_shape=Tensor(L, mc.hidden_size),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.BF16,
            ),
            batch_size=mc.num_experts_per_tok,
            num_layers=num_moe_layers,
        )
        combine_op = create_operator("transfer", combine_metadata)
        # 设置带宽（用于传输时间计算）
        combine_op._bandwidth_gb_s = combine_bandwidth
        self._add_transfer_operator(combine_op)
