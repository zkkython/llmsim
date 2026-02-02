from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.op.operator_base import (
    OperatorMetadata, OperatorIO, Tensor, DataType,
)
from src.arch.op.op_register import create_operator

class SimpleTransformerArch(BaseModelArch):
    """简单 Transformer 模型架构（如 Qwen3）"""
    
    def build_operators(self) -> None:
        """构建标准 Transformer 算子"""
        mc = self.model_config
        sc = self.schedule_config
        
        assert mc.num_attention_heads % sc.tp_size == 0
        if mc.num_key_value_heads > sc.tp_size:
            assert mc.num_key_value_heads % sc.tp_size == 0
        else:
            assert sc.tp_size % mc.num_key_value_heads == 0
        
        # 计算每个 rank 的头数
        num_heads_per_rank = mc.num_attention_heads // sc.tp_size
        kv_heads_per_rank = max(1, mc.num_key_value_heads // sc.tp_size)
        seq_len = self.get_seq_length()
        head_dim = getattr(mc, 'head_dim', mc.hidden_size // mc.num_attention_heads)
        
        # 1. QKV 投影
        qkv_proj_metadata = OperatorMetadata(
            name='qkv_proj',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim),
                weight_shape=Tensor(mc.hidden_size, (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=1,
            num_layers=mc.num_hidden_layers,
        )
        self._add_operator(create_operator('matmul', qkv_proj_metadata))
        
        # 2. 输出投影
        o_proj_metadata = OperatorMetadata(
            name='o_proj',
            op_type='matmul',
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
        self._add_operator(create_operator('matmul', o_proj_metadata))
        
        # 3. 注意力核心
        attn_operators = []
        
        # Q-K 注意力
        qk_metadata = OperatorMetadata(
            name='qk',
            op_type='attention',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, head_dim),
                output_shape=Tensor(seq_len, sc.max_seqlen),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.BF16,
                output_dtype=DataType.FP32,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=mc.num_hidden_layers,
        )
        attn_operators.append(create_operator('attention', qk_metadata))
        
        # Q-K-V 注意力
        qkv_metadata = OperatorMetadata(
            name='qkv',
            op_type='attention',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, sc.max_seqlen),
                output_shape=Tensor(seq_len, head_dim),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=kv_heads_per_rank,
            num_layers=mc.num_hidden_layers,
        )
        attn_operators.append(create_operator('attention', qkv_metadata))
        
        self._add_attention_operator('attention', attn_operators)
        
        # 4. 前馈网络 (FFN)
        assert mc.intermediate_size % sc.tp_size == 0
        intermediate_size = mc.intermediate_size // sc.tp_size
        
        # Gate-Up 投影
        gate_up_metadata = OperatorMetadata(
            name='dense_gate_up_proj',
            op_type='matmul',
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
        self._add_operator(create_operator('matmul', gate_up_metadata))
        
        # Down 投影
        down_metadata = OperatorMetadata(
            name='dense_down_proj',
            op_type='matmul',
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
        self._add_operator(create_operator('matmul', down_metadata))