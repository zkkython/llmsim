from src.config.model_config import ModelConfig
from src.layers.attn import Attn
from src.layers.ffn import FFN
from src.server_args import ServerArgs


class DecoderBlock:

    def __init__(self, serverArgs: ServerArgs, config: ModelConfig, layer_idx: int):
        self.serverArgs = serverArgs
        self.config = config
        self.layer_idx = layer_idx
        # calculate weights size of one layer decoder block
        self.attn = Attn.create(serverArgs, config, layer_idx)
        self.ffn = FFN.create(serverArgs, config, layer_idx)

    def attn_type(self):
        return self.attn.attn_type()

    def attn_weights_bytes(self):
        return self.attn.weights_size()

    def ffn_type(self):
        return self.ffn.layer_idx_ffn_state()

    def ffn_weights_bytes(self):
        return self.ffn.weights_size()

    def weights_bytes(self):
        attn_weights_bytes = self.attn.weights_size()
        # in FFN Class, it will calculate different weights size of dense mlp or moe by different config
        ffn_weights_bytes = self.ffn.weights_size()
        weight_size = attn_weights_bytes + ffn_weights_bytes
        return weight_size

    def flops_bytes(self):
        return 0

    def kvcache_bytes(self, context_len: int):
        static_bytes, per_token_bytes = self.attn.kv_cache_factors()
        return static_bytes + per_token_bytes * context_len

    def prefill_cost(self):
        return 0

    def decode_cost(self):
        return 0


class DecoderBlocks:
    def __init__(self, serverArgs: ServerArgs, config: ModelConfig):
        self.serverArgs = serverArgs
        self.config = config
        self.blocks = [
            DecoderBlock(serverArgs, config, layer_idx)
            for layer_idx in range(self.config.num_hidden_layers)
        ]
        self.block_weights_of_layers = dict()
        self.block_attn_weights_of_layers = dict()
        self.block_ffn_weights_of_layers = dict()
        for layer_idx, block in enumerate(self.blocks):
            self.block_weights_of_layers[layer_idx] = block.weights_bytes()
            self.block_attn_weights_of_layers[layer_idx] = dict(
                attn_type=block.attn_type(), weights_bytes=block.attn_weights_bytes()
            )
            self.block_ffn_weights_of_layers[layer_idx] = dict(
                ffn_type=block.ffn_type(), weights_bytes=block.ffn_weights_bytes()
            )

    def weights_bytes(self):
        return sum(block.weights_bytes() for block in self.blocks)

    def kvcache_bytes(self, context_len: int):
        return sum(block.kvcache_bytes(context_len) for block in self.blocks)

    def total_attn_weights(self):
        return sum(
            self.block_attn_weights_of_layers[layer_idx]["weights_bytes"]
            for layer_idx in self.block_attn_weights_of_layers
        )

    def total_ffn_weights(self):
        return sum(
            self.block_ffn_weights_of_layers[layer_idx]["weights_bytes"]
            for layer_idx in self.block_ffn_weights_of_layers
        )

    def print_decode_block_weights_info(self):
        header = f"{'Layer':<6} | {'Attn Type':<12} | {'FFN Type':<12} | {'Attn (MB)':<12} | {'FFN (MB)':<12} | {'Total (MB)':<12}"
        separator = "-" * len(header)

        print("\nDetailed Decoder Weights Information:")
        print(separator)
        print(header)
        print(separator)

        for layer_idx in range(len(self.blocks)):
            attn_info = self.block_attn_weights_of_layers[layer_idx]
            ffn_info = self.block_ffn_weights_of_layers[layer_idx]
            total_bytes = self.block_weights_of_layers[layer_idx]

            attn_mb = attn_info["weights_bytes"] / (1024**2)
            ffn_mb = ffn_info["weights_bytes"] / (1024**2)
            total_mb = total_bytes / (1024**2)

            print(
                f"{layer_idx:<6} | {attn_info['attn_type']:<12} | {ffn_info['ffn_type']:<12} | {attn_mb:<12.2f} | {ffn_mb:<12.2f} | {total_mb:<12.2f}"
            )

        print(separator)
        total_attn_mb = self.total_attn_weights() / (1024**2)
        total_ffn_mb = self.total_ffn_weights() / (1024**2)
        total_mb = self.weights_bytes() / (1024**2)

        print(
            f"{'Total':<6} | {'':<12} | {'':<12} | {total_attn_mb:<12.2f} | {total_ffn_mb:<12.2f} | {total_mb:<12.2f}"
        )
        print(
            f"{'Total (GB)':<6} | {'':<12} | {'':<12} | {total_attn_mb/1024:<12.4f} | {total_ffn_mb/1024:<12.4f} | {total_mb/1024:<12.4f}"
        )
        print(separator)

    def print_kvcache_info(self, context_len: int):
        header = f"{'Layer':<6} | {'Attn Type':<12} | {'KV/State (MB)':<15}"
        separator = "-" * len(header)

        print(f"\nKV Cache / States Information (Context Len: {context_len}):")
        print(separator)
        print(header)
        print(separator)

        for layer_idx, block in enumerate(self.blocks):
            kv_bytes = block.kvcache_bytes(context_len)
            kv_mb = kv_bytes / (1024**2)
            print(f"{layer_idx:<6} | {block.attn_type():<12} | {kv_mb:<15.2f}")

        print(separator)
        total_kv_mb = self.kvcache_bytes(context_len) / (1024**2)
        print(f"{'Total':<6} | {'':<12} | {total_kv_mb:<15.2f}")
        print(f"{'Total (GB)':<6} | {'':<12} | {total_kv_mb/1024:<15.4f}")
        print(separator)
