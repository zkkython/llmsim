from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerArgs:
    config_path: str
    device_type: str = "H20"
    world_size: int = 1
    num_nodes: int = 1
    max_prefill_tokens: int = 4096
    decode_bs: Optional[int] = None
    target_tgs: float = 2560
    target_tpot: float = 50
    target_isl: int = 4096
    target_osl: int = 2048
    use_fp8_gemm: bool = False
    use_fp8_kv: bool = False
    enable_deepep: bool = False
    enable_tbo: bool = False
    sm_ratio: float = 108 / 132
    prefill_only: bool = False
    decode_only: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(
            config_path=args.config_path,
            device_type=args.device_type,
            world_size=args.world_size,
            num_nodes=args.num_nodes,
            max_prefill_tokens=args.max_prefill_tokens,
            decode_bs=args.decode_bs,
            target_tgs=args.target_tgs,
            target_tpot=args.target_tpot,
            target_isl=args.target_isl,
            target_osl=args.target_osl,
            use_fp8_gemm=args.use_fp8_gemm,
            use_fp8_kv=args.use_fp8_kv,
            enable_deepep=args.enable_deepep,
            enable_tbo=args.enable_tbo,
            sm_ratio=args.sm_ratio,
            prefill_only=args.prefill_only,
            decode_only=args.decode_only,
        )
