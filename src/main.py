import argparse
import sys

from src.hardware.hardware_config import HardwareConfig, DEFAULT_HARDWARE, get_hardware_config
from src.arch.config import ModelConfig, ScheduleConfig, ForwardMode
from src.arch.models_arch.model_arch import create_model_arch
from src.arch.perf_calculator import PerformanceCalculator

def parse_args():
    parser = argparse.ArgumentParser(description="LLM 推理性能分析工具")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型配置文件路径",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="批次大小",
    )
    parser.add_argument(
        "--max_seqlen",
        type=int,
        default=4096,
        help="最大序列长度",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="extend",
        choices=["extend", "decode"],
        help="前向传递模式 [extend, decode]",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=4,
        help="张量并行度",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=4,
        help="数据并行度",
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=16,
        help="专家并行度",
    )
    parser.add_argument(
        "--enable_mtp",
        action="store_true",
        help="启用多 token 预测",
    )
    parser.add_argument(
        "--enable_deepep",
        action="store_true",
        help="启用深度专家并行",
    )
    parser.add_argument(
        "--enable_moe_dense_fully_dp",
        action="store_true",
        help="启用 MoE 密集层完全数据并行",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        choices=["default", "h20", "h800", "gb200", "klx_p800", "custom"],
        default="default",
        help="硬件配置预设 (default, h20, h800, gb200, klx_p800, custom)",
    )
    parser.add_argument(
        "--hardware_config",
        type=str,
        default=None,
        help="自定义硬件配置文件路径 (当 --hardware=custom 时使用)",
    )

    args = parser.parse_args()
    return args


def validate_args(args) -> None:
    """验证命令行参数"""
    if args.max_seqlen % args.tp_size != 0:
        raise ValueError(f"max_seqlen ({args.max_seqlen}) 必须能被 tp_size ({args.tp_size}) 整除")
    
    if args.batch_size > args.tp_size:
        if args.batch_size % args.tp_size != 0:
            raise ValueError(f"batch_size ({args.batch_size}) 必须能被 tp_size ({args.tp_size}) 整除")


def main() -> None:
    """主函数"""
    args = parse_args()
    
    try:
        validate_args(args)
    except ValueError as e:
        print(f"参数验证失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 加载模型配置
    print(f"加载模型配置: {args.model_path}")
    try:
        model_config = ModelConfig.from_json(args.model_path)
    except Exception as e:
        print(f"加载模型配置失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 创建调度配置
    schedule_config = ScheduleConfig(
        batch_size=args.batch_size,
        max_seqlen=args.max_seqlen,
        mode=ForwardMode.EXTEND if args.mode == "extend" else ForwardMode.DECODE,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        ep_size=args.ep_size,
        is_mtp=args.enable_mtp,
        deepep=args.enable_deepep,
        enable_moe_dense_fully_dp=args.enable_moe_dense_fully_dp,
    )
    
    # 选择硬件配置
    if args.hardware == "custom":
        # 自定义配置文件
        if args.hardware_config is None:
            print("错误: 使用 --hardware=custom 时必须指定 --hardware_config", file=sys.stderr)
            sys.exit(1)
        hardware_config = HardwareConfig.from_json(args.hardware_config)
    elif args.hardware in ["h20", "h800", "gb200", "klx_p800"]:
        # 从预定义配置加载
        hardware_config = get_hardware_config(args.hardware)
    else:
        # 默认配置
        hardware_config = DEFAULT_HARDWARE
    
    # 创建模型架构
    print(f"模型类型: {model_config.model_type}")
    print(f"前向模式: {schedule_config.mode.name}")
    print(f"硬件配置: {hardware_config.name}")
    print()
    
    try:
        model_arch = create_model_arch(model_config, schedule_config)
    except Exception as e:
        print(f"创建模型架构失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 计算性能
    print("计算性能指标...")
    calculator = PerformanceCalculator(hardware_config)
    
    try:
        model_perf = calculator.calculate_model_performance(model_arch)
    except Exception as e:
        print(f"性能计算失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 打印性能报告
    calculator.print_performance_report(model_perf)


if __name__ == "__main__":
    main()

