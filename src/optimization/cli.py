"""
Command-line interface for parameter optimization

Usage:
    python -m src.optimization.cli \
        --model_path hf_config/qwen3-32B_config.json \
        --hardware h800 \
        --max_seqlen 4096 \
        --tp_range "1,2,4,8" \
        --dp_range "1,2,4,8" \
        --batch_range "1-128" \
        --objective maximize_tps \
        --output result.json
"""

import argparse
import json
import sys
from typing import List, Union

from src.arch.config import ModelConfig
from src.hardware.hardware_config import get_hardware_config
from src.optimization.config import SearchSpaceConfig
from src.optimization.results import OptimizationResult
from src.optimization.service import OptimizationService


def parse_range(value: str) -> Union[List[int], tuple]:
    """Parse a range specification"""
    value = value.strip()

    # Try comma-separated list first
    if "," in value:
        return [int(x.strip()) for x in value.split(",")]

    # Try range format (min-max)
    if "-" in value:
        parts = value.split("-")
        if len(parts) == 2:
            return (int(parts[0].strip()), int(parts[1].strip()))

    # Try single value
    try:
        return [int(value)]
    except ValueError:
        pass

    raise argparse.ArgumentTypeError(f"Invalid range format: {value}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="LLMSim Parameter Optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize for maximum throughput
  python -m src.optimization.cli \\
      --model_path hf_config/qwen3-32B_config.json \\
      --hardware h800 \\
      --max_seqlen 4096 \\
      --objective maximize_tps

  # Quick recommendation with specific priority
  python -m src.optimization.cli \\
      --model_path hf_config/deepseek_671b_r1_config.json \\
      --hardware klx_p800 \\
      --max_seqlen 8192 \\
      --recommend throughput

  # Full grid search with custom ranges
  python -m src.optimization.cli \\
      --model_path hf_config/qwen3-32B_config.json \\
      --hardware h800 \\
      --max_seqlen 4096 \\
      --tp_range "1,2,4,8" \\
      --dp_range "1,2,4,8" \\
      --batch_range "1-128" \\
      --objective balanced \\
      --output result.json
        """,
    )

    # Model and hardware
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model configuration JSON file",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="h800",
        help="Hardware configuration name (h20, h800, gb200, klx_p800) or path to JSON",
    )

    # Fixed parameters
    parser.add_argument(
        "--max_seqlen",
        type=int,
        required=True,
        help="Maximum sequence length (fixed parameter)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="extend",
        choices=["extend", "decode"],
        help="Forward mode: extend (prefill) or decode",
    )

    # Search space
    parser.add_argument(
        "--tp_range",
        type=parse_range,
        default="1,2,4,8",
        help="Tensor Parallel range (e.g., '1,2,4,8' or '1-8')",
    )
    parser.add_argument(
        "--dp_range",
        type=parse_range,
        default="1,2,4,8",
        help="Data Parallel range (e.g., '1,2,4,8' or '1-8')",
    )
    parser.add_argument(
        "--ep_range",
        type=parse_range,
        default=None,
        help="Expert Parallel range (e.g., '1,2,4,8' or '1-16'). Auto-detected for MoE models.",
    )
    parser.add_argument(
        "--batch_range",
        type=parse_range,
        default="1-128",
        help="Batch size range (e.g., '1,2,4,8' or '1-128')",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Total number of GPUs (constrains TP * DP)",
    )

    # Optimization settings
    parser.add_argument(
        "--objective",
        type=str,
        default="maximize_tps",
        choices=["minimize_ttft", "maximize_tps", "balanced", "latency", "throughput"],
        help="Optimization objective",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="grid_search",
        choices=["grid_search"],
        help="Optimization algorithm",
    )
    parser.add_argument(
        "--max_evaluations",
        type=int,
        default=None,
        help="Maximum number of configurations to evaluate",
    )

    # Quick recommendation mode
    parser.add_argument(
        "--recommend",
        type=str,
        default=None,
        choices=["latency", "throughput", "balanced"],
        help="Quick recommendation mode (skips full search)",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default=None, help="Output file path (JSON format)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser


def load_model_config(model_path: str):
    """Load model configuration"""
    try:
        return ModelConfig.from_json(model_path)
    except Exception as e:
        print(f"Error loading model config: {e}", file=sys.stderr)
        sys.exit(1)


def load_hardware_config(hardware: str):
    """Load hardware configuration"""
    try:
        if hardware.endswith(".json") or hardware.endswith(".json5"):
            return get_hardware_config(hardware)
        else:
            return get_hardware_config(hardware)
    except Exception as e:
        print(f"Error loading hardware config: {e}", file=sys.stderr)
        sys.exit(1)


def format_result(result: OptimizationResult, verbose: bool = False) -> str:
    """Format optimization result for display"""
    lines = []
    lines.append("=" * 60)
    lines.append("OPTIMIZATION RESULTS")
    lines.append("=" * 60)

    if result.best_config:
        config = result.best_config
        lines.append("\nBest Configuration:")
        lines.append(f"  Tensor Parallel (TP): {config.tp_size}")
        lines.append(f"  Data Parallel (DP): {config.dp_size}")
        lines.append(f"  Expert Parallel (EP): {config.ep_size}")
        lines.append(f"  Batch Size: {config.batch_size}")
        lines.append(f"  Mode: {config.mode.name}")

        if result.best_metrics:
            lines.append("\nPerformance Metrics:")
            for key, value in result.best_metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
    else:
        lines.append("\nNo valid configuration found!")

    lines.append("\nOptimization Statistics:")
    lines.append(f"  Total evaluations: {result.total_evaluations}")
    lines.append(f"  Search space size: {result.search_space_size}")
    lines.append(f"  Total time: {result.total_time_seconds:.2f}s")

    if verbose and result.optimization_history:
        lines.append("\nTop 10 Configurations:")
        sorted_history = sorted(result.optimization_history, key=lambda x: x.score)
        for i, step in enumerate(sorted_history[:10], 1):
            config = step.config
            lines.append(
                f"  {i}. TP={config.tp_size} DP={config.dp_size} EP={config.ep_size} "
                f"BS={config.batch_size} Score={step.score:.4f}"
            )

    return "\n".join(lines)


def save_result(result: OptimizationResult, output_path: str):
    """Save result to JSON file"""
    data = {
        "best_config": {
            "tp_size": result.best_config.tp_size,
            "dp_size": result.best_config.dp_size,
            "ep_size": result.best_config.ep_size,
            "batch_size": result.best_config.batch_size,
            "max_seqlen": result.best_config.max_seqlen,
            "mode": result.best_config.mode.name,
        },
        "best_metrics": result.best_metrics,
        "statistics": {
            "total_evaluations": result.total_evaluations,
            "search_space_size": result.search_space_size,
            "total_time_seconds": result.total_time_seconds,
        },
        "optimization_history": [
            {
                "iteration": step.iteration,
                "tp_size": step.config.tp_size,
                "dp_size": step.config.dp_size,
                "ep_size": step.config.ep_size,
                "batch_size": step.config.batch_size,
                "score": step.score,
                "metrics": step.metrics,
            }
            for step in result.optimization_history
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Load configurations
    print(f"Loading model configuration from: {args.model_path}")
    model_config = load_model_config(args.model_path)

    print(f"Loading hardware configuration: {args.hardware}")
    hardware_config = load_hardware_config(args.hardware)

    # Create service
    service = OptimizationService()

    if args.recommend:
        # Quick recommendation mode
        print(f"\nGetting {args.recommend} recommendation...")
        recommended = service.get_recommended_config(
            model_config=model_config,
            hardware_config=hardware_config,
            max_seqlen=args.max_seqlen,
            priority=args.recommend,
            world_size=args.world_size,
        )

        print("\n" + "=" * 60)
        print("RECOMMENDED CONFIGURATION")
        print("=" * 60)
        print(recommended.explanation)

        if args.output:
            data = {
                "recommendation": {
                    "priority": recommended.priority,
                    "config": {
                        "tp_size": recommended.config.tp_size,
                        "dp_size": recommended.config.dp_size,
                        "ep_size": recommended.config.ep_size,
                        "batch_size": recommended.config.batch_size,
                        "mode": recommended.config.mode.name,
                    },
                    "metrics": recommended.metrics,
                    "explanation": recommended.explanation,
                }
            }
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    else:
        # Full optimization mode
        # Build search space config
        ep_range = args.ep_range
        if ep_range is None:
            # Auto-detect for MoE models
            model_type = getattr(model_config, "model_type", "")
            if "moe" in model_type.lower() or model_type == "deepseek_v3":
                ep_range = [1, 2, 4, 8, 16]
            else:
                ep_range = [1]

        search_config = SearchSpaceConfig(
            tp_size=args.tp_range,
            dp_size=args.dp_range,
            ep_size=ep_range,
            batch_size=args.batch_range,
            mode=args.mode,
            world_size=args.world_size,
        )

        print("\nSearch space:")
        print(f"  TP: {search_config.get_tp_values()}")
        print(f"  DP: {search_config.get_dp_values()}")
        print(f"  EP: {search_config.get_ep_values()}")
        print(f"  Batch: {search_config.get_batch_size_values()}")
        print(f"  Mode: {search_config.get_mode_values()}")
        print(f"  Total combinations: {search_config.get_search_space_size()}")

        print(f"\nRunning optimization with objective: {args.objective}")

        # Run optimization
        result = service.optimize(
            model_config=model_config,
            hardware_config=hardware_config,
            max_seqlen=args.max_seqlen,
            search_space_config=search_config,
            objective_type=args.objective,
            optimizer_type=args.optimizer,
            max_evaluations=args.max_evaluations,
        )

        # Display results
        print(format_result(result, verbose=args.verbose))

        # Save results if requested
        if args.output:
            save_result(result, args.output)


if __name__ == "__main__":
    main()
