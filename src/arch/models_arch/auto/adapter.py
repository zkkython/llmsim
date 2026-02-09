"""
Main Adapter API - High-level interface for SGLang model adaptation.

This module provides the main entry point for adapting SGLang models
to LLMSim ModelArch through the layered IR architecture.

Usage:
    from src.arch.models_arch.auto import auto_adapter, SglangAutoAdapter

    # One-step adaptation
    model_arch = auto_adapter(Qwen3MoeForCausalLM, config, schedule_config)

    # Or step-by-step for debugging
    adapter = SglangAutoAdapter(Qwen3MoeForCausalLM, config)
    ir_graph = adapter.parse()  # Step 1: Parse to IR
    model_arch = adapter.transform(schedule_config)  # Step 2: IR to ModelArch
"""

from typing import TYPE_CHECKING, Any, Optional, Type

from src.arch.config import ScheduleConfig
from src.arch.models_arch.auto.ir import ComputationalGraph
from src.arch.models_arch.auto.parser import ModelParser
from src.arch.models_arch.auto.transformer import IRToModelArchTransformer

if TYPE_CHECKING:
    from src.arch.models_arch.base_model_arch import BaseModelArch


class SglangAutoAdapter:
    """
    Main adapter class for SGLang model to LLMSim ModelArch conversion.

    This class orchestrates the two-phase adaptation process:
    1. Parse: SGLang model → ComputationalGraph (IR)
    2. Transform: ComputationalGraph → ModelArch

    The IR layer allows for inspection, debugging, and caching of the
    intermediate representation before final ModelArch construction.

    Example:
        adapter = SglangAutoAdapter(Qwen3MoeForCausalLM, config)

        # Parse to IR (can be inspected/debugged)
        ir_graph = adapter.parse()
        print(f"Model has {len(ir_graph.nodes)} operators")

        # Transform to ModelArch
        model_arch = adapter.transform(schedule_config)

        # Or do both in one step
        model_arch = adapter.adapt(schedule_config)
    """

    def __init__(self, model_class: Type, config: Any):
        """
        Initialize the adapter.

        Args:
            model_class: SGLang model class (e.g., Qwen3MoeForCausalLM)
            config: Model configuration object (transformers.PretrainedConfig or similar)
        """
        self.model_class = model_class
        self.config = config
        self._ir_graph: Optional[ComputationalGraph] = None
        self._model_arch: Optional["BaseModelArch"] = None

    @property
    def ir_graph(self) -> Optional[ComputationalGraph]:
        """Get the parsed IR graph (available after parse())."""
        return self._ir_graph

    @property
    def model_arch(self) -> Optional["BaseModelArch"]:
        """Get the transformed ModelArch (available after transform())."""
        return self._model_arch

    def parse(self) -> ComputationalGraph:
        """
        Phase 1: Parse SGLang model to ComputationalGraph IR.

        This method extracts the model structure and converts it to the
        framework-agnostic intermediate representation.

        Returns:
            ComputationalGraph containing all operators

        Raises:
            MissingDependencyError: If PyTorch is not installed
            ModelInstantiationError: If model cannot be instantiated
        """
        parser = ModelParser(self.config)
        self._ir_graph = parser.parse(self.model_class)
        return self._ir_graph

    def transform(self, schedule_config: ScheduleConfig) -> "BaseModelArch":
        """
        Phase 2: Transform IR to LLMSim ModelArch.

        This method converts the intermediate representation to an actual
        ModelArch instance with properly configured operators.

        Args:
            schedule_config: Runtime scheduling configuration

        Returns:
            Configured ModelArch instance

        Raises:
            ValueError: If parse() has not been called first
        """
        if self._ir_graph is None:
            raise ValueError(
                "IR graph not available. Call parse() first, "
                "or use adapt() for one-step conversion."
            )

        transformer = IRToModelArchTransformer(self._ir_graph, schedule_config)
        self._model_arch = transformer.transform()
        return self._model_arch

    def adapt(self, schedule_config: ScheduleConfig) -> "BaseModelArch":
        """
        One-step adaptation: Parse + Transform.

        This is a convenience method that performs both phases in sequence.

        Args:
            schedule_config: Runtime scheduling configuration

        Returns:
            Configured ModelArch instance
        """
        self.parse()
        return self.transform(schedule_config)

    def get_ir_summary(self) -> dict:
        """
        Get a summary of the IR graph for debugging.

        Returns:
            Dictionary with IR statistics
        """
        if self._ir_graph is None:
            return {"error": "IR graph not available. Call parse() first."}

        ir = self._ir_graph
        return {
            "model_name": ir.model_name,
            "model_type": ir.model_type,
            "total_nodes": len(ir.nodes),
            "matmul_nodes": len(ir.get_matmul_nodes()),
            "attention_nodes": len(ir.get_attention_nodes()),
            "transfer_nodes": len(ir.get_transfer_nodes()),
            "has_moe": ir.has_moe,
            "has_mla": ir.has_mla,
            "kv_cache_type": ir.kv_cache_type,
            "validation_errors": ir.validate(),
        }


def auto_adapter(
    model_class: Type,
    config: Any,
    schedule_config: ScheduleConfig,
) -> "BaseModelArch":
    """
    Convenience function for one-step model adaptation.

    This is the simplest way to adapt a SGLang model to LLMSim.

    Args:
        model_class: SGLang model class (e.g., Qwen3MoeForCausalLM)
        config: Model configuration object
        schedule_config: Runtime scheduling configuration

    Returns:
        Configured ModelArch instance

    Example:
        from transformers import AutoConfig
        from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
        from src.arch.config import ScheduleConfig, ForwardMode

        config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
        schedule_config = ScheduleConfig(
            mode=ForwardMode.EXTEND,
            tp_size=4,
            dp_size=2,
            ep_size=8,
        )

        model_arch = auto_adapter(Qwen3MoeForCausalLM, config, schedule_config)
    """
    adapter = SglangAutoAdapter(model_class, config)
    return adapter.adapt(schedule_config)


def parse_model(
    model_class: Type,
    config: Any,
) -> ComputationalGraph:
    """
    Parse SGLang model to IR without transforming to ModelArch.

    This is useful for debugging, visualization, or caching the IR.

    Args:
        model_class: SGLang model class
        config: Model configuration object

    Returns:
        ComputationalGraph IR

    Example:
        ir_graph = parse_model(Qwen3MoeForCausalLM, config)

        # Inspect the IR
        for node in ir_graph.nodes:
            print(f"{node.name}: {node.op_type}")

        # Save for later
        import json
        with open("model_ir.json", "w") as f:
            json.dump(ir_graph.to_dict(), f, indent=2)
    """
    adapter = SglangAutoAdapter(model_class, config)
    return adapter.parse()
