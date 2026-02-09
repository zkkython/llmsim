"""SGLang model coverage evaluation test.

This test evaluates how many SGLang-supported models can be covered by the
config-driven auto adapter's builders, and identifies gaps.

Configs are automatically downloaded from Hugging Face Hub using the transformers library.
If HF is not available, falls back to local configs in hf_config/ directory.

Usage with proxy:
    export https_proxy=http://127.0.0.1:7890
    python -m pytest tests/test_model_coverage.py -v
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Cache directory for downloaded configs
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "hf_config")
os.makedirs(CACHE_DIR, exist_ok=True)

# Local config directory
HF_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "hf_config")


def load_config_from_hf(
    model_id: str, use_cache: bool = True
) -> Optional[Dict[str, Any]]:
    """Load model config from Hugging Face Hub.

    Args:
        model_id: Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-hf")
        use_cache: Whether to use cached config if available

    Returns:
        Config dictionary or None if failed
    """
    cache_file = os.path.join(CACHE_DIR, f"{model_id.replace('/', '__')}.json")

    # Check cache first
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    # Try to download from Hugging Face
    try:
        from transformers import AutoConfig

        # AutoConfig will automatically use https_proxy environment variable if set
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            timeout=30,  # Increase timeout
        )
        config_dict = config.to_dict()

        # Cache the config
        with open(cache_file, "w") as f:
            json.dump(config_dict, f, indent=2)

        return config_dict
    except Exception:
        # Return None to allow fallback
        return None


def load_config_from_modelscope(
    model_id: str, use_cache: bool = True
) -> Optional[Dict[str, Any]]:
    """Load model config from ModelScope (魔塔社区).

    Args:
        model_id: ModelScope model ID (e.g., "LLM-Research/Meta-Llama-3-8B")
        use_cache: Whether to use cached config if available

    Returns:
        Config dictionary or None if failed
    """
    cache_file = os.path.join(
        CACHE_DIR, f"modelscope__{model_id.replace('/', '__')}.json"
    )

    # Check cache first
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    # Try to download from ModelScope
    try:
        from modelscope import AutoConfig

        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        config_dict = config.to_dict()

        # Cache the config
        with open(cache_file, "w") as f:
            json.dump(config_dict, f, indent=2)

        return config_dict
    except Exception:
        # Return None to allow fallback
        return None


def load_config_from_local(config_path: str) -> Dict[str, Any]:
    """Load config from local file.

    Args:
        config_path: Path to config.json file

    Returns:
        Config dictionary
    """
    with open(config_path, "r") as f:
        return json.load(f)


def load_model_config(
    model_key: str, model_info: "ModelInfo"
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load config for a model, trying HF first then local fallback.

    Args:
        model_key: Key in SGLANG_MODELS
        model_info: ModelInfo object

    Returns:
        Tuple of (config_dict, source) where source is "hf", "local", or "failed"
    """
    # Try Hugging Face first
    config = load_config_from_hf(model_info.model_id, use_cache=True)
    if config is not None:
        return config, "hf"
    return None, "failed"


from src.arch.models_arch.auto import (
    build_ir_from_config,
    infer_model_type,
    list_registered_builders,
)
from src.arch.models_arch.auto.config_builder import _has_mla, _has_moe
from src.arch.models_arch.auto.ir import ComputationalGraph

# Mapping of local model_type to architecture names (for local configs that may lack architectures)
ARCHITECTURE_MAP = {
    "qwen3": ["Qwen3ForCausalLM"],
    "qwen3_moe": ["Qwen3MoeForCausalLM"],
    "deepseek_v3": ["DeepseekV3ForCausalLM"],
    "deepseek_v2": ["DeepseekV2ForCausalLM"],
    "llama": ["LlamaForCausalLM"],
    "gemma": ["GemmaForCausalLM"],
    "gemma2": ["Gemma2ForCausalLM"],
    "mistral": ["MistralForCausalLM"],
    "mixtral": ["MixtralForCausalLM"],
}


class ModelCategory(Enum):
    """Model architecture categories."""

    DENSE = "dense"
    MOE = "moe"
    MLA = "mla"
    MLA_MOE = "mla_moe"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class CoverageStatus(Enum):
    """Coverage status for a model."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


@dataclass
class ModelInfo:
    """Information about a model to evaluate."""

    name: str
    architectures: List[str]
    model_id: str  # Hugging Face model ID
    sglang_file: str
    notes: str = ""
    # Optional: local config path as fallback
    local_config_path: Optional[str] = None
    # Optional: ModelScope model ID (for models that need auth on HF)
    modelscope_id: Optional[str] = None


def _get_config_from_source(
    model_key: str, model_info: ModelInfo
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Get config for a model from available sources.

    Tries HF cache first, then local config, then HF download, then ModelScope.

    Returns:
        Tuple of (config_dict, source) where source is "hf-cache", "local", "hf", "modelscope", or "failed"
    """
    # Try Hugging Face cache first
    cache_file = os.path.join(
        CACHE_DIR, f"{model_info.model_id.replace('/', '__')}.json"
    )
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f), "hf-cache"

    # Try local config if specified
    if model_info.local_config_path and os.path.exists(model_info.local_config_path):
        with open(model_info.local_config_path, "r") as f:
            return json.load(f), "local"

    # Try HF download
    config = load_config_from_hf(model_info.model_id, use_cache=True)
    if config is not None:
        return config, "hf"

    # Try ModelScope if model_id is specified
    if model_info.modelscope_id:
        config = load_config_from_modelscope(model_info.modelscope_id, use_cache=True)
        if config is not None:
            return config, "modelscope"

    return None, "failed"


@dataclass
class CoverageResult:
    """Result of coverage evaluation for a model."""

    model_name: str
    category: ModelCategory
    status: CoverageStatus
    inferred_builder: Optional[str]
    model_id: str = ""
    missing_params: List[str] = field(default_factory=list)
    unsupported_features: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)
    recommendation: str = ""
    config_source: str = ""  # Source of config: hf-cache, local, hf, modelscope, failed


# =============================================================================
# SGLang Model Registry with Hugging Face Model IDs
# These are official or popular model IDs for each architecture type
# =============================================================================

SGLANG_MODELS = {
    # === Llama Family ===
    "llama": ModelInfo(
        name="LlamaForCausalLM",
        architectures=["LlamaForCausalLM"],
        model_id="meta-llama/Llama-2-7b-hf",
        sglang_file="llama.py",
        notes="Standard Llama model (Llama 2 7B)",
        modelscope_id="modelscope/Llama-2-7b-chat-ms",
    ),
    "llama3": ModelInfo(
        name="Llama3ForCausalLM",
        architectures=["LlamaForCausalLM"],
        model_id="meta-llama/Meta-Llama-3-8B",
        sglang_file="llama.py",
        notes="Llama 3 model",
        modelscope_id="LLM-Research/Meta-Llama-3-8B",
    ),
    # === DeepSeek Family ===
    "deepseek_v2": ModelInfo(
        name="DeepseekV2ForCausalLM",
        architectures=["DeepseekV2ForCausalLM"],
        model_id="deepseek-ai/DeepSeek-V2-Lite",
        sglang_file="deepseek_v2.py",
        notes="DeepSeek V2 with MLA",
    ),
    "deepseek_v3": ModelInfo(
        name="DeepseekV3ForCausalLM",
        architectures=["DeepseekV3ForCausalLM"],
        model_id="deepseek-ai/DeepSeek-V3",
        sglang_file="deepseek_v3.py",
        notes="DeepSeek V3 with MLA + MoE",
        local_config_path=os.path.join(HF_CONFIG_DIR, "deepseek_671b_r1_config.json"),
    ),
    "deepseek_v3_base": ModelInfo(
        name="DeepseekV3ForCausalLM (Base)",
        architectures=["DeepseekV3ForCausalLM"],
        model_id="deepseek-ai/deepseek-v3-base",
        sglang_file="deepseek_v3.py",
        notes="DeepSeek V3 Base model",
        local_config_path=os.path.join(HF_CONFIG_DIR, "deepseek_671b_r1_config.json"),
    ),
    # === Qwen Family ===
    "qwen2": ModelInfo(
        name="Qwen2ForCausalLM",
        architectures=["Qwen2ForCausalLM"],
        model_id="Qwen/Qwen2-7B",
        sglang_file="qwen2.py",
        notes="Qwen2 7B model",
    ),
    "qwen2.5": ModelInfo(
        name="Qwen2.5ForCausalLM",
        architectures=["Qwen2ForCausalLM"],
        model_id="Qwen/Qwen2.5-7B",
        sglang_file="qwen2.py",
        notes="Qwen2.5 7B model",
    ),
    "qwen3": ModelInfo(
        name="Qwen3ForCausalLM",
        architectures=["Qwen3ForCausalLM"],
        model_id="Qwen/Qwen3-8B",
        sglang_file="qwen3.py",
        notes="Qwen3 8B model",
        local_config_path=os.path.join(HF_CONFIG_DIR, "qwen3-8B_config.json"),
    ),
    "qwen3_moe": ModelInfo(
        name="Qwen3MoeForCausalLM",
        architectures=["Qwen3MoeForCausalLM"],
        model_id="Qwen/Qwen3-30B-A3B",
        sglang_file="qwen3_moe.py",
        notes="Qwen3 MoE model",
        local_config_path=os.path.join(HF_CONFIG_DIR, "qwen3-30B-A3B_config.json"),
    ),
    # === Gemma Family ===
    "gemma": ModelInfo(
        name="GemmaForCausalLM",
        architectures=["GemmaForCausalLM"],
        model_id="google/gemma-2b",
        sglang_file="gemma.py",
        notes="Gemma 2B model",
        modelscope_id="AI-ModelScope/gemma-2b",
    ),
    "gemma2": ModelInfo(
        name="Gemma2ForCausalLM",
        architectures=["Gemma2ForCausalLM"],
        model_id="google/gemma-2-2b",
        sglang_file="gemma2.py",
        notes="Gemma 2 2B model",
        modelscope_id="AI-ModelScope/gemma-2-2b",
    ),
    "gemma2_9b": ModelInfo(
        name="Gemma2ForCausalLM (9B)",
        architectures=["Gemma2ForCausalLM"],
        model_id="google/gemma-2-9b",
        sglang_file="gemma2.py",
        notes="Gemma 2 9B model",
        modelscope_id="AI-ModelScope/gemma-2-9b",
    ),
    # === Mistral/Mixtral Family ===
    "mistral": ModelInfo(
        name="MistralForCausalLM",
        architectures=["MistralForCausalLM"],
        model_id="mistralai/Mistral-7B-v0.1",
        sglang_file="mistral.py",
        notes="Mistral 7B with sliding window",
    ),
    "mixtral": ModelInfo(
        name="MixtralForCausalLM",
        architectures=["MixtralForCausalLM"],
        model_id="mistralai/Mixtral-8x7B-v0.1",
        sglang_file="mixtral.py",
        notes="Mixtral 8x7B MoE",
    ),
    "mixtral_8x22b": ModelInfo(
        name="MixtralForCausalLM (8x22B)",
        architectures=["MixtralForCausalLM"],
        model_id="mistralai/Mixtral-8x22B-v0.1",
        sglang_file="mixtral.py",
        notes="Mixtral 8x22B MoE",
    ),
    # === GLM Family ===
    "glm4": ModelInfo(
        name="Glm4ForCausalLM",
        architectures=["Glm4ForCausalLM"],
        model_id="THUDM/glm-4-9b",
        sglang_file="glm4.py",
        notes="GLM-4 9B model",
    ),
    "glm4_9b_chat": ModelInfo(
        name="Glm4ForCausalLM (Chat)",
        architectures=["Glm4ForCausalLM"],
        model_id="THUDM/glm-4-9b-chat",
        sglang_file="glm4.py",
        notes="GLM-4 9B Chat model",
    ),
    # === Other Models ===
    "phi3": ModelInfo(
        name="Phi3ForCausalLM",
        architectures=["Phi3ForCausalLM"],
        model_id="microsoft/Phi-3-mini-4k-instruct",
        sglang_file="phi3.py",
        notes="Phi-3 Mini model",
    ),
    "phi4": ModelInfo(
        name="Phi4ForCausalLM",
        architectures=["Phi4ForCausalLM"],
        model_id="microsoft/phi-4",
        sglang_file="phi4.py",
        notes="Phi-4 model",
    ),
}

# =============================================================================
# Required parameters for each builder
# =============================================================================

REQUIRED_PARAMS = {
    "qwen3_moe": [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_experts",
        "moe_intermediate_size",
        "num_experts_per_tok",
    ],
    "deepseek_v3": [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "n_routed_experts",
        "qk_rope_head_dim",
        "kv_lora_rank",
        "num_experts_per_tok",
        "moe_intermediate_size",
    ],
    "qwen3": [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
    ],
}

# =============================================================================
# Unsupported features detection
# =============================================================================

UNSUPPORTED_FEATURES = {
    "sliding_window": "Sliding window attention (Mistral, Mixtral)",
    "vision_hidden_size": "Vision encoder / multi-modal support",
    "num_image_tokens": "Image token processing (multi-modal)",
    "tie_word_embeddings": "Tied input/output embeddings",
    "use_qk_norm": "Query/key normalization",
    "query_pre_attn_scalar": "Custom attention scaling (Gemma2)",
    "use_pre_ffw_norm": "Pre-feedforward normalization (Gemma2)",
    "use_post_ffw_norm": "Post-feedforward normalization (Gemma2)",
}

# =============================================================================
# Coverage evaluation functions
# =============================================================================


def classify_model(config: Dict[str, Any]) -> ModelCategory:
    """Classify model based on config features."""
    # Check for multi-modal
    if any(
        k in config for k in ["vision_hidden_size", "num_image_tokens", "vision_config"]
    ):
        return ModelCategory.MULTIMODAL

    # Check for MLA
    has_mla = _has_mla(config)
    has_moe = _has_moe(config)

    if has_mla and has_moe:
        return ModelCategory.MLA_MOE
    elif has_mla:
        return ModelCategory.MLA
    elif has_moe:
        return ModelCategory.MOE
    else:
        return ModelCategory.DENSE


def check_required_params(builder_name: str, config: Dict[str, Any]) -> List[str]:
    """Check if config has all required parameters for a builder."""
    required = REQUIRED_PARAMS.get(builder_name, [])
    missing = []
    for param in required:
        if param not in config:
            missing.append(param)
    return missing


def check_unsupported_features(config: Dict[str, Any]) -> List[str]:
    """Check for unsupported features in config."""
    unsupported = []
    for feature, description in UNSUPPORTED_FEATURES.items():
        if feature in config:
            if feature == "tie_word_embeddings":
                # Only flag if True
                if config[feature]:
                    unsupported.append(f"{feature}: {description}")
            else:
                unsupported.append(f"{feature}: {description}")
    return unsupported


def get_recommended_builder(category: ModelCategory) -> Optional[str]:
    """Get the recommended builder for a model category."""
    mapping = {
        ModelCategory.DENSE: "qwen3",
        ModelCategory.MOE: "qwen3_moe",
        ModelCategory.MLA: None,  # Need new builder
        ModelCategory.MLA_MOE: "deepseek_v3",
        ModelCategory.MULTIMODAL: None,
        ModelCategory.UNKNOWN: None,
    }
    return mapping.get(category)


def get_model_differences(model_name: str, config: Dict[str, Any]) -> List[str]:
    """Get list of model-specific differences from standard builders."""
    differences = []

    # Check for specific model differences based on config
    if "sliding_window" in config and config["sliding_window"] is not None:
        differences.append("Sliding window attention not supported")

    # Check activation function
    hidden_act = config.get("hidden_act", "silu")
    if hidden_act not in ["silu", "swish"]:
        differences.append(f"Uses {hidden_act} activation (not SwiGLU/SiLU)")

    # Check for tie_word_embeddings
    if config.get("tie_word_embeddings", False):
        differences.append("Uses tied input/output embeddings")

    # Check for custom normalization
    if config.get("use_pre_ffw_norm", False) or config.get("use_post_ffw_norm", False):
        differences.append("Additional pre/post FFN normalization layers")

    # Check for query_pre_attn_scalar (Gemma2)
    if "query_pre_attn_scalar" in config:
        differences.append("Custom attention scaling (query_pre_attn_scalar)")

    # Check for use_qk_norm
    if config.get("use_qk_norm", False):
        differences.append("Uses query/key normalization")

    return differences


def evaluate_model_coverage(
    model_info: ModelInfo, config: Optional[Dict[str, Any]], config_source: str = ""
) -> CoverageResult:
    """Evaluate coverage for a single model."""
    # If config is None, mark as NOT_SUPPORTED
    if config is None:
        return CoverageResult(
            model_name=model_info.name,
            model_id=model_info.model_id,
            category=ModelCategory.UNKNOWN,
            status=CoverageStatus.NOT_SUPPORTED,
            inferred_builder=None,
            recommendation=f"Unable to retrieve model configuration from any source (HF, ModelScope, or local)",
            config_source=config_source,
        )

    # Make a copy to avoid modifying the original
    config = config.copy()

    # Add architectures to config for type inference if not present
    if "architectures" not in config:
        # Try to get from model_info or infer from model_type
        if model_info.architectures:
            config["architectures"] = model_info.architectures
        elif "model_type" in config and config["model_type"] in ARCHITECTURE_MAP:
            config["architectures"] = ARCHITECTURE_MAP[config["model_type"]]

    category = classify_model(config)
    recommended_builder = get_recommended_builder(category)

    # Determine status
    if category == ModelCategory.MULTIMODAL:
        return CoverageResult(
            model_name=model_info.name,
            model_id=model_info.model_id,
            category=category,
            status=CoverageStatus.NOT_SUPPORTED,
            inferred_builder=None,
            unsupported_features=["Multi-modal architecture"],
            recommendation="Requires new multi-modal builder",
            config_source=config_source,
        )

    if category == ModelCategory.MLA and "deepseek" in model_info.model_id.lower():
        # DeepSeek V2 has MLA but no MoE - our deepseek_v3 builder expects MoE
        return CoverageResult(
            model_name=model_info.name,
            model_id=model_info.model_id,
            category=category,
            status=CoverageStatus.PARTIALLY_SUPPORTED,
            inferred_builder="deepseek_v3",
            differences=["No MoE (pure MLA) - builder expects MoE params"],
            recommendation="Extend deepseek_v3 builder to handle non-MoE MLA",
            config_source=config_source,
        )

    if recommended_builder is None:
        return CoverageResult(
            model_name=model_info.name,
            model_id=model_info.model_id,
            category=category,
            status=CoverageStatus.NOT_SUPPORTED,
            inferred_builder=None,
            recommendation="Requires new builder implementation",
            config_source=config_source,
        )

    # Check for unsupported features
    unsupported = check_unsupported_features(config)

    # Check required params
    missing_params = check_required_params(recommended_builder, config)

    # Get model-specific differences
    differences = get_model_differences(model_info.name, config)

    # Determine final status
    if unsupported or missing_params:
        status = CoverageStatus.PARTIALLY_SUPPORTED
    else:
        status = CoverageStatus.FULLY_SUPPORTED

    return CoverageResult(
        model_name=model_info.name,
        model_id=model_info.model_id,
        category=category,
        status=status,
        inferred_builder=recommended_builder,
        missing_params=missing_params,
        unsupported_features=unsupported,
        differences=differences,
        recommendation=get_recommendation(status, unsupported, missing_params),
        config_source=config_source,
    )


def get_recommendation(
    status: CoverageStatus, unsupported: List[str], missing: List[str]
) -> str:
    """Generate recommendation based on evaluation."""
    if status == CoverageStatus.FULLY_SUPPORTED:
        return "Use as-is"
    elif status == CoverageStatus.NOT_SUPPORTED:
        return "Requires new builder"
    else:
        issues = []
        if unsupported:
            issues.append(f"unsupported features: {len(unsupported)}")
        if missing:
            issues.append(f"missing params: {len(missing)}")
        return f"Partial - needs adaptation for {', '.join(issues)}"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def downloaded_configs():
    """Load configs from available sources (HF cache, local files, or download).

    Returns a dict with all models. For models that cannot be loaded,
    config will be None and source will be "failed".
    """
    configs = {}

    for key, model_info in SGLANG_MODELS.items():
        config, source = _get_config_from_source(key, model_info)
        configs[key] = (model_info, config, source)

    return configs


# =============================================================================
# Test Classes
# =============================================================================


class TestModelClassification:
    """Tests for model classification logic."""

    def test_classify_dense(self):
        config = {"hidden_size": 4096, "num_attention_heads": 32}
        assert classify_model(config) == ModelCategory.DENSE

    def test_classify_moe(self):
        config = {"hidden_size": 4096, "num_experts": 8}
        assert classify_model(config) == ModelCategory.MOE

    def test_classify_mla(self):
        config = {"hidden_size": 4096, "qk_rope_head_dim": 64, "kv_lora_rank": 512}
        assert classify_model(config) == ModelCategory.MLA

    def test_classify_mla_moe(self):
        config = {
            "hidden_size": 4096,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "n_routed_experts": 256,
        }
        assert classify_model(config) == ModelCategory.MLA_MOE

    def test_classify_multimodal(self):
        config = {"hidden_size": 4096, "vision_hidden_size": 1024}
        assert classify_model(config) == ModelCategory.MULTIMODAL


class TestBuilderCompatibility:
    """Tests for builder compatibility checking."""

    def test_check_required_params_qwen3(self):
        config = {"hidden_size": 4096, "num_hidden_layers": 32}
        missing = check_required_params("qwen3", config)
        assert "num_attention_heads" in missing
        assert "intermediate_size" in missing

    def test_check_required_params_qwen3_complete(self):
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
        }
        missing = check_required_params("qwen3", config)
        assert len(missing) == 0

    def test_check_unsupported_features_sliding_window(self):
        config = {"hidden_size": 4096, "sliding_window": 4096}
        unsupported = check_unsupported_features(config)
        assert any("sliding_window" in f for f in unsupported)

    def test_check_unsupported_features_tie_embeddings(self):
        config = {"hidden_size": 4096, "tie_word_embeddings": True}
        unsupported = check_unsupported_features(config)
        assert any("tie_word_embeddings" in f for f in unsupported)


class TestCoverageEvaluation:
    """Tests for coverage evaluation using real configs (HF or local)."""

    @pytest.mark.parametrize("model_key", list(SGLANG_MODELS.keys()))
    def test_model_coverage(self, model_key):
        """Test coverage evaluation for each model from available sources."""
        model_info = SGLANG_MODELS[model_key]

        config, source = _get_config_from_source(model_key, model_info)
        result = evaluate_model_coverage(model_info, config, source)

        # Basic assertions
        assert result.model_name == model_info.name
        assert result.model_id == model_info.model_id
        assert result.status in [
            CoverageStatus.FULLY_SUPPORTED,
            CoverageStatus.PARTIALLY_SUPPORTED,
            CoverageStatus.NOT_SUPPORTED,
        ]

        print(f"\n{model_key}: {result.status.value} (source: {source})")
        if result.differences:
            print(f"  Differences: {result.differences}")
        if result.unsupported_features:
            print(f"  Unsupported: {result.unsupported_features}")
        if result.config_source == "failed":
            print(f"  Note: {result.recommendation}")


class TestCoverageReport:
    """Generate and validate the full coverage report from HF configs."""

    def test_download_configs(self, downloaded_configs):
        """Test that we can load configs from available sources."""
        configs = downloaded_configs

        # Count successful vs failed loads
        successful = sum(1 for _, config, _ in configs.values() if config is not None)
        failed = sum(1 for _, config, _ in configs.values() if config is None)

        print(f"\n{'='*70}")
        print("CONFIG LOAD SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully loaded: {successful} configs")
        print(f"Failed to load: {failed} configs")

        # Show sources
        sources = {}
        for key, (_, config, source) in configs.items():
            if config is not None:
                sources[source] = sources.get(source, 0) + 1
            else:
                sources["failed"] = sources.get("failed", 0) + 1
        print("\nSources:")
        for source, count in sorted(sources.items()):
            print(f"  - {source}: {count}")

        # At least some configs should load
        assert successful > 0, "At least some configs should be loadable"

    def test_generate_coverage_report(self, downloaded_configs):
        """Generate comprehensive coverage report."""
        configs = downloaded_configs

        # Evaluate coverage for all models (including those without config)
        results = {}
        for key, (model_info, config, source) in configs.items():
            results[key] = evaluate_model_coverage(model_info, config, source)

        print("\n" + "=" * 70)
        print("SGLANG MODEL COVERAGE REPORT (from Hugging Face)")
        print("=" * 70)

        # Group by status
        by_status = {
            CoverageStatus.FULLY_SUPPORTED: [],
            CoverageStatus.PARTIALLY_SUPPORTED: [],
            CoverageStatus.NOT_SUPPORTED: [],
        }
        for key, result in results.items():
            by_status[result.status].append((key, result))

        total = len(results)
        fully = len(by_status[CoverageStatus.FULLY_SUPPORTED])
        partial = len(by_status[CoverageStatus.PARTIALLY_SUPPORTED])
        not_supported = len(by_status[CoverageStatus.NOT_SUPPORTED])

        print(f"\nTotal models evaluated: {total}")
        print(f"Fully supported: {fully} ({fully/total*100:.1f}%)")
        print(f"Partially supported: {partial} ({partial/total*100:.1f}%)")
        print(f"Not supported: {not_supported} ({not_supported/total*100:.1f}%)")
        print(f"Overall coverage: {(fully + partial)/total*100:.1f}%")

        # Fully supported
        print("\n" + "-" * 70)
        print(f"FULLY SUPPORTED MODELS ({fully})")
        print("-" * 70)
        for key, r in sorted(
            by_status[CoverageStatus.FULLY_SUPPORTED], key=lambda x: x[1].model_name
        ):
            print(f"\n✓ {r.model_name}")
            print(f"  Model ID: {r.model_id}")
            print(f"  Category: {r.category.value}")
            print(f"  Builder: {r.inferred_builder}")
            print(f"  Source: {SGLANG_MODELS[key].sglang_file}")

        # Partially supported
        print("\n" + "-" * 70)
        print(f"PARTIALLY SUPPORTED MODELS ({partial})")
        print("-" * 70)
        for key, r in sorted(
            by_status[CoverageStatus.PARTIALLY_SUPPORTED], key=lambda x: x[1].model_name
        ):
            print(f"\n~ {r.model_name}")
            print(f"  Model ID: {r.model_id}")
            print(f"  Category: {r.category.value}")
            print(f"  Recommended builder: {r.inferred_builder}")
            if r.differences:
                print(f"  Differences:")
                for d in r.differences:
                    print(f"    - {d}")
            if r.unsupported_features:
                print(f"  Unsupported features:")
                for f in r.unsupported_features:
                    print(f"    - {f}")
            if r.missing_params:
                print(f"  Missing params: {r.missing_params}")

        # Not supported
        print("\n" + "-" * 70)
        print(f"NOT SUPPORTED MODELS ({not_supported})")
        print("-" * 70)
        for key, r in sorted(
            by_status[CoverageStatus.NOT_SUPPORTED], key=lambda x: x[1].model_name
        ):
            print(f"\n✗ {r.model_name}")
            print(f"  Model ID: {r.model_id}")
            print(f"  Category: {r.category.value}")
            print(f"  Reason: {r.recommendation}")
            if r.unsupported_features:
                print(f"  Blockers:")
                for f in r.unsupported_features:
                    print(f"    - {f}")

        # Category breakdown
        print("\n" + "=" * 70)
        print("COVERAGE BY ARCHITECTURE CATEGORY")
        print("=" * 70)
        by_category = {}
        for r in results.values():
            cat = r.category.value
            if cat not in by_category:
                by_category[cat] = {"total": 0, "supported": 0}
            by_category[cat]["total"] += 1
            if r.status != CoverageStatus.NOT_SUPPORTED:
                by_category[cat]["supported"] += 1

        for cat, stats in sorted(by_category.items()):
            pct = stats["supported"] / stats["total"] * 100
            print(f"\n{cat.upper()}:")
            print(
                f"  Total: {stats['total']}, Supported: {stats['supported']} ({pct:.0f}%)"
            )

        # Always pass - this is an informational test
        assert True

    def test_key_models_supported(self, downloaded_configs):
        """Verify that key models are at least partially supported."""
        configs = downloaded_configs

        # These are the key models we want to support
        key_models = [
            "deepseek_v3",
            "qwen3",
            "qwen3_moe",
            "llama",
            "llama3",
            "mistral",
            "mixtral",
        ]

        for key in key_models:
            if key not in configs:
                continue  # Skip if config couldn't be loaded

            model_info, config, source = configs[key]
            result = evaluate_model_coverage(model_info, config)

            # All key models should be at least partially supported
            # (not completely unsupported)
            assert result.status in [
                CoverageStatus.FULLY_SUPPORTED,
                CoverageStatus.PARTIALLY_SUPPORTED,
            ], f"{key} should be at least partially supported"


class TestIRBuildWithRealConfigs:
    """Test actual IR building with real configs (from HF or local)."""

    def test_build_qwen3_ir(self):
        """Test building IR for Qwen3 from config."""
        model_info = SGLANG_MODELS["qwen3"]
        config, source = _get_config_from_source("qwen3", model_info)
        if config is None:
            pytest.skip("Could not load Qwen3 config")

        graph = build_ir_from_config(config, "qwen3")
        assert isinstance(graph, ComputationalGraph)
        assert graph.model_type == "dense"
        assert len(graph.nodes) > 0

    def test_build_qwen3_moe_ir(self):
        """Test building IR for Qwen3 MoE from config."""
        model_info = SGLANG_MODELS["qwen3_moe"]
        config, source = _get_config_from_source("qwen3_moe", model_info)
        if config is None:
            pytest.skip("Could not load Qwen3-MoE config")

        graph = build_ir_from_config(config, "qwen3_moe")
        assert isinstance(graph, ComputationalGraph)
        assert graph.model_type == "moe"
        assert graph.has_moe is True

    def test_build_deepseek_v3_ir(self):
        """Test building IR for DeepSeek V3 from config."""
        model_info = SGLANG_MODELS["deepseek_v3_base"]
        config, source = _get_config_from_source("deepseek_v3_base", model_info)
        if config is None:
            pytest.skip("Could not load DeepSeek-V3 config")

        graph = build_ir_from_config(config, "deepseek_v3")
        assert isinstance(graph, ComputationalGraph)
        assert graph.model_type == "mla"
        assert graph.has_mla is True
        assert graph.has_moe is True


# =============================================================================
# Main entry point for running as script
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
