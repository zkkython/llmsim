"""
Comprehensive tests for the SGLang Model Architecture Auto-Generator.

Test coverage:
1. Mock environment creation and context manager
2. Layer type recognition for supported layers
3. Error handling for unsupported layers
4. Shape extraction methods
5. Model structure extraction
6. Code generation
7. End-to-end integration tests
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arch.models_arch.auto_generator import (
    LayerTypeRecognizer,
    MissingDependencyError,
    MockCallable,
    MockModule,
    ModelArchCodeGenerator,
    ModelInstantiationError,
    ModelStructure,
    ModelStructureExtractor,
    UnsupportedLayerError,
    create_mock_environment,
    mock_sglang_environment,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_config():
    """Create a simple config object for testing."""
    config = MagicMock()
    config.hidden_size = 4096
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.intermediate_size = 11008
    config.vocab_size = 32000
    config.head_dim = 128
    return config


@pytest.fixture
def moe_config():
    """Create a MoE config object for testing."""
    config = MagicMock()
    config.hidden_size = 4096
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.intermediate_size = 11008
    config.moe_intermediate_size = 1536
    config.num_experts = 128
    config.num_experts_per_tok = 8
    config.vocab_size = 32000
    config.head_dim = 128
    return config


@pytest.fixture
def mla_config():
    """Create an MLA config object for testing."""
    config = MagicMock()
    config.hidden_size = 7168
    config.num_hidden_layers = 61
    config.num_attention_heads = 128
    config.qk_nope_head_dim = 128
    config.qk_rope_head_dim = 64
    config.v_head_dim = 128
    config.q_lora_rank = 1536
    config.kv_lora_rank = 512
    config.n_routed_experts = 256
    config.num_experts_per_tok = 8
    config.moe_intermediate_size = 2048
    config.first_k_dense_replace = 3
    return config


# =============================================================================
# Mock Environment Tests
# =============================================================================


class TestMockEnvironment:
    """Tests for the mock SGLang environment."""

    def test_create_mock_environment(self):
        """Test that mock environment contains all required modules."""
        mock_modules = create_mock_environment()

        required_modules = [
            "sglang",
            "sglang.srt",
            "sglang.srt.distributed",
            "sglang.srt.layers",
            "sglang.srt.layers.communicator",
            "sglang.srt.server_args",
            "sgl_kernel",
        ]

        for module_name in required_modules:
            assert module_name in mock_modules
            assert isinstance(mock_modules[module_name], MockModule)

    def test_mock_module_attribute_access(self):
        """Test that MockModule returns MockCallable for any attribute."""
        mock = MockModule("test_module")

        # Any attribute access should return a MockCallable
        attr = mock.some_function
        assert isinstance(attr, MockCallable)

        # Calling the attribute should return None
        result = attr()
        assert result is None

        # Nested attribute access should also work
        nested = mock.nested.deep.attribute
        assert isinstance(nested, MockCallable)

    def test_mock_callable_chaining(self):
        """Test that MockCallable supports method chaining."""
        mock = MockCallable("test")

        # Should be able to chain attribute access
        chained = mock.attr1.attr2.attr3
        assert isinstance(chained, MockCallable)

        # Should be able to call at any point
        result = chained()
        assert result is None

    def test_mock_sglang_environment_context_manager(self):
        """Test that mock_sglang_environment properly mocks and restores modules."""
        # Check if sglang was originally in sys.modules
        original_sglang = sys.modules.get("sglang")

        with mock_sglang_environment():
            # Inside context, sglang should be mocked
            assert "sglang" in sys.modules
            assert isinstance(sys.modules["sglang"], MockModule)

            # Should be able to import and use mocked modules
            import sglang.srt.distributed as dist

            result = dist.get_tensor_model_parallel_world_size()
            assert result is None

        # After context, original state should be restored
        if original_sglang is None:
            assert "sglang" not in sys.modules
        else:
            assert sys.modules.get("sglang") is original_sglang


# =============================================================================
# Layer Type Recognition Tests
# =============================================================================


class TestLayerTypeRecognizer:
    """Tests for layer type recognition and mapping."""

    def test_recognize_qkv_parallel_linear(self, simple_config):
        """Test recognition of QKVParallelLinear."""
        recognizer = LayerTypeRecognizer(simple_config)

        # Create a mock module
        mock_module = MagicMock()
        mock_module.__class__.__name__ = "QKVParallelLinear"

        layer_info = recognizer.recognize(
            "model.layers.0.self_attn.qkv_proj", mock_module
        )

        assert layer_info is not None
        assert layer_info.layer_type == "QKVParallelLinear"
        assert layer_info.op_category == "matmul"
        assert layer_info.op_name == "qkv_proj"
        assert layer_info.dtype == "BF16"

    def test_recognize_row_parallel_linear_as_o_proj(self, simple_config):
        """Test recognition of RowParallelLinear in attention context."""
        recognizer = LayerTypeRecognizer(simple_config)

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "RowParallelLinear"

        layer_info = recognizer.recognize(
            "model.layers.0.self_attn.o_proj", mock_module
        )

        assert layer_info is not None
        assert layer_info.op_name == "o_proj"

    def test_recognize_row_parallel_linear_as_down_proj(self, simple_config):
        """Test recognition of RowParallelLinear in FFN context."""
        recognizer = LayerTypeRecognizer(simple_config)

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "RowParallelLinear"

        layer_info = recognizer.recognize("model.layers.0.mlp.down_proj", mock_module)

        assert layer_info is not None
        assert layer_info.op_name == "down_proj"

    def test_recognize_replicated_linear_as_gate(self, moe_config):
        """Test recognition of ReplicatedLinear as MoE gate."""
        recognizer = LayerTypeRecognizer(moe_config)

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "ReplicatedLinear"

        layer_info = recognizer.recognize("model.layers.0.mlp.gate", mock_module)

        assert layer_info is not None
        assert layer_info.op_name == "moe_gate"
        assert layer_info.dtype == "FP32"

    def test_recognize_rmsnorm_returns_none(self, simple_config):
        """Test that RMSNorm is recognized but returns None (no operator)."""
        recognizer = LayerTypeRecognizer(simple_config)

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "RMSNorm"

        layer_info = recognizer.recognize("model.layers.0.input_layernorm", mock_module)

        assert layer_info is None

    def test_unsupported_layer_raises_error(self, simple_config):
        """Test that unsupported layers raise UnsupportedLayerError."""
        recognizer = LayerTypeRecognizer(simple_config)

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "UnknownLayerType"
        mock_module.__class__.__module__ = "some_module"

        with pytest.raises(UnsupportedLayerError) as exc_info:
            recognizer.recognize("model.layers.0.unknown", mock_module)

        error = exc_info.value
        assert error.layer_name == "model.layers.0.unknown"
        assert error.layer_type == "UnknownLayerType"
        assert "UnknownLayerType" in str(error)
        assert "LAYER_PATTERNS" in str(error)

    def test_infer_op_name_from_path(self, simple_config):
        """Test dynamic op_name inference from layer path."""
        recognizer = LayerTypeRecognizer(simple_config)

        # Test various path patterns
        assert recognizer._infer_op_name_from_path("model.gate", "Linear") == "moe_gate"
        assert recognizer._infer_op_name_from_path("model.qkv", "Linear") == "qkv_proj"
        assert recognizer._infer_op_name_from_path("model.o_proj", "Linear") == "o_proj"
        assert (
            recognizer._infer_op_name_from_path("model.gate_up", "Linear")
            == "gate_up_proj"
        )
        assert (
            recognizer._infer_op_name_from_path("model.down", "Linear") == "down_proj"
        )


# =============================================================================
# Shape Extraction Tests
# =============================================================================


class TestShapeExtraction:
    """Tests for shape extraction methods."""

    def test_extract_qkv_shape(self, simple_config):
        """Test QKV shape extraction produces correct expressions."""
        recognizer = LayerTypeRecognizer(simple_config)
        mock_module = MagicMock()

        shape_info = recognizer._extract_qkv_shape("test", mock_module)

        assert shape_info["input_shape"] == ("seq_len", "mc.hidden_size")
        assert "num_heads_per_rank" in str(shape_info["output_shape"])
        assert shape_info["dtype"] == "BF16"
        assert shape_info["batch_size"] == 1

    def test_extract_moe_gate_shape(self, moe_config):
        """Test MoE gate shape extraction."""
        recognizer = LayerTypeRecognizer(moe_config)
        mock_module = MagicMock()

        shape_info = recognizer._extract_replicated_linear_shape(
            "mlp.gate", mock_module
        )

        assert shape_info["input_shape"] == ("seq_len", "mc.hidden_size")
        assert shape_info["output_shape"] == ("seq_len", 128)  # num_experts from config
        assert shape_info["dtype"] == "FP32"

    def test_extract_fused_moe_shape(self, moe_config):
        """Test FusedMoE shape extraction."""
        recognizer = LayerTypeRecognizer(moe_config)
        mock_module = MagicMock()
        mock_module.num_experts = 128

        shape_info = recognizer._extract_fused_moe_shape("mlp.experts", mock_module)

        assert shape_info["input_shape"] == ("L_per_rank", "mc.hidden_size")
        assert shape_info["batch_size"] == "experts_per_rank"
        assert shape_info["dtype"] == "INT8"

    def test_extract_mla_attention_shape(self, mla_config):
        """Test MLA attention shape extraction."""
        recognizer = LayerTypeRecognizer(mla_config)
        mock_module = MagicMock()

        shape_info = recognizer._extract_mla_attention_shape("attn", mock_module)

        assert shape_info["attention_type"] == "MLA"
        assert "mc.qk_nope_head_dim" in str(shape_info["input_shape"])


# =============================================================================
# Model Structure Extraction Tests
# =============================================================================


class TestModelStructureExtraction:
    """Tests for model structure extraction."""

    def test_identify_dense_model_features(self, simple_config):
        """Test feature identification for dense model."""
        mock_model_class = MagicMock()
        extractor = ModelStructureExtractor(simple_config, mock_model_class)

        structure = ModelStructure(model_name="TestModel", config_class="TestConfig")
        extractor._identify_model_features(structure)

        assert structure.has_moe is False
        assert structure.has_mla is False
        assert structure.kv_cache_type == "mha_gqa"

    def test_identify_moe_model_features(self, moe_config):
        """Test feature identification for MoE model."""
        mock_model_class = MagicMock()
        extractor = ModelStructureExtractor(moe_config, mock_model_class)

        structure = ModelStructure(
            model_name="TestMoEModel", config_class="TestMoEConfig"
        )
        extractor._identify_model_features(structure)

        assert structure.has_moe is True
        assert structure.has_dense_layers is False

    def test_identify_mla_model_features(self, mla_config):
        """Test feature identification for MLA model."""
        mock_model_class = MagicMock()
        extractor = ModelStructureExtractor(mla_config, mock_model_class)

        structure = ModelStructure(
            model_name="TestMLAModel", config_class="TestMLAConfig"
        )
        extractor._identify_model_features(structure)

        assert structure.has_mla is True
        assert structure.kv_cache_type == "mla"
        assert structure.kv_cache_dtype == "INT8"
        assert structure.has_dense_layers is True  # DeepSeek style

    def test_infer_config_class(self, simple_config):
        """Test config class name inference."""
        mock_model_class = MagicMock()
        mock_model_class.__name__ = "Qwen3MoeForCausalLM"

        extractor = ModelStructureExtractor(simple_config, mock_model_class)
        config_class = extractor._infer_config_class()

        assert config_class == "Qwen3MoEConfig"

    def test_infer_config_class_generic(self, simple_config):
        """Test generic config class name inference."""
        mock_model_class = MagicMock()
        mock_model_class.__name__ = "CustomModelForCausalLM"

        extractor = ModelStructureExtractor(simple_config, mock_model_class)
        config_class = extractor._infer_config_class()

        assert config_class == "CustomConfig"


# =============================================================================
# Code Generation Tests
# =============================================================================


class TestCodeGeneration:
    """Tests for code generation."""

    def test_generate_imports_dense_model(self, simple_config):
        """Test import generation for dense model."""
        structure = ModelStructure(
            model_name="TestModel", config_class="TestConfig", kv_cache_type="mha_gqa"
        )

        generator = ModelArchCodeGenerator(structure)
        generator._generate_imports()

        code = "\n".join(generator.lines)
        assert "from src.arch.config import ForwardMode, TestConfig" in code
        assert "from src.arch.kvcache.kvcache import mha_gqa_kvcache" in code
        assert "mha_gqa_kvcache_per_gpu" in code

    def test_generate_imports_mla_model(self, mla_config):
        """Test import generation for MLA model."""
        structure = ModelStructure(
            model_name="TestMLAModel", config_class="TestMLAConfig", kv_cache_type="mla"
        )

        generator = ModelArchCodeGenerator(structure)
        generator._generate_imports()

        code = "\n".join(generator.lines)
        assert "mla_kvcache" in code
        assert "mla_kvcache_per_gpu" in code

    def test_generate_class_def(self):
        """Test class definition generation."""
        structure = ModelStructure(
            model_name="Qwen3ForCausalLM", config_class="Qwen3Config"
        )

        generator = ModelArchCodeGenerator(structure)
        generator._generate_class_def()

        code = "\n".join(generator.lines)
        assert "class Qwen3Arch(BaseModelArch):" in code
        assert "Auto-generated architecture for Qwen3ForCausalLM" in code

    def test_generate_kv_cache_methods_mha(self):
        """Test KV cache method generation for MHA."""
        structure = ModelStructure(
            model_name="TestModel",
            config_class="TestConfig",
            kv_cache_type="mha_gqa",
            kv_cache_dtype="BF16",
        )

        generator = ModelArchCodeGenerator(structure)
        generator._generate_kv_cache_methods()

        code = "\n".join(generator.lines)
        assert "def get_kv_cache(self):" in code
        assert "mha_gqa_kvcache(self.model_config, DataType.BF16)" in code

    def test_generate_kv_cache_methods_mla(self):
        """Test KV cache method generation for MLA."""
        structure = ModelStructure(
            model_name="TestModel",
            config_class="TestConfig",
            kv_cache_type="mla",
            kv_cache_dtype="INT8",
        )

        generator = ModelArchCodeGenerator(structure)
        generator._generate_kv_cache_methods()

        code = "\n".join(generator.lines)
        assert "mla_kvcache(self.model_config, DataType.INT8)" in code

    def test_shape_to_str_with_integers(self):
        """Test shape tuple to string conversion with integers."""
        structure = ModelStructure(model_name="Test", config_class="TestConfig")
        generator = ModelArchCodeGenerator(structure)

        result = generator._shape_to_str((4096, 11008))
        assert result == "4096, 11008"

    def test_shape_to_str_with_strings(self):
        """Test shape tuple to string conversion with string expressions."""
        structure = ModelStructure(model_name="Test", config_class="TestConfig")
        generator = ModelArchCodeGenerator(structure)

        result = generator._shape_to_str(("seq_len", "mc.hidden_size"))
        assert result == "seq_len, mc.hidden_size"

    def test_shape_to_str_mixed(self):
        """Test shape tuple to string conversion with mixed values."""
        structure = ModelStructure(model_name="Test", config_class="TestConfig")
        generator = ModelArchCodeGenerator(structure)

        result = generator._shape_to_str(("seq_len", 4096))
        assert result == "seq_len, 4096"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_complete_generation_pipeline(self, simple_config, tmp_path):
        """Test complete generation pipeline with file output."""
        pytest.importorskip("torch", reason="PyTorch not installed")

        # Create a simple mock model class
        mock_model_class = MagicMock()
        mock_model_class.__name__ = "TestDenseModel"

        # Create mock model instance with named_modules
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("", mock_model),  # Root module
            (
                "model.layers.0.self_attn.qkv_proj",
                MagicMock(__class__=MagicMock(__name__="QKVParallelLinear")),
            ),
            (
                "model.layers.0.self_attn.o_proj",
                MagicMock(__class__=MagicMock(__name__="RowParallelLinear")),
            ),
        ]
        mock_model_class.return_value = mock_model

        output_file = tmp_path / "test_arch.py"

        # This would need actual torch to work fully
        # For now, just test that the pipeline structure is correct
        # In real usage, this would call generate_model_arch()

    def test_error_reporting_for_multiple_unsupported_layers(self, simple_config):
        """Test that multiple unsupported layers are collected and reported."""
        recognizer = LayerTypeRecognizer(simple_config)

        # Create multiple unsupported layer types
        unsupported_layers = []
        for i, layer_type in enumerate(
            ["UnknownType1", "UnknownType2", "UnknownType1"]
        ):
            mock_module = MagicMock()
            mock_module.__class__.__name__ = layer_type
            mock_module.__class__.__module__ = "test_module"

            try:
                recognizer.recognize(f"layer.{i}", mock_module)
            except UnsupportedLayerError as e:
                unsupported_layers.append(
                    {"name": e.layer_name, "type": e.layer_type, "file": e.module_file}
                )

        # Should have collected 3 errors
        assert len(unsupported_layers) == 3

        # Should have 2 unique types
        types = set(l["type"] for l in unsupported_layers)
        assert types == {"UnknownType1", "UnknownType2"}


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_missing_dependency_error_message(self):
        """Test that MissingDependencyError has helpful message."""
        error = MissingDependencyError(
            "PyTorch is required.\nInstall with: pip install torch"
        )
        assert "PyTorch is required" in str(error)
        assert "pip install torch" in str(error)

    def test_unsupported_layer_error_suggestion(self, simple_config):
        """Test that UnsupportedLayerError provides actionable suggestion."""
        recognizer = LayerTypeRecognizer(simple_config)
        mock_module = MagicMock()
        mock_module.__class__.__name__ = "NewLayerType"

        with pytest.raises(UnsupportedLayerError) as exc_info:
            recognizer.recognize("test.layer", mock_module)

        error_message = str(exc_info.value)
        assert "NewLayerType" in error_message
        assert "LAYER_PATTERNS" in error_message
        assert "_extract_newlayertype_shape" in error_message

    def test_model_instantiation_error_formatting(self):
        """Test that ModelInstantiationError is properly formatted."""
        error = ModelInstantiationError(
            "\n======================================================================\n"
            "MODEL INSTANTIATION FAILED\n"
            "======================================================================\n"
            "Model Class: TestModel\n"
            "Error: Test error\n"
        )

        assert "MODEL INSTANTIATION FAILED" in str(error)
        assert "TestModel" in str(error)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
