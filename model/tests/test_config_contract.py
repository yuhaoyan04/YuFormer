import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yuformer_model.config import ModelConfig


@pytest.mark.parametrize("field,value", [
    ("selective_recompute", "mlpp"),
    ("qk_norm", "false"),
    ("attention_output_gate", "false"),
    ("tie_word_embeddings", "false"),
    ("bf16_hygiene", "false"),
    ("rope_base", float("inf")),
    ("rope_base", -10000),
    ("norm_eps", float("nan")),
    ("num_layers", 0),
    ("num_kv_heads", 0),
    ("max_seq_len", 4096.5),
])
def test_model_config_rejects_silent_fallbacks_and_invalid_position_settings(field, value):
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: value})
