"""Declared capacity limits of the semif backend and how they are worded on refusal.

The Jev Decision Index `http` engine records a 4xx as a declared limit (status `unsupported`) only when the body
contains one of its capacity markers; anything else becomes a retried `error`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omj.backends.semif_prompt import LABELS, render_question
from omj.config import BackendSection
from omj.gateway.schema import SchemaError, check_token_budget


def test_token_budget_refusal_names_the_context_limit() -> None:
    with pytest.raises(SchemaError) as exc_info:
        check_token_budget(n_tokens=5000, max_tokens=4096)
    assert exc_info.value.status == 422
    assert "maximum context length" in exc_info.value.message


def test_too_many_options_refusal_names_the_option_limit() -> None:
    question = {
        "type": "choice",
        "instructions": "Pick one.",
        "criteria": {f"k{i}": f"option {i}" for i in range(4)},
    }
    with pytest.raises(ValueError, match="options per choice"):
        render_question("state", question, LABELS[:3])


def test_backend_max_state_tokens_defaults_to_none_and_must_be_positive() -> None:
    assert BackendSection().max_state_tokens is None
    assert BackendSection(max_state_tokens=32768).max_state_tokens == 32768
    with pytest.raises(ValueError):
        BackendSection(max_state_tokens=0)


class _Tokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [100 + LABELS.index(text[1:])]


class _Model:
    def __init__(self, config: object | None) -> None:
        self.config = config

    def parameters(self):
        return iter(())


@pytest.mark.parametrize(
    ("config", "requested", "expected"),
    [
        (None, None, 4096),
        (SimpleNamespace(max_position_embeddings=262144), None, 4096),
        (SimpleNamespace(max_position_embeddings=262144), 32768, 32768),
        (SimpleNamespace(max_position_embeddings=2048), 32768, 2048),
        (SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=16384)), 32768, 16384),
    ],
)
def test_load_applies_max_state_tokens_within_the_model_limit(
    config: object | None, requested: int | None, expected: int
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from omj.backends.semif import SemifBackend

    backend = SemifBackend(model=_Model(config), tokenizer=_Tokenizer())
    backend.load(BackendSection(name="semif", model="tiny/model", max_state_tokens=requested))
    assert backend.capabilities.max_state_tokens == expected
