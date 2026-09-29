"""Backend protocol shape, RawAnswer invariants, and registry lookup errors.
# REQ-022
"""

from __future__ import annotations

import importlib

import pytest

from omj.backends.base import Backend, Capabilities, Health, RawAnswer, render_state
from omj.backends.mock import MockBackend
from omj.backends.registry import create_backend
from omj.errors import ErrorCode, OmjError


def test_raw_answer_requires_exactly_one_of_logits_or_probs() -> None:
    # REQ-022: a plain `assert` is stripped under python -O, so this invariant
    # must raise a real exception, not AssertionError.
    with pytest.raises(ValueError):
        RawAnswer(qid="q1", kind="choice", keys=["a", "b"])
    with pytest.raises(ValueError):
        RawAnswer(qid="q1", kind="choice", keys=["a", "b"], logits=[1.0, 2.0], probs=[0.5, 0.5])


def test_raw_answer_values_length_must_match_keys() -> None:
    # REQ-022
    with pytest.raises(ValueError):
        RawAnswer(qid="q1", kind="choice", keys=["a", "b", "c"], logits=[1.0, 2.0])
    with pytest.raises(ValueError):
        RawAnswer(qid="q1", kind="score", keys=["0", "1"], probs=[0.5])


def test_raw_answer_noul_keys_must_be_yes_no() -> None:
    # REQ-022
    with pytest.raises(ValueError):
        RawAnswer(qid="q1", kind="noul", keys=["true", "false"], logits=[1.0, 0.0])

    ok = RawAnswer(qid="q1", kind="noul", keys=["yes", "no"], logits=[1.0, 0.0])
    assert ok.keys == ["yes", "no"]
    assert ok.calibrated is False
    assert ok.meta == {}


def test_render_state_dict_keeps_korean_without_unicode_escapes() -> None:
    # REQ-022: render_state must use ensure_ascii=False so Korean text stays
    # readable in decision-log lines and token counts instead of \\uXXXX.
    rendered = render_state({"who": "홍길동", "note": "피해자는 사망하였는가"})
    assert "홍길동" in rendered
    assert "피해자는 사망하였는가" in rendered
    assert "\\u" not in rendered


def test_render_state_list_keeps_korean_without_unicode_escapes() -> None:
    # REQ-022
    rendered = render_state(["고객이 환불을 요청했습니다", 42])
    assert "고객이 환불을 요청했습니다" in rendered
    assert "\\u" not in rendered


def test_raw_answer_accepts_valid_choice_with_probs() -> None:
    # REQ-022
    answer = RawAnswer(qid="q1", kind="choice", keys=["a", "b"], probs=[0.3, 0.7], calibrated=True)
    assert answer.probs == [0.3, 0.7]
    assert answer.logits is None
    assert answer.calibrated is True


def test_mock_backend_satisfies_backend_protocol() -> None:
    # REQ-022
    backend = MockBackend()
    assert isinstance(backend, Backend)
    assert isinstance(backend.capabilities, Capabilities)
    assert isinstance(backend.health(), Health)


def test_create_backend_mock_works() -> None:
    # REQ-022
    backend = create_backend("mock")
    assert isinstance(backend, Backend)
    assert backend.name == "mock"
    assert backend.model_id == "mock-keyword-v1"


def test_create_backend_unknown_name_raises_e_backend() -> None:
    # REQ-022
    with pytest.raises(OmjError) as exc_info:
        create_backend("nope")
    assert exc_info.value.code == ErrorCode.E_BACKEND


def test_create_backend_semif_without_torch_raises_e_backend_naming_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REQ-022: semif isn't implemented here and torch isn't in the dev extra,
    # so this already fails naturally. Guard with a monkeypatch in case a
    # future environment happens to have both the module and torch installed.
    try:
        importlib.import_module("omj.backends.semif")
        already_importable = True
    except ImportError:
        already_importable = False

    if already_importable:
        def _fail(name: str, *args: object, **kwargs: object) -> None:
            raise ImportError("simulated: torch not installed")

        monkeypatch.setattr(importlib, "import_module", _fail)

    with pytest.raises(OmjError) as exc_info:
        create_backend("semif")

    assert exc_info.value.code == ErrorCode.E_BACKEND
    assert "semif" in exc_info.value.message
    assert "uv sync --extra semif" in exc_info.value.message
