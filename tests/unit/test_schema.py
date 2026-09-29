"""Request schema violations -> 422 validation_error, valid parse, token budget check.
# REQ-012, REQ-031
"""

from __future__ import annotations

import pytest

from omj.gateway.schema import (
    Question,
    SchemaError,
    SystemOneRequest,
    check_token_budget,
    parse_request,
)


def _base_payload() -> dict:
    return {
        "state": "user asked about pricing",
        "model": "test-model",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "Pick the best matching option",
                "criteria": {"a": "Option A", "b": "Option B"},
            }
        },
    }


def _missing_state() -> dict:
    payload = _base_payload()
    del payload["state"]
    return payload


def _empty_questions() -> dict:
    payload = _base_payload()
    payload["questions"] = {}
    return payload


def _unknown_type() -> dict:
    payload = _base_payload()
    payload["questions"]["q1"]["type"] = "bogus"
    return payload


def _missing_instructions() -> dict:
    payload = _base_payload()
    del payload["questions"]["q1"]["instructions"]
    return payload


def _choice_one_option() -> dict:
    payload = _base_payload()
    payload["questions"]["q1"]["criteria"] = {"a": "Option A"}
    return payload


def _choice_256_options() -> dict:
    payload = _base_payload()
    payload["questions"]["q1"]["criteria"] = {f"opt{i}": f"Option {i}" for i in range(256)}
    return payload


def _score_one_level() -> dict:
    payload = _base_payload()
    payload["questions"]["q1"] = {
        "type": "score",
        "instructions": "Rate it",
        "criteria": ["only level"],
    }
    return payload


def _score_11_levels() -> dict:
    payload = _base_payload()
    payload["questions"]["q1"] = {
        "type": "score",
        "instructions": "Rate it",
        "criteria": [f"level {i}" for i in range(11)],
    }
    return payload


VIOLATIONS = [
    ("missing_state", _missing_state, "state"),
    ("empty_questions", _empty_questions, "questions"),
    ("unknown_type", _unknown_type, "questions.q1.type"),
    ("missing_instructions", _missing_instructions, "questions.q1.instructions"),
    ("choice_one_option", _choice_one_option, "questions.q1"),
    ("choice_256_options", _choice_256_options, "questions.q1"),
    ("score_one_level", _score_one_level, "questions.q1"),
    ("score_11_levels", _score_11_levels, "questions.q1"),
]


@pytest.mark.parametrize("name,builder,expected_loc", VIOLATIONS, ids=[v[0] for v in VIOLATIONS])
def test_schema_violation_returns_422_validation_error(name: str, builder, expected_loc: str) -> None:
    # REQ-012
    payload = builder()
    with pytest.raises(SchemaError) as exc_info:
        parse_request(payload)

    err = exc_info.value
    assert err.status == 422
    assert err.error_type == "validation_error"
    assert expected_loc in err.message
    assert err.to_body() == {"error": {"type": "validation_error", "message": err.message}}


def test_valid_request_parses() -> None:
    # REQ-012
    req = parse_request(_base_payload())
    assert isinstance(req, SystemOneRequest)
    assert req.model == "test-model"
    assert set(req.questions.keys()) == {"q1"}
    assert isinstance(req.questions["q1"], Question)


def test_token_budget_exceeded_raises_context_length_exceeded() -> None:
    # REQ-031
    with pytest.raises(SchemaError) as exc_info:
        check_token_budget(n_tokens=5000, max_tokens=4096)

    err = exc_info.value
    assert err.status == 422
    assert err.error_type == "context_length_exceeded"
    assert err.to_body()["error"]["type"] == "context_length_exceeded"


def test_token_budget_within_limit_does_not_raise() -> None:
    # REQ-031
    check_token_budget(n_tokens=100, max_tokens=4096)
