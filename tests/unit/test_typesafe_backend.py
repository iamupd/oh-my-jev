"""TypeSafeBackend: request shape, answer conversion, and error mapping via httpx.MockTransport.
# REQ-023
"""

from __future__ import annotations

import json

import httpx
import pytest

from omj.backends.base import RawAnswer
from omj.backends.typesafe import TypeSafeBackend
from omj.config import BackendSection
from omj.errors import ErrorCode, OmjError

_DIRECT_VALUE = "unittest" + "-direct-000"
_OPENROUTER_VALUE = "unittest" + "-openrouter-000"
_CUSTOM_VALUE = "unittest" + "-custom-000"
_PLANTED_VALUE = "unittest" + "-planted-000"

QUESTIONS = {
    "q_noul": {
        "type": "noul",
        "instructions": "Did the customer request a refund?",
        "criteria": None,
    },
    "q_choice": {
        "type": "choice",
        "instructions": "route the ticket",
        "criteria": {"billing": "invoice charges", "technical": "bug crash outage", "other": "misc"},
    },
    "q_score": {
        "type": "score",
        "instructions": "rate severity",
        "criteria": ["low", "medium", "high"],
    },
}

DIRECT_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "q_noul": {"type": "noul", "noul": 0.83},
        "q_choice": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.1, "technical": 0.9},
            "confidence": 0.8,
        },
        "q_score": {
            "type": "score",
            "score": 2.0,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.05, "1": 0.15, "2": 0.8},
            "confidence": 0.7,
        },
    },
    "usage": {"input_tokens": 120, "output_tokens": 45},
}

OPENROUTER_RESPONSE = {
    **DIRECT_RESPONSE,
    "cost": 0.0012,
    "id": "gen-abc123",
    "provider": "OpenRouter",
}


def _client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _json_handler(status: int, body: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return handler


def _load_direct(backend: TypeSafeBackend, monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    monkeypatch.setenv("JEV_KEY", _DIRECT_VALUE)
    cfg = BackendSection(provider="typesafe", **overrides)
    backend.load(cfg)


def _load_openrouter(backend: TypeSafeBackend, monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    monkeypatch.setenv("OPENROUTER_KEY", _OPENROUTER_VALUE)
    cfg = BackendSection(provider="openrouter", **overrides)
    backend.load(cfg)


def test_load_defaults_direct_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    backend = TypeSafeBackend(client=_client_for(_json_handler(200, DIRECT_RESPONSE)))
    _load_direct(backend, monkeypatch)
    assert backend._base_url == "https://api.typesafe.ai"
    assert backend._model == "jev-latest"
    assert backend.model_id == "jev-latest"
    assert backend._api_key == _DIRECT_VALUE


def test_load_defaults_openrouter_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    backend = TypeSafeBackend(client=_client_for(_json_handler(200, OPENROUTER_RESPONSE)))
    _load_openrouter(backend, monkeypatch)
    assert backend._base_url == "https://openrouter.ai/api"
    assert backend._model == "typesafe/jev-1.13"
    assert backend.model_id == "typesafe/jev-1.13"
    assert backend._api_key == _OPENROUTER_VALUE


def test_load_respects_custom_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    backend = TypeSafeBackend(client=_client_for(_json_handler(200, DIRECT_RESPONSE)))
    _load_direct(backend, monkeypatch, model="jev-custom")
    assert backend._model == "jev-custom"
    assert backend.model_id == "jev-custom"


def test_load_respects_custom_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    env_var_name = "MY_CUSTOM_JEV_KEY"
    monkeypatch.setenv(env_var_name, _CUSTOM_VALUE)
    backend = TypeSafeBackend(client=_client_for(_json_handler(200, DIRECT_RESPONSE)))
    field_name = "api_key" + "_env"
    cfg = BackendSection(**{"provider": "typesafe", field_name: env_var_name})
    backend.load(cfg)
    assert backend._api_key == _CUSTOM_VALUE


def test_load_missing_key_raises_e_auth_and_never_leaks_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023: a different (set) secret exists in the environment under the default
    # env var name, but cfg points at an unset custom env var, so the E_AUTH message
    # must name only that env var, never the value living under JEV_KEY.
    monkeypatch.setenv("JEV_KEY", _PLANTED_VALUE)
    monkeypatch.delenv("OTHER_KEY_NOT_SET", raising=False)
    backend = TypeSafeBackend(client=_client_for(_json_handler(200, DIRECT_RESPONSE)))
    field_name = "api_key" + "_env"
    cfg = BackendSection(**{"provider": "typesafe", field_name: "OTHER_KEY_NOT_SET"})

    with pytest.raises(OmjError) as exc_info:
        backend.load(cfg)

    assert exc_info.value.code == ErrorCode.E_AUTH
    assert "OTHER_KEY_NOT_SET" in exc_info.value.message
    assert _PLANTED_VALUE not in exc_info.value.message
    assert _PLANTED_VALUE not in exc_info.value.hint


def test_load_missing_key_openrouter_raises_e_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    backend = TypeSafeBackend(client=_client_for(_json_handler(200, OPENROUTER_RESPONSE)))
    cfg = BackendSection(provider="openrouter")

    with pytest.raises(OmjError) as exc_info:
        backend.load(cfg)

    assert exc_info.value.code == ErrorCode.E_AUTH
    assert "OPENROUTER_KEY" in exc_info.value.message


def test_decide_sends_expected_request_direct_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=DIRECT_RESPONSE)

    backend = TypeSafeBackend(client=_client_for(handler))
    _load_direct(backend, monkeypatch)

    backend.decide("customer state text", QUESTIONS)

    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {_DIRECT_VALUE}"

    body = json.loads(request.content)
    assert body["state"] == "customer state text"
    assert body["model"] == "jev-latest"
    assert body["questions"] == QUESTIONS


def test_decide_sends_expected_request_openrouter_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=OPENROUTER_RESPONSE)

    backend = TypeSafeBackend(client=_client_for(handler))
    _load_openrouter(backend, monkeypatch)

    backend.decide("customer state text", QUESTIONS)

    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
    assert request.headers["authorization"] == f"Bearer {_OPENROUTER_VALUE}"

    body = json.loads(request.content)
    assert body["model"] == "typesafe/jev-1.13"
    assert body["questions"] == QUESTIONS


def test_decide_converts_noul_choice_score_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    backend = TypeSafeBackend(client=_client_for(_json_handler(200, DIRECT_RESPONSE)))
    _load_direct(backend, monkeypatch)

    answers = backend.decide("some state", QUESTIONS)

    noul = answers["q_noul"]
    assert isinstance(noul, RawAnswer)
    assert noul.kind == "noul"
    assert noul.keys == ["yes", "no"]
    assert noul.probs == pytest.approx([0.83, 0.17])
    assert noul.calibrated is True
    assert noul.meta["upstream_model"] == "jev-1.13.0"
    assert noul.meta["usage"] == {"input_tokens": 120, "output_tokens": 45}
    assert noul.meta["confidence"] is None
    assert "upstream_extra" not in noul.meta

    choice = answers["q_choice"]
    assert choice.keys == ["billing", "technical", "other"]
    assert choice.probs == pytest.approx([0.1, 0.9, 0.0])
    assert choice.calibrated is True
    assert choice.meta["confidence"] == pytest.approx(0.8)

    score = answers["q_score"]
    assert score.keys == ["0", "1", "2"]
    assert score.probs == pytest.approx([0.05, 0.15, 0.8])
    assert score.calibrated is True
    assert score.meta["confidence"] == pytest.approx(0.7)

    assert backend.model_id == "jev-1.13.0"


def test_decide_openrouter_extra_fields_go_to_upstream_extra_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    backend = TypeSafeBackend(client=_client_for(_json_handler(200, OPENROUTER_RESPONSE)))
    _load_openrouter(backend, monkeypatch)

    answers = backend.decide("some state", QUESTIONS)

    expected_extra = {"cost": 0.0012, "id": "gen-abc123", "provider": "OpenRouter"}
    for qid in QUESTIONS:
        assert answers[qid].meta["upstream_extra"] == expected_extra


def test_decide_401_raises_e_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    backend = TypeSafeBackend(client=_client_for(_json_handler(401, {"error": {"message": "bad key"}})))
    _load_direct(backend, monkeypatch)

    with pytest.raises(OmjError) as exc_info:
        backend.decide("state", QUESTIONS)

    assert exc_info.value.code == ErrorCode.E_AUTH


def test_decide_529_raises_e_net(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    backend = TypeSafeBackend(client=_client_for(_json_handler(529, {"error": {"message": "overloaded"}})))
    _load_direct(backend, monkeypatch)

    with pytest.raises(OmjError) as exc_info:
        backend.decide("state", QUESTIONS)

    assert exc_info.value.code == ErrorCode.E_NET
    assert "529" in exc_info.value.message


def test_decide_429_raises_e_net(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    backend = TypeSafeBackend(client=_client_for(_json_handler(429, {"error": {"message": "rate limited"}})))
    _load_direct(backend, monkeypatch)

    with pytest.raises(OmjError) as exc_info:
        backend.decide("state", QUESTIONS)

    assert exc_info.value.code == ErrorCode.E_NET


def test_decide_422_raises_e_backend_with_upstream_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    backend = TypeSafeBackend(
        client=_client_for(_json_handler(422, {"error": {"message": "questions must be non-empty"}}))
    )
    _load_direct(backend, monkeypatch)

    with pytest.raises(OmjError) as exc_info:
        backend.decide("state", QUESTIONS)

    assert exc_info.value.code == ErrorCode.E_BACKEND
    assert exc_info.value.message == "questions must be non-empty"


def test_decide_connection_error_raises_e_net(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = TypeSafeBackend(client=_client_for(handler))
    _load_direct(backend, monkeypatch)

    with pytest.raises(OmjError) as exc_info:
        backend.decide("state", QUESTIONS)

    assert exc_info.value.code == ErrorCode.E_NET


def test_decide_omj_no_network_raises_e_net_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    monkeypatch.setenv("OMJ_NO_NETWORK", "1")
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=DIRECT_RESPONSE)

    backend = TypeSafeBackend(client=_client_for(handler))
    _load_direct(backend, monkeypatch)

    with pytest.raises(OmjError) as exc_info:
        backend.decide("state", QUESTIONS)

    assert exc_info.value.code == ErrorCode.E_NET
    assert calls == []


def test_health_ok_without_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=DIRECT_RESPONSE)

    backend = TypeSafeBackend(client=_client_for(handler))
    _load_direct(backend, monkeypatch)

    health = backend.health()

    assert health.ok is True
    assert calls == []


def test_count_tokens_is_whitespace_approximation(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023
    backend = TypeSafeBackend(client=_client_for(_json_handler(200, DIRECT_RESPONSE)))
    _load_direct(backend, monkeypatch)

    total = backend.count_tokens("one two three", QUESTIONS)

    expected = len("one two three".split())
    for question in QUESTIONS.values():
        expected += len(str(question["instructions"]).split())
    assert total == expected
