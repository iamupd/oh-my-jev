"""KevBackend HTTP proxy: requests, responses, error handling, health checks.
# REQ-027
"""

from __future__ import annotations

import os

import httpx
import pytest

from omj.backends.kev import KevBackend
from omj.config import BackendSection
from omj.errors import ErrorCode, OmjError


@pytest.fixture
def allow_mock_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)


class MockTransport(httpx.BaseTransport):
    """Mock HTTP transport to intercept and assert on requests."""

    def __init__(self, response_data: dict | None = None, status_code: int = 200, side_effect: Exception | None = None):
        self.response_data = response_data or {}
        self.status_code = status_code
        self.side_effect = side_effect
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        if self.side_effect:
            raise self.side_effect
        return httpx.Response(self.status_code, json=self.response_data)


def test_kev_backend_attributes() -> None:
    # REQ-027
    backend = KevBackend()
    assert backend.name == "kev"
    assert backend.capabilities.max_options == 255
    assert backend.capabilities.max_state_tokens == 32768
    assert backend.capabilities.supports_batch is True
    assert backend.capabilities.calibrated is True
    assert backend.capabilities.device == "remote"


def test_kev_backend_model_id_from_config(allow_mock_network: None) -> None:
    # REQ-027: model_id = upstream model when seen else cfg.model or "kev-latest"
    backend = KevBackend()
    cfg = BackendSection(model="custom-model")
    backend.load(cfg)
    assert backend.model_id == "custom-model"


def test_kev_backend_model_id_from_upstream(allow_mock_network: None) -> None:
    # REQ-027: model_id set from upstream response
    backend = KevBackend()
    cfg = BackendSection(model="custom-model")
    transport = MockTransport(
        response_data={
            "model": "upstream-kev",
            "answers": {"q1": {"type": "noul", "noul": 0.6}},
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "latency_ms": 100,
        }
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}
    backend.decide("test state", questions)
    assert backend.model_id == "upstream-kev"


def test_kev_backend_load_strips_trailing_slash(allow_mock_network: None) -> None:
    # REQ-027: base URL strip trailing slash
    backend = KevBackend()
    cfg = BackendSection(kev_base_url="http://127.0.0.1:8009/")
    transport = MockTransport(
        response_data={
            "answers": {"q1": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency_ms": 0,
        }
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}
    backend.decide("test state", questions)
    assert backend._base_url == "http://127.0.0.1:8009"


def test_kev_backend_decide_sends_correct_request(allow_mock_network: None) -> None:
    # REQ-027: POST to {base}/v1/systemone with correct body
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(
        response_data={
            "model": "kev-4b",
            "answers": {"q1": {"type": "noul", "noul": 0.7}},
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "latency_ms": 50,
        }
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}
    backend.decide("test state", questions)

    assert transport.last_request is not None
    assert transport.last_request.method == "POST"
    assert "/v1/systemone" in str(transport.last_request.url)
    request_body = transport.last_request.content
    assert b"test state" in request_body
    assert b"kev-latest" in request_body or b"kev-4b" in request_body


def test_kev_backend_convert_noul_answer(allow_mock_network: None) -> None:
    # REQ-027: noul → ["yes","no"], [p, 1-p]
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(
        response_data={
            "model": "kev-4b",
            "answers": {"q1": {"type": "noul", "noul": 0.75}},
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "latency_ms": 50,
        }
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}
    result = backend.decide("test state", questions)

    answer = result["q1"]
    assert answer.qid == "q1"
    assert answer.kind == "noul"
    assert answer.keys == ["yes", "no"]
    assert answer.probs == [0.75, 0.25]
    assert answer.calibrated is True


def test_kev_backend_convert_choice_answer(allow_mock_network: None) -> None:
    # REQ-027: choice → keys in request criteria order, probs from probabilities
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(
        response_data={
            "model": "kev-4b",
            "answers": {
                "q1": {
                    "type": "choice",
                    "choice": "option_b",
                    "probabilities": {"option_a": 0.2, "option_b": 0.6, "option_c": 0.2},
                    "confidence": 0.85,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "latency_ms": 50,
        }
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {
        "q1": {
            "type": "choice",
            "instructions": "test",
            "criteria": {"option_a": "desc_a", "option_b": "desc_b", "option_c": "desc_c"},
        }
    }
    result = backend.decide("test state", questions)

    answer = result["q1"]
    assert answer.qid == "q1"
    assert answer.kind == "choice"
    assert answer.keys == ["option_a", "option_b", "option_c"]
    assert answer.probs == [0.2, 0.6, 0.2]
    assert answer.calibrated is True
    assert answer.meta["confidence"] == 0.85


def test_kev_backend_convert_choice_answer_with_missing_probs(allow_mock_network: None) -> None:
    # REQ-027: choice probabilities missing keys → 0.0
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(
        response_data={
            "model": "kev-4b",
            "answers": {
                "q1": {
                    "type": "choice",
                    "choice": "option_a",
                    "probabilities": {"option_a": 0.8},
                    "confidence": 0.90,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "latency_ms": 50,
        }
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {
        "q1": {
            "type": "choice",
            "instructions": "test",
            "criteria": {"option_a": "a", "option_b": "b"},
        }
    }
    result = backend.decide("test state", questions)

    answer = result["q1"]
    assert answer.probs == [0.8, 0.0]


def test_kev_backend_convert_score_answer(allow_mock_network: None) -> None:
    # REQ-027: score → "0".."n-1"
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(
        response_data={
            "model": "kev-4b",
            "answers": {
                "q1": {
                    "type": "score",
                    "score": 2.1,
                    "legend": {"0": "bad", "1": "ok", "2": "good", "3": "excellent"},
                    "probabilities": {"0": 0.1, "1": 0.2, "2": 0.5, "3": 0.2},
                    "confidence": 0.88,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "latency_ms": 50,
        }
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {
        "q1": {
            "type": "score",
            "instructions": "test",
            "criteria": ["bad", "ok", "good", "excellent"],
        }
    }
    result = backend.decide("test state", questions)

    answer = result["q1"]
    assert answer.qid == "q1"
    assert answer.kind == "score"
    assert answer.keys == ["0", "1", "2", "3"]
    assert answer.probs == [0.1, 0.2, 0.5, 0.2]
    assert answer.calibrated is True
    assert answer.meta["confidence"] == 0.88


def test_kev_backend_connection_error_raises_e_net(allow_mock_network: None) -> None:
    # REQ-027: connection error → E_NET
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(side_effect=httpx.ConnectError("Connection failed"))
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}

    with pytest.raises(OmjError) as exc_info:
        backend.decide("test state", questions)

    assert exc_info.value.code == ErrorCode.E_NET


def test_kev_backend_http_4xx_raises_e_backend(allow_mock_network: None) -> None:
    # REQ-027: HTTP 4xx → E_BACKEND
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(
        status_code=422, response_data={"detail": "Invalid request schema"}
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}

    with pytest.raises(OmjError) as exc_info:
        backend.decide("test state", questions)

    assert exc_info.value.code == ErrorCode.E_BACKEND


def test_kev_backend_http_5xx_raises_e_net(allow_mock_network: None) -> None:
    # REQ-027: HTTP 5xx → E_NET
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(
        status_code=500, response_data={"detail": "Internal server error"}
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}

    with pytest.raises(OmjError) as exc_info:
        backend.decide("test state", questions)

    assert exc_info.value.code == ErrorCode.E_NET


def test_kev_backend_omj_no_network_raises_e_net(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-027: env OMJ_NO_NETWORK == "1" → OmjError(E_NET) with zero transport calls
    monkeypatch.setenv("OMJ_NO_NETWORK", "1")
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(response_data={})
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}

    with pytest.raises(OmjError) as exc_info:
        backend.decide("test state", questions)

    assert exc_info.value.code == ErrorCode.E_NET
    assert transport.last_request is None


def test_kev_backend_health_ok(allow_mock_network: None) -> None:
    # REQ-027: GET {base}/health → ok
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(response_data={"status": "ok"})
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    health = backend.health()
    assert health.ok is True


def test_kev_backend_health_connection_error(allow_mock_network: None) -> None:
    # REQ-027: connection error in health → Health(ok=False)
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(side_effect=httpx.ConnectError("Connection failed"))
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    health = backend.health()
    assert health.ok is False
    assert "Connection" in health.detail or "error" in health.detail


def test_kev_backend_health_omj_no_network() -> None:
    # REQ-027: under OMJ_NO_NETWORK return ok=False with detail "network disabled"
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(response_data={})
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    health = backend.health()
    assert health.ok is False
    assert "disabled" in health.detail


def test_kev_backend_count_tokens_whitespace_approximation() -> None:
    # REQ-027: count_tokens whitespace approximation
    backend = KevBackend()
    cfg = BackendSection()
    backend.load(cfg)

    state = "one two three four five"
    questions = {
        "q1": {"type": "noul", "instructions": "question one two", "criteria": {}},
        "q2": {"type": "choice", "instructions": "other", "criteria": {}},
    }

    token_count = backend.count_tokens(state, questions)
    assert token_count > 0
    assert isinstance(token_count, int)
    assert token_count >= 5


def test_kev_backend_meta_includes_upstream_info(allow_mock_network: None) -> None:
    # REQ-027: meta = {"upstream_model", "latency_ms", "confidence"}
    backend = KevBackend()
    cfg = BackendSection()
    transport = MockTransport(
        response_data={
            "model": "kev-upstream-v1",
            "answers": {"q1": {"type": "noul", "noul": 0.7}},
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "latency_ms": 123,
        }
    )
    backend.load(cfg, http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}
    result = backend.decide("test state", questions)

    answer = result["q1"]
    assert answer.meta["upstream_model"] == "kev-upstream-v1"
    assert answer.meta["latency_ms"] == 123


def test_kev_backend_missing_answer_for_qid_raises_e_backend(allow_mock_network: None) -> None:
    # REQ-027: a silently fabricated answer is worse than an error.
    backend = KevBackend()
    transport = MockTransport(
        response_data={
            "model": "kev-4b",
            "answers": {"other": {"type": "noul", "noul": 0.5}},
            "latency_ms": 10,
        }
    )
    backend.load(BackendSection(), http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}
    with pytest.raises(OmjError) as exc_info:
        backend.decide("test state", questions)

    assert exc_info.value.code == ErrorCode.E_BACKEND
    assert "upstream answer missing for 'q1'" in exc_info.value.message


def test_kev_backend_answers_field_absent_raises_e_backend(allow_mock_network: None) -> None:
    # REQ-027
    backend = KevBackend()
    transport = MockTransport(response_data={"model": "kev-4b", "latency_ms": 10})
    backend.load(BackendSection(), http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}
    with pytest.raises(OmjError) as exc_info:
        backend.decide("test state", questions)

    assert exc_info.value.code == ErrorCode.E_BACKEND


def test_kev_backend_noul_without_value_raises_e_backend(allow_mock_network: None) -> None:
    # REQ-027: a missing noul must not become a confident 0.0.
    backend = KevBackend()
    transport = MockTransport(
        response_data={
            "model": "kev-4b",
            "answers": {"q1": {"type": "noul"}},
            "latency_ms": 10,
        }
    )
    backend.load(BackendSection(), http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "noul", "instructions": "test", "criteria": {}}}
    with pytest.raises(OmjError) as exc_info:
        backend.decide("test state", questions)

    assert exc_info.value.code == ErrorCode.E_BACKEND
    assert "q1" in exc_info.value.message


def test_kev_backend_choice_without_probabilities_raises_e_backend(
    allow_mock_network: None,
) -> None:
    # REQ-027: a missing distribution must not become a uniform-zero vector.
    backend = KevBackend()
    transport = MockTransport(
        response_data={
            "model": "kev-4b",
            "answers": {"q1": {"type": "choice", "choice": "option_a"}},
            "latency_ms": 10,
        }
    )
    backend.load(BackendSection(), http_client=httpx.Client(transport=transport))

    questions = {
        "q1": {
            "type": "choice",
            "instructions": "test",
            "criteria": {"option_a": "a", "option_b": "b"},
        }
    }
    with pytest.raises(OmjError) as exc_info:
        backend.decide("test state", questions)

    assert exc_info.value.code == ErrorCode.E_BACKEND


def test_kev_backend_score_without_probabilities_raises_e_backend(
    allow_mock_network: None,
) -> None:
    # REQ-027
    backend = KevBackend()
    transport = MockTransport(
        response_data={
            "model": "kev-4b",
            "answers": {"q1": {"type": "score", "score": 1.0}},
            "latency_ms": 10,
        }
    )
    backend.load(BackendSection(), http_client=httpx.Client(transport=transport))

    questions = {"q1": {"type": "score", "instructions": "test", "criteria": ["low", "high"]}}
    with pytest.raises(OmjError) as exc_info:
        backend.decide("test state", questions)

    assert exc_info.value.code == ErrorCode.E_BACKEND


def test_kev_backend_noul_without_criteria_key(allow_mock_network: None) -> None:
    # REQ-027: a noul question may omit "criteria" entirely (same guard as mock)
    backend = KevBackend()
    transport = MockTransport(
        response_data={"model": "kev-4b", "answers": {"q1": {"type": "noul", "noul": 0.6}}}
    )
    backend.load(BackendSection(), http_client=httpx.Client(transport=transport))
    result = backend.decide("state", {"q1": {"type": "noul", "instructions": "test"}})
    assert result["q1"].keys == ["yes", "no"]
    assert result["q1"].probs == [0.6, 0.4]
