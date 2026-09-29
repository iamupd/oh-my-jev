"""Gateway failure paths: 529 on backend trouble, 422 on bad or oversized requests.
# REQ-017
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omj.backends.base import Capabilities, Health, RawAnswer
from omj.backends.mock import MockBackend
from omj.config import Config
from omj.errors import ErrorCode, OmjError
from omj.gateway.app import create_app
from omj.gateway.decision_log import DecisionLogger

PAYLOAD = {
    "state": "The customer says the package never arrived and demands a refund today.",
    "model": "jev-latest",
    "questions": {
        "q_choice": {
            "type": "choice",
            "instructions": "Pick the best resolution.",
            "criteria": {"refund": "issue a refund", "deny": "deny the claim"},
        }
    },
}


class UnhealthyBackend(MockBackend):
    name = "unhealthy"
    model_id = "unhealthy-v1"

    def health(self) -> Health:
        return Health(ok=False, detail="model weights are not loaded")


class ExplodingBackend(MockBackend):
    name = "exploding"
    model_id = "exploding-v1"

    def decide(self, state, questions) -> dict[str, RawAnswer]:
        raise RuntimeError("CUDA out of memory")


class TinyBudgetBackend(MockBackend):
    name = "tiny"
    model_id = "tiny-v1"
    capabilities = Capabilities(
        max_options=255,
        max_state_tokens=3,
        supports_batch=False,
        calibrated=False,
        device="cpu",
    )


def build_client(backend, tmp_path: Path) -> TestClient:
    app = create_app(
        backend,
        Config(),
        decision_logger=DecisionLogger(log_dir=tmp_path / "logs"),
    )
    return TestClient(app)


def test_failing_health_returns_529_with_retry_after(tmp_path: Path) -> None:
    client = build_client(UnhealthyBackend(), tmp_path)

    response = client.post("/v1/systemone", json=PAYLOAD)

    assert response.status_code == 529
    assert response.headers["retry-after"] == "2"
    assert response.json()["error"]["type"] == "overloaded_error"


def test_failing_health_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    client = build_client(UnhealthyBackend(), tmp_path)

    with caplog.at_level("ERROR", logger="omj.gateway"):
        client.post("/v1/systemone", json=PAYLOAD)

    assert any("model weights are not loaded" in record.getMessage() for record in caplog.records)


def test_health_route_returns_529_when_backend_is_unhealthy(tmp_path: Path) -> None:
    client = build_client(UnhealthyBackend(), tmp_path)

    response = client.get("/health")

    assert response.status_code == 529
    assert response.json()["status"] == "degraded"


def test_decide_exception_returns_529_with_retry_after(tmp_path: Path) -> None:
    client = build_client(ExplodingBackend(), tmp_path)

    response = client.post("/v1/systemone", json=PAYLOAD)

    assert response.status_code == 529
    assert response.headers["retry-after"] == "2"
    assert response.json()["error"]["type"] == "overloaded_error"


def test_decide_exception_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    client = build_client(ExplodingBackend(), tmp_path)

    with caplog.at_level("ERROR", logger="omj.gateway"):
        client.post("/v1/systemone", json=PAYLOAD)

    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_decide_exception_does_not_leak_the_permit(tmp_path: Path) -> None:
    client = build_client(ExplodingBackend(), tmp_path)

    for _ in range(3):
        assert client.post("/v1/systemone", json=PAYLOAD).status_code == 529


def test_malformed_request_returns_422_validation_error(tmp_path: Path) -> None:
    client = build_client(MockBackend(), tmp_path)
    payload = {
        "state": "hello",
        "model": "jev-latest",
        "questions": {"q1": {"type": "choice", "instructions": "pick", "criteria": {"only": "one"}}},
    }

    response = client.post("/v1/systemone", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "validation_error"
    assert body["error"]["message"]


def test_body_that_is_not_json_returns_422(tmp_path: Path) -> None:
    client = build_client(MockBackend(), tmp_path)

    response = client.post(
        "/v1/systemone",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "validation_error"


class AuthFailingBackend(MockBackend):
    name = "auth-failing"
    model_id = "auth-failing-v1"

    def decide(self, state, questions) -> dict[str, RawAnswer]:
        raise OmjError(ErrorCode.E_AUTH, "upstream authentication failed (status 401)")


class RejectingBackend(MockBackend):
    name = "rejecting"
    model_id = "rejecting-v1"

    def decide(self, state, questions) -> dict[str, RawAnswer]:
        raise OmjError(ErrorCode.E_BACKEND, "criteria must have at least two options")


class OfflineBackend(MockBackend):
    name = "offline"
    model_id = "offline-v1"

    def decide(self, state, questions) -> dict[str, RawAnswer]:
        raise OmjError(ErrorCode.E_NET, "connection error: upstream refused")


class LoopWatchingBackend(MockBackend):
    """Records whether health()/count_tokens() ran on the asyncio event loop thread."""

    name = "loop-watching"
    model_id = "loop-watching-v1"

    def __init__(self) -> None:
        super().__init__()
        self.health_on_loop: bool | None = None
        self.count_tokens_on_loop: bool | None = None

    def health(self) -> Health:
        self.health_on_loop = _running_in_event_loop()
        return super().health()

    def count_tokens(self, state, questions) -> int:
        self.count_tokens_on_loop = _running_in_event_loop()
        return super().count_tokens(state, questions)


def _running_in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_backend_auth_error_returns_401_without_key_text(tmp_path: Path) -> None:
    # REQ-018: an upstream 401 is the caller's credential problem, not overload.
    client = build_client(AuthFailingBackend(), tmp_path)

    response = client.post("/v1/systemone", json=PAYLOAD)

    assert response.status_code == 401
    body = response.json()
    assert body["error"]["type"] == "authentication_error"
    assert body["error"]["message"] == "upstream authentication failed"
    assert "401" not in body["error"]["message"]


def test_backend_validation_error_returns_422_with_upstream_message(tmp_path: Path) -> None:
    # REQ-018
    client = build_client(RejectingBackend(), tmp_path)

    response = client.post("/v1/systemone", json=PAYLOAD)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "validation_error"
    assert body["error"]["message"] == "criteria must have at least two options"


def test_backend_network_error_still_returns_529_with_retry_after(tmp_path: Path) -> None:
    # REQ-017
    client = build_client(OfflineBackend(), tmp_path)

    response = client.post("/v1/systemone", json=PAYLOAD)

    assert response.status_code == 529
    assert response.headers["retry-after"] == "2"
    assert response.json()["error"]["type"] == "overloaded_error"


def test_backend_error_mapping_does_not_leak_the_permit(tmp_path: Path) -> None:
    # REQ-017
    client = build_client(AuthFailingBackend(), tmp_path)

    for _ in range(3):
        assert client.post("/v1/systemone", json=PAYLOAD).status_code == 401


def test_health_and_count_tokens_run_off_the_event_loop(tmp_path: Path) -> None:
    # REQ-017: a blocking backend health()/count_tokens() must not freeze the loop.
    backend = LoopWatchingBackend()
    client = build_client(backend, tmp_path)

    assert client.post("/v1/systemone", json=PAYLOAD).status_code == 200
    assert backend.count_tokens_on_loop is False
    assert backend.health_on_loop is False


def test_health_route_runs_health_off_the_event_loop(tmp_path: Path) -> None:
    # REQ-017
    backend = LoopWatchingBackend()
    client = build_client(backend, tmp_path)

    assert client.get("/health").status_code == 200
    assert backend.health_on_loop is False


def test_oversized_state_returns_422_context_length_exceeded(tmp_path: Path) -> None:
    client = build_client(TinyBudgetBackend(), tmp_path)

    response = client.post("/v1/systemone", json=PAYLOAD)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "context_length_exceeded"
    assert "3" in body["error"]["message"]
