"""Concurrency ceiling: requests beyond serve.max_concurrency are refused, not queued.
# REQ-019
"""

from __future__ import annotations

import threading
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from omj.backends.base import RawAnswer
from omj.backends.mock import MockBackend
from omj.config import Config, ServeSection
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


class BlockingBackend(MockBackend):
    name = "blocking"
    model_id = "blocking-v1"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()

    def decide(self, state, questions) -> dict[str, RawAnswer]:
        self.entered.set()
        assert self.released.wait(timeout=10), "test never released the blocking backend"
        return super().decide(state, questions)


def test_request_beyond_max_concurrency_gets_429(tmp_path: Path) -> None:
    backend = BlockingBackend()
    app = create_app(
        backend,
        Config(serve=ServeSection(max_concurrency=1)),
        decision_logger=DecisionLogger(log_dir=tmp_path / "logs"),
    )
    responses: dict[str, httpx.Response] = {}

    def fire(label: str) -> None:
        responses[label] = TestClient(app).post("/v1/systemone", json=PAYLOAD)

    first = threading.Thread(target=fire, args=("first",), daemon=True)
    first.start()
    assert backend.entered.wait(timeout=10), "first request never reached the backend"

    second = threading.Thread(target=fire, args=("second",), daemon=True)
    second.start()
    second.join(timeout=10)
    assert not second.is_alive(), "the over-limit request was queued instead of refused"

    assert responses["second"].status_code == 429
    assert responses["second"].json()["error"]["type"] == "rate_limit_error"

    backend.released.set()
    first.join(timeout=10)
    assert responses["first"].status_code == 200


def test_permit_is_returned_after_each_request(tmp_path: Path) -> None:
    app = create_app(
        MockBackend(),
        Config(serve=ServeSection(max_concurrency=1)),
        decision_logger=DecisionLogger(log_dir=tmp_path / "logs"),
    )
    client = TestClient(app)

    for _ in range(5):
        assert client.post("/v1/systemone", json=PAYLOAD).status_code == 200
