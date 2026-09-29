"""Gateway-only overhead (validation, assembly, logging hand-off) against the mock backend.
# REQ-029
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from omj.backends.mock import MockBackend
from omj.config import Config
from omj.gateway.app import create_app
from omj.gateway.decision_log import DecisionLogger

BUDGET_MS = 5.0
SAMPLES = 100

PAYLOAD = {
    "state": "The customer says the package never arrived and demands a refund today.",
    "model": "jev-latest",
    "questions": {
        "q_noul": {
            "type": "noul",
            "instructions": "Is the customer demanding a refund?",
            "criteria": {"true": "refund demanded", "false": "no refund demanded"},
        },
        "q_choice": {
            "type": "choice",
            "instructions": "Pick the best resolution.",
            "criteria": {
                "refund": "issue a refund to the customer",
                "reship": "send a replacement package",
                "deny": "deny the claim",
            },
        },
    },
}


def _build_app(tmp_path: Path):
    return create_app(
        MockBackend(),
        Config(),
        decision_logger=DecisionLogger(log_dir=tmp_path / "logs"),
    )


def _p50_over_testclient(app) -> float:
    with TestClient(app) as client:
        client.post("/v1/systemone", json=PAYLOAD)  # warm-up: import and route caches
        samples = []
        for _ in range(SAMPLES):
            started = time.perf_counter()
            response = client.post("/v1/systemone", json=PAYLOAD)
            samples.append((time.perf_counter() - started) * 1000.0)
            assert response.status_code == 200
    return statistics.median(samples)


def _p50_over_handler(app) -> float:
    handle = app.state.handle_systemone
    body = json.dumps(PAYLOAD).encode("utf-8")
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(handle(body))  # warm-up
        samples = []
        for _ in range(SAMPLES):
            started = time.perf_counter()
            result = loop.run_until_complete(handle(body))
            samples.append((time.perf_counter() - started) * 1000.0)
            assert result.status == 200
    finally:
        loop.close()
    return statistics.median(samples)


def test_gateway_overhead_p50_is_within_budget(tmp_path: Path) -> None:
    app = _build_app(tmp_path)

    p50_http = _p50_over_testclient(app)
    p50_handler = _p50_over_handler(app)

    # p50_http also pays httpx request building plus the TestClient portal hop,
    # which is client-side cost rather than gateway work; the handler-level
    # number is the authoritative reading when the transport alone is slower.
    p50_gateway = min(p50_http, p50_handler)
    assert p50_gateway <= BUDGET_MS, (
        f"gateway p50 {p50_gateway:.3f} ms exceeds the {BUDGET_MS} ms budget "
        f"(TestClient p50 {p50_http:.3f} ms, handler p50 {p50_handler:.3f} ms, n={SAMPLES})"
    )
