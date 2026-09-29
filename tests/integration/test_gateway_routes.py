"""Gateway routes: three POST aliases, /health, /v1/models, x-omj-* headers.
# REQ-011
# REQ-015
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omj.backends.mock import MockBackend
from omj.config import Config
from omj.gateway.app import create_app
from omj.gateway.decision_log import DecisionLogger

SYSTEMONE_PATHS = ["/v1/systemone", "/api/jev", "/v1/inference"]

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
        "q_score": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["not urgent at all", "somewhat urgent", "extremely urgent"],
        },
    },
}


def build_client(tmp_path: Path) -> TestClient:
    app = create_app(
        MockBackend(),
        Config(),
        decision_logger=DecisionLogger(log_dir=tmp_path / "logs"),
    )
    return TestClient(app)


def test_three_post_paths_share_one_handler(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    bodies = []
    for path in SYSTEMONE_PATHS:
        response = client.post(path, json=PAYLOAD)
        assert response.status_code == 200, (path, response.text)
        bodies.append(response.json())

    assert bodies[0] == bodies[1] == bodies[2]


def test_response_envelope_has_model_answers_usage(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    body = client.post("/v1/systemone", json=PAYLOAD).json()

    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == MockBackend.model_id
    assert set(body["answers"]) == set(PAYLOAD["questions"])
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] == 0


def test_arbitrary_model_value_is_accepted_verbatim(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    payload = dict(PAYLOAD, model="definitely-not-a-real-model-9000")

    response = client.post("/v1/systemone", json=payload)

    assert response.status_code == 200
    assert response.json()["model"] == MockBackend.model_id


def test_headers_expose_backend_model_calibration_and_latency(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post("/v1/systemone", json=PAYLOAD)

    assert response.headers["x-omj-backend"] == MockBackend.name
    assert response.headers["x-omj-model"] == MockBackend.model_id
    assert response.headers["x-omj-calibrated"] == "false"
    latency = response.headers["x-omj-latency-ms"]
    assert float(latency) >= 0.0
    assert latency.count(".") == 1
    assert len(latency.split(".")[1]) == 1


def test_noul_answer_shape(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    answer = client.post("/v1/systemone", json=PAYLOAD).json()["answers"]["q_noul"]

    assert set(answer) == {"type", "noul"}
    assert answer["type"] == "noul"
    assert 0.0 <= answer["noul"] <= 1.0


def test_choice_answer_shape(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    answer = client.post("/v1/systemone", json=PAYLOAD).json()["answers"]["q_choice"]

    assert set(answer) == {"type", "choice", "probabilities", "confidence"}
    assert answer["choice"] in PAYLOAD["questions"]["q_choice"]["criteria"]
    assert set(answer["probabilities"]) == set(PAYLOAD["questions"]["q_choice"]["criteria"])
    assert abs(sum(answer["probabilities"].values()) - 1.0) <= 1e-6
    assert 0.0 <= answer["confidence"] <= 1.0


def test_score_answer_shape(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    answer = client.post("/v1/systemone", json=PAYLOAD).json()["answers"]["q_score"]

    assert set(answer) == {"type", "score", "legend", "probabilities", "confidence"}
    assert answer["legend"] == {
        "0": "not urgent at all",
        "1": "somewhat urgent",
        "2": "extremely urgent",
    }
    assert set(answer["probabilities"]) == {"0", "1", "2"}
    assert abs(sum(answer["probabilities"].values()) - 1.0) <= 1e-6
    assert 0.0 <= answer["score"] <= 2.0


def test_health_route_reports_ok(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "backend", "model", "calibrated", "uptime_s"}
    assert body["status"] == "ok"
    assert body["backend"] == MockBackend.name
    assert body["model"] == MockBackend.model_id
    assert body["calibrated"] is False
    assert body["uptime_s"] >= 0.0


def test_models_route_lists_backend_model_with_alias(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "id": MockBackend.model_id,
                "backend": MockBackend.name,
                "aliases": ["jev-latest"],
            }
        ]
    }


def test_successful_request_is_written_to_the_decision_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    client = build_client(tmp_path)

    client.post("/v1/systemone", json=PAYLOAD)

    lines = [line for path in sorted(log_dir.glob("decisions-*.jsonl")) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
