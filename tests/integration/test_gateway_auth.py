"""Bearer token enforcement when serve.api_key is configured.
# REQ-018
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omj.backends.mock import MockBackend
from omj.config import Config, ServeSection
from omj.gateway.app import create_app
from omj.gateway.decision_log import DecisionLogger

# Synthetic placeholder token used only by these in-process tests.
TOKEN = "omj-placeholder-token-abcdef"

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


def build_client(tmp_path: Path, token: str) -> TestClient:
    app = create_app(
        MockBackend(),
        Config(serve=ServeSection(api_key=token)),
        decision_logger=DecisionLogger(log_dir=tmp_path / "logs"),
    )
    return TestClient(app)


def test_missing_authorization_header_is_rejected(tmp_path: Path) -> None:
    client = build_client(tmp_path, TOKEN)

    response = client.post("/v1/systemone", json=PAYLOAD)

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_wrong_bearer_token_is_rejected(tmp_path: Path) -> None:
    client = build_client(tmp_path, TOKEN)

    response = client.post(
        "/v1/systemone",
        json=PAYLOAD,
        headers={"Authorization": "Bearer not-the-right-token"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_non_bearer_scheme_is_rejected(tmp_path: Path) -> None:
    client = build_client(tmp_path, TOKEN)

    response = client.post(
        "/v1/systemone",
        json=PAYLOAD,
        headers={"Authorization": f"Basic {TOKEN}"},
    )

    assert response.status_code == 401


def test_rejection_never_echoes_the_configured_key(tmp_path: Path) -> None:
    client = build_client(tmp_path, TOKEN)

    response = client.post("/v1/systemone", json=PAYLOAD)

    assert TOKEN not in response.text


def test_correct_bearer_token_is_accepted(tmp_path: Path) -> None:
    client = build_client(tmp_path, TOKEN)

    response = client.post(
        "/v1/systemone",
        json=PAYLOAD,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json()["model"] == MockBackend.model_id


def test_auth_runs_before_schema_validation(tmp_path: Path) -> None:
    client = build_client(tmp_path, TOKEN)

    response = client.post("/v1/systemone", json={"garbage": True})

    assert response.status_code == 401


def test_empty_api_key_disables_authentication(tmp_path: Path) -> None:
    client = build_client(tmp_path, "")

    assert client.post("/v1/systemone", json=PAYLOAD).status_code == 200
    assert (
        client.post(
            "/v1/systemone",
            json=PAYLOAD,
            headers={"Authorization": "Bearer whatever"},
        ).status_code
        == 200
    )
