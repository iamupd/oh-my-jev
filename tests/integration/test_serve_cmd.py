"""`omj serve`: config load, CLI overrides, backend/calibration wiring, uvicorn launch.

# REQ-018, REQ-049
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from omj.cli import serve_cmd
from omj.cli.main import app
from omj.config import config_path

runner = CliRunner()

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


def _combined_output(result) -> str:
    output = result.stdout
    try:
        output += result.stderr
    except ValueError:
        pass  # stdout/stderr were mixed into one stream
    return output


def _fake_run(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake(app, **kwargs):
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(serve_cmd.uvicorn, "run", fake)
    return calls


def test_defaults_use_port_8799_and_host_127001(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-049
    calls = _fake_run(monkeypatch)

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0, _combined_output(result)
    assert len(calls) == 1
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 8799


def test_port_and_host_options_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_run(monkeypatch)

    result = runner.invoke(app, ["serve", "--port", "9100", "--host", "0.0.0.0"])

    assert result.exit_code == 0, _combined_output(result)
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 9100


def test_backend_option_overrides_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_run(monkeypatch)
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text('[backend]\nname = "does-not-exist"\n', encoding="utf-8")

    result = runner.invoke(app, ["serve", "--backend", "mock"])

    assert result.exit_code == 0, _combined_output(result)
    assert calls[0]["app"].state.backend.name == "mock"


def test_unknown_backend_exits_one_with_e_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_run(monkeypatch)

    result = runner.invoke(app, ["serve", "--backend", "no-such-backend"])

    assert result.exit_code == 1
    assert "E_BACKEND" in _combined_output(result)


def test_api_key_option_enforces_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-018
    calls = _fake_run(monkeypatch)

    result = runner.invoke(app, ["serve", "--api-key", "omj-placeholder-token"])

    assert result.exit_code == 0, _combined_output(result)
    client = TestClient(calls[0]["app"])

    missing = client.post("/v1/systemone", json=PAYLOAD)
    assert missing.status_code == 401

    correct = client.post(
        "/v1/systemone",
        json=PAYLOAD,
        headers={"Authorization": "Bearer omj-placeholder-token"},
    )
    assert correct.status_code == 200


def test_api_key_option_warns_it_is_visible_in_process_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_run(monkeypatch)

    result = runner.invoke(app, ["serve", "--api-key", "omj-placeholder-token"])

    assert result.exit_code == 0, _combined_output(result)
    assert "process list" in result.stdout.lower()


def test_calibration_option_marks_response_as_calibrated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_run(monkeypatch)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "version": 1,
                "backend": "mock",
                "model": "mock-keyword-v1",
                "temperatures": {"noul": 1.0, "choice": 1.0, "score": 1.0},
                "fitted_on": {"suite": "omj-smoke", "n": 10},
                "ece_before": 0.1,
                "ece_after": 0.05,
                "created_at": "2026-09-22T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["serve", "--calibration", str(calibration_path)])

    assert result.exit_code == 0, _combined_output(result)
    client = TestClient(calls[0]["app"])
    response = client.post("/v1/systemone", json=PAYLOAD)

    assert response.status_code == 200
    assert response.headers["x-omj-calibrated"] == "true"
