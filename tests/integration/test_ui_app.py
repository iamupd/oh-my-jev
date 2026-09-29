"""omj ui: root app with mounted per-target gateways (mock and typesafe offline).

# REQ-U01
# REQ-U02
# REQ-U04
# REQ-U06
# REQ-U08
# REQ-U09
# REQ-U11
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omj.cli.serve_cmd import build_app
from omj.config import BackendSection, Config
from omj.errors import ErrorCode, OmjError
from omj.ui.app import create_ui_app, load_presets
from omj.ui.policy import evaluate_policy
from omj.ui.targets import TargetSpec, build_target_apps

# A recognizable non-credential marker; the test asserts it never leaves the server.
KEY_SENTINEL = "-".join(["omj", "ui", "leak", "probe", "7731"])


def _client(specs: list[TargetSpec], access_code: str | None = None, **kw) -> TestClient:
    config = Config(backend=BackendSection(name="mock"))
    return TestClient(create_ui_app(build_target_apps(config, specs, **kw), access_code=access_code))


def test_index_and_targets_listing() -> None:
    client = _client([TargetSpec("a", "mock"), TargetSpec("b", "mock")])
    page = client.get("/")
    assert page.status_code == 200 and "omj ui" in page.text
    targets = client.get("/ui/api/targets").json()["targets"]
    assert [t["name"] for t in targets] == ["a", "b"]
    assert targets[0]["endpoint"] == "/t/a/v1/systemone"
    assert targets[0]["model"] == "mock-keyword-v1"


def test_every_preset_locale_runs_on_each_target_and_policy_evaluates() -> None:
    client = _client([TargetSpec("a", "mock"), TargetSpec("b", "mock")])
    payload = client.get("/ui/api/presets").json()
    assert payload["default_locale"] == "en"
    presets = payload["presets"]
    assert len(presets) == len(load_presets())
    for preset in presets:
        for code, loc in preset["locales"].items():
            body = {"model": "jev-latest", "state": loc["state"], "questions": loc["questions"]}
            for name in ("a", "b"):
                r = client.post(f"/t/{name}/v1/systemone", json=body)
                assert r.status_code == 200, (preset["id"], code, r.text)
                assert r.headers["x-omj-model"] == "mock-keyword-v1"
                assert float(r.headers["x-omj-latency-ms"]) >= 0
                answers = r.json()["answers"]
                assert set(answers) == set(loc["questions"])
                verdict = evaluate_policy(preset["policy"], answers)
                assert verdict["label"] in (preset["policy"]["pass_label"], preset["policy"]["fail_label"])


def test_invalid_request_returns_gateway_error_for_that_target_only() -> None:
    client = _client([TargetSpec("a", "mock"), TargetSpec("b", "mock")])
    bad = client.post("/t/a/v1/systemone", json={"model": "jev-latest", "state": "x", "questions": {}})
    assert bad.status_code == 422 and bad.json()["error"]["type"] == "validation_error"
    ok = client.post("/t/b/v1/systemone", json={"model": "jev-latest", "state": "x",
                                                "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    assert ok.status_code == 200


def test_backend_failure_is_reported_per_target(monkeypatch: pytest.MonkeyPatch) -> None:
    from omj.backends.mock import MockBackend

    def boom(self, state, questions):
        raise OmjError(ErrorCode.E_BACKEND, "upstream rejected the request")

    client = _client([TargetSpec("a", "mock")])
    monkeypatch.setattr(MockBackend, "decide", boom)
    r = client.post("/t/a/v1/systemone", json={"model": "jev-latest", "state": "x",
                                               "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    assert r.status_code == 422
    assert "upstream rejected" in r.json()["error"]["message"]


def test_api_key_never_reaches_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_KEY", KEY_SENTINEL)
    monkeypatch.setenv("OMJ_NO_NETWORK", "1")
    client = _client([TargetSpec("local", "mock"), TargetSpec("jev", "typesafe")])
    for path in ("/", "/ui/api/targets", "/ui/api/presets", "/t/jev/health", "/t/jev/v1/models"):
        assert KEY_SENTINEL not in client.get(path).text, path


def test_target_load_failure_names_the_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEV_KEY", raising=False)
    with pytest.raises(OmjError) as exc:
        build_target_apps(Config(backend=BackendSection(name="mock")), [TargetSpec("jev", "typesafe")])
    assert "target 'jev'" in exc.value.message


def test_ui_command_help_is_registered() -> None:
    from typer.testing import CliRunner

    from omj.cli.main import app

    result = CliRunner().invoke(app, ["ui", "--help"])
    assert result.exit_code == 0 and "--target" in result.stdout


def test_build_ui_uses_injected_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    from omj.cli import ui_cmd

    monkeypatch.setattr(ui_cmd, "load_config", lambda: Config(backend=BackendSection(name="mock")))
    ui = ui_cmd.build_ui(["x=mock", "y=mock"], builder=build_app)
    assert [t["name"] for t in ui.state.targets] == ["x", "y"]


ACCESS = "-".join(["remote", "access", "probe", "42"])


def test_remote_access_requires_code_everywhere() -> None:
    client = _client([TargetSpec("a", "mock")], access_code=ACCESS)
    body = {"model": "jev-latest", "state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    for method, path in (("get", "/"), ("get", "/ui/api/targets"), ("get", "/ui/api/presets")):
        assert getattr(client, method)(path).status_code == 401, path
    assert client.post("/t/a/v1/systemone", json=body).status_code == 401
    assert client.get("/?token=wrong", follow_redirects=False).status_code == 401


def test_remote_access_code_moves_into_cookie_then_works() -> None:
    client = _client([TargetSpec("a", "mock")], access_code=ACCESS)
    first = client.get(f"/?token={ACCESS}", follow_redirects=False)
    assert first.status_code == 303 and first.headers["location"] == "/"
    assert "httponly" in first.headers["set-cookie"].lower()
    assert client.get("/ui/api/targets").status_code == 200  # cookie kept by the client
    body = {"model": "jev-latest", "state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    assert client.post("/t/a/v1/systemone", json=body).status_code == 200


def test_remote_access_header_for_api_clients() -> None:
    client = _client([TargetSpec("a", "mock")], access_code=ACCESS)
    assert client.get("/ui/api/presets", headers={"x-omj-ui-token": ACCESS}).status_code == 200
