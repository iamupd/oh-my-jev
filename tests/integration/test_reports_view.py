"""Reports page: run index, /reports routes, and `omj bench --view` (no browser, no real server)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omj.bench.report_index import load_runs, register_run, run_id_for
from omj.ui import viewer
from omj.ui.app import create_ui_app


def _write_run(folder: Path, finished: str, acc: float, *, details: bool = True) -> Path:
    folder.mkdir(parents=True)
    report = {
        "environment": {"backend": "semif", "model": "Qwen/Qwen3.5-2B@15852e8c", "quant": "bf16", "route": "local", "run_at": finished},
        "suites": {"omj-smoke": {"n": 30, "accuracy": acc, "ece": 0.05}},
        "run": {"finished_at": finished, "wall_seconds": 12.0, "command": "omj bench --suite omj-smoke", "options": {"adapter": "/x/best"}},
    }
    if details:
        report["details"] = {"omj-smoke": {"accuracy_ci95": [0.8, 0.95], "by_tag": {}, "reliability": [], "secret": "ignored"}}
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return folder


def test_load_runs_lists_runs_newest_first_including_registered_outside_runs(omj_home: Path, tmp_path: Path) -> None:
    _write_run(omj_home / "runs" / "old", "2026-09-01T00:00:00+00:00", 0.7)
    _write_run(omj_home / "runs" / "new", "2026-09-02T00:00:00+00:00", 0.8, details=False)
    outside = _write_run(tmp_path / "elsewhere" / "new", "2026-09-03T00:00:00+00:00", 0.9)
    register_run(outside)
    register_run(omj_home / "runs" / "new")  # inside runs: not recorded twice

    runs = load_runs()

    assert [r["id"] for r in runs] == ["new", "new-2", "old"]  # same folder name gets a suffix
    assert runs[0]["path"] == str(outside) and runs[0]["adapter"] == "best"
    assert "secret" not in runs[0]["details"]["omj-smoke"]
    assert run_id_for(outside) == "new"


def test_reports_routes_and_reports_only_server(omj_home: Path) -> None:
    _write_run(omj_home / "runs" / "r1", "2026-09-02T00:00:00+00:00", 0.8)
    client = TestClient(create_ui_app([]))

    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307 and root.headers["location"] == "/reports"
    page = client.get("/reports")
    assert page.status_code == 200 and "omj Reports" in page.text
    body = client.get("/ui/api/reports").json()
    assert body["playground"] is False and body["runs"][0]["id"] == "r1"
    assert client.get("/ui/api/health").json() == {"app": "omj-ui", "playground": False}


def test_view_run_reuses_a_running_omj_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(viewer, "is_omj_ui", lambda port, **k: True)
    result = viewer.view_run("r 1", opener=opened.append, serve=lambda *a: pytest.fail("must not serve"), out=io.StringIO())
    assert result == "reused" and opened == ["http://127.0.0.1:8800/reports?run=r%201"]


def test_view_run_serves_on_a_free_port_when_8800_is_taken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viewer, "is_omj_ui", lambda port, **k: False)
    monkeypatch.setattr(viewer, "port_in_use", lambda port, **k: True)
    monkeypatch.setattr(viewer, "free_port", lambda **k: 51234)
    served, opened = [], []

    def fake_serve(port, on_ready):
        served.append(port)
        on_ready()

    assert viewer.view_run("r1", opener=opened.append, serve=fake_serve, out=io.StringIO()) == "served"
    assert served == [51234] and opened == ["http://127.0.0.1:51234/reports?run=r1"]


def test_bench_view_opens_the_new_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from omj.cli.main import app

    seen: list[str | None] = []
    monkeypatch.setattr("omj.ui.viewer.view_run", lambda run_id, **k: seen.append(run_id) or "reused")
    out_dir = tmp_path / "my-run"
    result = CliRunner().invoke(app, ["bench", "--backend", "mock", "--suite", "omj-smoke", "--out", str(out_dir), "--view"])

    assert result.exit_code == 0, result.output
    assert seen == ["my-run"]


def test_reports_api_carries_run_references_and_the_bundled_jev_values(omj_home: Path) -> None:
    folder = _write_run(omj_home / "runs" / "r1", "2026-09-02T00:00:00+00:00", 0.8)
    data = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    data["reference"] = {"name": "Jev", "suites": {"omj-smoke": {"metrics": {"accuracy": 1.0}, "source": "bundled", "run": "", "measured_at": "2026-09-22"}}}
    (folder / "report.json").write_text(json.dumps(data), encoding="utf-8")

    body = TestClient(create_ui_app([])).get("/ui/api/reports").json()

    assert body["runs"][0]["reference"]["omj-smoke"]["metrics"]["accuracy"] == 1.0
    assert body["bundled_reference"]["name"] == "Jev 1.13" and "jevbench-public" in body["bundled_reference"]["suites"]


def test_ui_serves_only_the_logo_and_favicon_files() -> None:
    client = TestClient(create_ui_app([]))
    for name in ("logo.png", "logo-dark.png", "favicon.png"):
        response = client.get(f"/ui/{name}")
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert client.get("/ui/index.html").status_code == 404
    assert client.get("/ui/api/health").json()["app"] == "omj-ui"
    page = client.get("/reports").text
    assert 'href="/ui/favicon.png"' in page and 'src="/ui/logo.png"' in page
