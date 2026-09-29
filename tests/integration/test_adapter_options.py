"""`omj serve --adapter` / `omj bench --adapter`: option wiring and directory validation.

# REQ-012
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from omj.backends.mock import MockBackend
from omj.backends.registry import create_backend
from omj.cli import bench_cmd, serve_cmd
from omj.cli.bench_cmd import BenchOptions, _default_target_factory
from omj.cli.main import app
from omj.cli.serve_cmd import ServeOverrides, build_server_config
from omj.config import BackendSection, Config
from omj.errors import ErrorCode, OmjError

runner = CliRunner()


def _combined_output(result) -> str:
    output = result.stdout
    try:
        output += result.stderr
    except ValueError:
        pass  # stdout/stderr were mixed into one stream
    return output


def _adapter_dir(tmp_path: Path) -> Path:
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    return path


def _fake_run(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake(app, **kwargs):
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(serve_cmd.uvicorn, "run", fake)
    return calls


# --- build_server_config (pure) ---------------------------------------------


def test_build_server_config_applies_adapter(tmp_path: Path) -> None:
    # REQ-012
    adapter_path = tmp_path / "adapter"

    result = build_server_config(Config(), ServeOverrides(adapter=str(adapter_path)))

    assert result.backend.adapter == str(adapter_path)


def test_build_server_config_leaves_adapter_unset_by_default() -> None:
    # REQ-012
    result = build_server_config(Config(), ServeOverrides())

    assert getattr(result.backend, "adapter", "") == ""


def test_build_server_config_validates_the_adapter_override() -> None:
    # REQ-012: the adapter goes through Config validation like every other override,
    # instead of a model_copy(update=...) that would bypass it.
    with pytest.raises(ValidationError):
        build_server_config(Config(), ServeOverrides(adapter=123))  # type: ignore[arg-type]


def test_default_target_factory_validates_the_adapter_override(tmp_path: Path) -> None:
    # REQ-012: same for bench's BackendSection, which used to be built with model_copy.
    with pytest.raises(ValidationError):
        _default_target_factory(BenchOptions(backend="mock", adapter=123), Config())  # type: ignore[arg-type]


# --- the one shared adapter-directory validator --------------------------------


def test_serve_bench_and_semif_share_one_adapter_validator(tmp_path: Path) -> None:
    # REQ-012
    semif = pytest.importorskip("omj.backends.semif")
    validate_adapter_dir = semif.validate_adapter_dir
    adapter_dir = _adapter_dir(tmp_path)

    assert validate_adapter_dir(adapter_dir) == adapter_dir
    assert serve_cmd._require_adapter_dir(adapter_dir) == str(adapter_dir)
    assert bench_cmd._require_adapter_dir(adapter_dir) == str(adapter_dir)

    empty_dir = tmp_path / "empty-adapter"
    empty_dir.mkdir()
    missing = tmp_path / "does-not-exist"
    callers = (validate_adapter_dir, serve_cmd._require_adapter_dir, bench_cmd._require_adapter_dir)

    for bad in (empty_dir, missing):
        messages: set[str] = set()
        for caller in callers:
            with pytest.raises(OmjError) as exc_info:
                caller(bad)
            assert exc_info.value.code is ErrorCode.E_BACKEND
            messages.add(exc_info.value.message)
        assert len(messages) == 1, f"three different messages for {bad}: {messages}"

    with pytest.raises(OmjError) as exc_info:
        validate_adapter_dir(empty_dir)
    assert "adapter_config.json" in exc_info.value.message


# --- omj serve --adapter -----------------------------------------------------


def test_serve_nonexistent_adapter_dir_exits_one_with_e_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REQ-012
    _fake_run(monkeypatch)
    missing = tmp_path / "does-not-exist"

    result = runner.invoke(app, ["serve", "--backend", "mock", "--adapter", str(missing)])

    assert result.exit_code == 1
    assert "E_BACKEND" in _combined_output(result)


def test_serve_adapter_dir_without_adapter_config_json_exits_one_with_e_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REQ-012
    _fake_run(monkeypatch)
    empty_dir = tmp_path / "empty-adapter"
    empty_dir.mkdir()

    result = runner.invoke(app, ["serve", "--backend", "mock", "--adapter", str(empty_dir)])

    assert result.exit_code == 1
    assert "E_BACKEND" in _combined_output(result)


def test_serve_valid_adapter_dir_starts_normally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REQ-012
    calls = _fake_run(monkeypatch)
    adapter_dir = _adapter_dir(tmp_path)

    result = runner.invoke(app, ["serve", "--backend", "mock", "--adapter", str(adapter_dir)])

    assert result.exit_code == 0, _combined_output(result)
    assert len(calls) == 1


# --- omj bench --adapter ------------------------------------------------------


def test_bench_adapter_with_endpoint_exits_one_with_e_config(tmp_path: Path) -> None:
    # REQ-012
    adapter_dir = _adapter_dir(tmp_path)

    result = runner.invoke(
        app,
        [
            "bench",
            "--endpoint",
            "http://127.0.0.1:8799",
            "--adapter",
            str(adapter_dir),
            "--out",
            str(tmp_path / "run"),
        ],
    )

    assert result.exit_code == 1
    assert "E_CONFIG" in _combined_output(result)


def test_bench_nonexistent_adapter_dir_exits_one_with_e_backend(tmp_path: Path) -> None:
    # REQ-012
    missing = tmp_path / "does-not-exist"

    result = runner.invoke(
        app,
        ["bench", "--backend", "mock", "--adapter", str(missing), "--out", str(tmp_path / "run")],
    )

    assert result.exit_code == 1
    assert "E_BACKEND" in _combined_output(result)


def test_bench_mock_backend_ignores_adapter_but_reports_backend_model_id(tmp_path: Path) -> None:
    # REQ-012
    adapter_dir = _adapter_dir(tmp_path)
    out_dir = tmp_path / "run"

    result = runner.invoke(
        app,
        [
            "bench",
            "--backend",
            "mock",
            "--adapter",
            str(adapter_dir),
            "--suite",
            "omj-smoke",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))

    mock_backend = create_backend("mock")
    assert report["environment"]["backend"] == "mock"
    assert report["environment"]["model"] == mock_backend.model_id


def test_default_target_factory_sets_adapter_on_backend_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REQ-012: proves --adapter reaches the BackendSection handed to
    # `backend.load(...)`, even though mock's own load() ignores it.
    adapter_dir = _adapter_dir(tmp_path)
    captured: dict[str, object] = {}
    original_load = MockBackend.load

    def spying_load(self, cfg):
        captured["adapter"] = getattr(cfg, "adapter", None)
        return original_load(self, cfg)

    monkeypatch.setattr(MockBackend, "load", spying_load)

    _default_target_factory(BenchOptions(backend="mock", adapter=str(adapter_dir)), Config())

    assert captured["adapter"] == str(adapter_dir)


def test_mock_backend_load_accepts_adapter_attribute_via_model_construct() -> None:
    # REQ-012: exercises the shape BackendSection will have once T02 declares
    # `adapter` as a real field; model_construct bypasses validation so this
    # works whether or not the field exists yet.
    cfg = BackendSection.model_construct(name="mock", adapter="some/adapter/dir")
    backend = MockBackend()

    backend.load(cfg)  # must not raise

    assert backend.model_id == "mock-keyword-v1"
