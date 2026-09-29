"""Jev reference next to bench results: local run, fresh measurement with a key, or bundled values."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from omj.bench.reference import bundled, verdict
from omj.cli.bench_cmd import BenchOptions, run_bench


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEV_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)


def _bench(tmp_path: Path, **kw) -> dict:
    opts = BenchOptions(backend="mock", suites=["omj-smoke"], out=tmp_path / "run", **kw)
    return run_bench(opts, out=io.StringIO())


def test_verdict_marks_ci_overlap_and_direction() -> None:
    assert verdict(0.857, 0.853, [0.806, 0.896], "up") == "≈"
    assert verdict(0.90, 0.80, [0.85, 0.95], "up") == "▲"
    assert verdict(0.075, 0.040, None, "down") == "▼"
    assert verdict(None, 0.5, None, "up") == ""


def test_without_a_key_the_bundled_jev_values_are_used(tmp_path: Path) -> None:
    report = _bench(tmp_path)
    info = report["reference"]["suites"]["omj-smoke"]
    assert info["source"] == "bundled" and info["measured_at"] == bundled()["measured_at"]
    assert info["metrics"]["accuracy"] == bundled()["suites"]["omj-smoke"]["accuracy"]
    md = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert "### Compared with Jev" in md and "bundled Jev 1.13" in md


def test_reference_none_turns_it_off(tmp_path: Path) -> None:
    assert "reference" not in _bench(tmp_path, reference="none")


def test_a_local_jev_run_is_preferred(tmp_path: Path, omj_home: Path) -> None:
    jev = omj_home / "runs" / "jev-local"
    jev.mkdir(parents=True)
    (jev / "report.json").write_text(json.dumps({
        "environment": {"backend": "typesafe", "model": "jev-latest", "run_at": "2026-09-25T00:00:00+00:00"},
        "suites": {"omj-smoke": {"n": 30, "accuracy": 0.9}},
    }), encoding="utf-8")
    info = _bench(tmp_path)["reference"]["suites"]["omj-smoke"]
    assert (info["source"], info["run"], info["metrics"]["accuracy"]) == ("local run", "jev-local", 0.9)


def test_with_a_key_jev_is_measured_once_and_saved(tmp_path: Path, omj_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import omj.cli.bench_cmd as bench_cmd
    from omj.backends.mock import MockBackend

    monkeypatch.setenv("JEV_KEY", "test-" + "placeholder")
    real = bench_cmd.create_backend
    monkeypatch.setattr(bench_cmd, "create_backend", lambda name: MockBackend() if name == "typesafe" else real(name))

    info = _bench(tmp_path)["reference"]["suites"]["omj-smoke"]
    assert info["source"] == "measured now" and info["run"].startswith("jev-reference-")
    saved = omj_home / "runs" / info["run"] / "report.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["environment"]["backend"] == "typesafe"

    again = run_bench(BenchOptions(backend="mock", suites=["omj-smoke"], out=tmp_path / "run2"), out=io.StringIO())
    assert again["reference"]["suites"]["omj-smoke"]["source"] == "local run"  # reused, not measured twice


def test_terminal_summary_shows_the_jev_comparison(tmp_path: Path) -> None:
    buf = io.StringIO()
    run_bench(BenchOptions(backend="mock", suites=["omj-smoke"], out=tmp_path / "run"), out=buf)
    text = buf.getvalue()
    assert "vs Jev" in text and "Jev acc" in text and "bundled Jev 1.13" in text
