"""compare module and `omj compare` command: compare.md/compare.json, deltas vs. first. # REQ-013"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer

from omj.bench.compare import METRICS, compare_reports, write_compare
from omj.cli import compare_cmd
from omj.errors import ErrorCode, OmjError


def _env(backend: str, model: str, route: str = "local", run_at: str = "2026-09-23T00:00:00+00:00") -> dict[str, Any]:
    return {"backend": backend, "model": model, "route": route, "run_at": run_at}


def _suite(
    *,
    accuracy: float | None,
    hard_accuracy: float | None,
    ece: float | None,
    brier: float | None,
    coverage_at_risk: float | None,
    latency_p50_ms: float | None,
    cost_per_1k: float | None,
) -> dict[str, Any]:
    return {
        "n": 300,
        "accuracy": accuracy,
        "hard_accuracy": hard_accuracy,
        "ece": ece,
        "brier": brier,
        "coverage_at_risk": coverage_at_risk,
        "latency_p50_ms": latency_p50_ms,
        "cost_per_1k": cost_per_1k,
    }


def _write_report(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def make_three_reports(tmp_path: Path) -> list[Path]:
    """Two common suites (massive-ko, omj-smoke) + one suite only in report A ("solo-a").

    Report A and report C share the same backend+model so the default-label
    fallback (parent directory name) is exercised. cost_per_1k is null in the
    baseline report A, exercising null-baseline delta propagation; accuracy is
    null in report C, exercising a null non-baseline value.
    """
    report_a = {
        "environment": _env("semif", "Qwen/Qwen3.5-2B"),
        "suites": {
            "massive-ko": _suite(
                accuracy=0.80,
                hard_accuracy=0.70,
                ece=0.05,
                brier=0.20,
                coverage_at_risk=0.60,
                latency_p50_ms=100.0,
                cost_per_1k=None,
            ),
            "omj-smoke": _suite(
                accuracy=0.90,
                hard_accuracy=0.85,
                ece=0.02,
                brier=0.10,
                coverage_at_risk=0.95,
                latency_p50_ms=50.0,
                cost_per_1k=1.0,
            ),
            "solo-a": _suite(
                accuracy=0.5,
                hard_accuracy=0.5,
                ece=0.1,
                brier=0.3,
                coverage_at_risk=0.5,
                latency_p50_ms=10.0,
                cost_per_1k=0.1,
            ),
        },
    }
    report_b = {
        "environment": _env("typesafe", "jev-latest"),
        "suites": {
            "massive-ko": _suite(
                accuracy=0.85,
                hard_accuracy=0.75,
                ece=0.04,
                brier=0.18,
                coverage_at_risk=0.65,
                latency_p50_ms=90.0,
                cost_per_1k=1.2,
            ),
            "omj-smoke": _suite(
                accuracy=0.92,
                hard_accuracy=0.87,
                ece=0.015,
                brier=0.09,
                coverage_at_risk=0.96,
                latency_p50_ms=45.0,
                cost_per_1k=1.1,
            ),
        },
    }
    report_c = {
        "environment": _env("semif", "Qwen/Qwen3.5-2B"),
        "suites": {
            "massive-ko": _suite(
                accuracy=None,
                hard_accuracy=0.78,
                ece=0.03,
                brier=0.17,
                coverage_at_risk=0.70,
                latency_p50_ms=80.0,
                cost_per_1k=1.3,
            ),
            "omj-smoke": _suite(
                accuracy=0.95,
                hard_accuracy=0.90,
                ece=0.01,
                brier=0.08,
                coverage_at_risk=0.97,
                latency_p50_ms=40.0,
                cost_per_1k=1.05,
            ),
        },
    }

    return [
        _write_report(tmp_path / "before" / "report.json", report_a),
        _write_report(tmp_path / "jev" / "report.json", report_b),
        _write_report(tmp_path / "after" / "report.json", report_c),
    ]


class TestCompareReports:
    def test_common_suites_only(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        assert set(result["suites"].keys()) == {"massive-ko", "omj-smoke"}
        assert "solo-a" not in result["suites"]

    def test_metrics_columns_match_metrics_constant(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        for suite_data in result["suites"].values():
            for metric in METRICS:
                assert metric in suite_data
                assert metric in suite_data["delta"]

    def test_deltas_computed_against_first_report(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        ko = result["suites"]["massive-ko"]
        assert ko["accuracy"] == [0.80, 0.85, None]
        assert ko["delta"]["accuracy"] == [None, pytest.approx(0.05), None]
        assert ko["hard_accuracy"] == [0.70, 0.75, 0.78]
        assert ko["delta"]["hard_accuracy"] == [None, pytest.approx(0.05), pytest.approx(0.08)]
        assert ko["delta"]["ece"] == [None, pytest.approx(-0.01), pytest.approx(-0.02)]

    def test_null_baseline_propagates_null_delta(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        ko = result["suites"]["massive-ko"]
        assert ko["cost_per_1k"] == [None, 1.2, 1.3]
        assert ko["delta"]["cost_per_1k"] == [None, None, None]

    def test_null_non_baseline_value_has_null_delta(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        ko = result["suites"]["massive-ko"]
        assert ko["accuracy"][2] is None
        assert ko["delta"]["accuracy"][2] is None

    def test_default_labels_fall_back_to_parent_dir_on_collision(self, tmp_path: Path) -> None:
        # REQ-013: report A and C share backend+model ("semif+Qwen3.5-2B"), so
        # both fall back to their parent directory name; report B is unique.
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        assert result["labels"] == ["before", "typesafe+jev-latest", "after"]

    def test_report_metadata_recorded(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        assert result["reports"][0]["backend"] == "semif"
        assert result["reports"][0]["model"] == "Qwen/Qwen3.5-2B"
        assert result["reports"][1]["route"] == "local"
        assert result["reports"][2]["run_at"] == "2026-09-23T00:00:00+00:00"
        assert result["reports"][0]["path"] == str(paths[0])

    def test_explicit_labels_used_verbatim(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths, labels=["Jev", "학습 전", "학습 후"])
        assert result["labels"] == ["Jev", "학습 전", "학습 후"]

    def test_label_count_mismatch_raises_e_config(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        with pytest.raises(OmjError) as exc_info:
            compare_reports(paths, labels=["only-one", "two"])
        assert exc_info.value.code == ErrorCode.E_CONFIG

    def test_fewer_than_two_reports_raises_e_config(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        with pytest.raises(OmjError) as exc_info:
            compare_reports(paths[:1])
        assert exc_info.value.code == ErrorCode.E_CONFIG

    def test_no_common_suites_raises_e_config(self, tmp_path: Path) -> None:
        # REQ-013
        report_x = {"environment": _env("mock", "mock-a"), "suites": {"suite-x": _suite(
            accuracy=0.5, hard_accuracy=0.5, ece=0.1, brier=0.2, coverage_at_risk=0.5,
            latency_p50_ms=1.0, cost_per_1k=1.0,
        )}}
        report_y = {"environment": _env("mock", "mock-b"), "suites": {"suite-y": _suite(
            accuracy=0.5, hard_accuracy=0.5, ece=0.1, brier=0.2, coverage_at_risk=0.5,
            latency_p50_ms=1.0, cost_per_1k=1.0,
        )}}
        paths = [
            _write_report(tmp_path / "x" / "report.json", report_x),
            _write_report(tmp_path / "y" / "report.json", report_y),
        ]
        with pytest.raises(OmjError) as exc_info:
            compare_reports(paths)
        assert exc_info.value.code == ErrorCode.E_CONFIG

    def test_missing_report_raises_e_config_not_filenotfound(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        missing = tmp_path / "gone" / "report.json"

        with pytest.raises(OmjError) as exc_info:
            compare_reports([paths[0], missing])

        assert exc_info.value.code == ErrorCode.E_CONFIG
        assert "report not readable" in exc_info.value.message
        assert str(missing) in exc_info.value.message

    def test_unparsable_report_raises_e_config_not_jsondecodeerror(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        broken = tmp_path / "broken" / "report.json"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_text("{not json", encoding="utf-8")

        with pytest.raises(OmjError) as exc_info:
            compare_reports([paths[0], broken])

        assert exc_info.value.code == ErrorCode.E_CONFIG
        assert "report not readable" in exc_info.value.message

    def test_report_whose_top_level_is_not_an_object_raises_e_config(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        listy = tmp_path / "listy" / "report.json"
        listy.parent.mkdir(parents=True, exist_ok=True)
        listy.write_text("[]", encoding="utf-8")

        with pytest.raises(OmjError) as exc_info:
            compare_reports([paths[0], listy])

        assert exc_info.value.code == ErrorCode.E_CONFIG


class TestWriteCompare:
    def test_writes_both_files(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        out_dir = tmp_path / "out"
        json_path, md_path = write_compare(out_dir, result)
        assert json_path == out_dir / "compare.json"
        assert md_path == out_dir / "compare.md"
        assert json_path.exists()
        assert md_path.exists()

    def test_compare_json_round_trips(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        json_path, _ = write_compare(tmp_path / "out", result)
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded == result

    def test_compare_md_contains_tables_and_na_no_em_dash(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        _, md_path = write_compare(tmp_path / "out", result)
        text = md_path.read_text(encoding="utf-8")

        assert "N/A" in text
        assert "—" not in text
        assert "massive-ko" in text
        assert "omj-smoke" in text
        assert "| Metric | before | typesafe+jev-latest | after |" in text
        assert "0.8500 (Δ+0.0500)" in text
        assert "Label" in text
        assert "Backend" in text

    def test_compare_md_uses_lf_line_endings(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        result = compare_reports(paths)
        json_path, md_path = write_compare(tmp_path / "out", result)
        assert b"\r\n" not in json_path.read_bytes()
        assert b"\r\n" not in md_path.read_bytes()


class TestCompareCommand:
    def test_register_adds_compare_command(self) -> None:
        # REQ-013
        app = typer.Typer()
        compare_cmd.register(app)
        names = [info.name or info.callback.__name__ for info in app.registered_commands]
        assert "compare" in names

    def test_direct_call_writes_files_and_prints_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # REQ-013: CliRunner-free — call the registered callback directly, bypassing
        # click's argument parsing entirely.
        paths = make_three_reports(tmp_path)
        app = typer.Typer()
        compare_cmd.register(app)
        callback = app.registered_commands[0].callback
        assert callback is not None

        out_dir = tmp_path / "cmd-out"
        callback(
            reports=paths,
            out=out_dir,
            labels="Jev,학습 전,학습 후",
            json_output=True,
        )

        captured = capsys.readouterr()
        printed = json.loads(captured.out)
        assert printed["labels"] == ["Jev", "학습 전", "학습 후"]
        assert (out_dir / "compare.json").exists()
        assert (out_dir / "compare.md").exists()

    def test_direct_call_without_json_prints_summary_not_raw_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        app = typer.Typer()
        compare_cmd.register(app)
        callback = app.registered_commands[0].callback
        assert callback is not None

        out_dir = tmp_path / "cmd-out-2"
        callback(reports=paths, out=out_dir, labels=None, json_output=False)

        captured = capsys.readouterr()
        assert "compare.json" in captured.out
        assert "compare.md" in captured.out
        with pytest.raises(json.JSONDecodeError):
            json.loads(captured.out)

    def test_direct_call_propagates_e_config(self, tmp_path: Path) -> None:
        # REQ-013
        paths = make_three_reports(tmp_path)
        app = typer.Typer()
        compare_cmd.register(app)
        callback = app.registered_commands[0].callback
        assert callback is not None

        with pytest.raises(OmjError) as exc_info:
            callback(reports=paths[:1], out=tmp_path / "cmd-out-3", labels=None, json_output=False)
        assert exc_info.value.code == ErrorCode.E_CONFIG


def test_compare_md_omits_metrics_no_report_measured(tmp_path: Path) -> None:
    from omj.bench.compare import _render_markdown, compare_reports

    base = {"omj": {"version": "0"}, "environment": {"backend": "mock", "model": "m", "route": "local", "run_at": "x"}}
    paths = []
    for i, acc in enumerate((0.7, 0.8)):
        report = dict(base, suites={"s": {"n": 10, "accuracy": acc, "order_shift_max": None}})
        path = tmp_path / f"r{i}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    text = _render_markdown(compare_reports(paths))
    assert "| accuracy |" in text
    assert "order_shift_max" not in text


def test_compare_cli_prints_terminal_tables(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from omj.cli.main import app

    base = {"omj": {"version": "0"}, "environment": {"backend": "mock", "model": "m", "route": "local", "run_at": "2026-09-28T00:00:00+00:00"}}
    paths = []
    for i, (acc, ece) in enumerate(((0.70, 0.05), (0.80, 0.08))):
        path = tmp_path / f"r{i}.json"
        path.write_text(json.dumps(dict(base, suites={"s": {"n": 10, "accuracy": acc, "ece": ece}})), encoding="utf-8")
        paths.append(str(path))
    result = CliRunner().invoke(app, ["compare", *paths, "--labels", "old,new", "--out", str(tmp_path / "cmp")])

    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "Reports" in out and "old" in out and "new" in out
    assert "0.800 (+0.100)" in out and "0.080 (+0.030)" in out
    assert "| ---" not in out  # terminal tables, not markdown
    assert (tmp_path / "cmp" / "compare.md").exists() and "Wrote" in out
