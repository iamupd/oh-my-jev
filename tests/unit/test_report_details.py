"""report.md content: run header, CI, breakdowns, and N/A handling."""

from __future__ import annotations

import json
from pathlib import Path

from omj.bench.details import suite_details, wilson_interval
from omj.bench.report import Environment, write_report
from omj.bench.types import Metrics, ResultRow


def _row(i: int, correct: bool, conf: float, kind: str = "choice", tags: tuple[str, ...] = ()) -> ResultRow:
    keys = ["a", "b"]
    probs = [conf, 1.0 - conf] if correct else [1.0 - conf, conf]
    return ResultRow(
        item_id=f"i{i}", suite="s", qid="q", kind=kind, keys=keys, probs=probs, logits=None,
        expected="a", correct=None, latency_ms=10.0, cost_usd=None, backend="mock", model="m", tags=list(tags),
    )


def _env() -> Environment:
    return Environment(hardware={}, backend="mock", model="mock-model", revision="", quant="",
                       route="local", git_commit="abc1234", omj_version="0.0.0", run_at="2026-09-28T01:02:03+00:00")


def test_wilson_interval_brackets_the_proportion() -> None:
    lo, hi = wilson_interval(80, 100)
    assert lo < 0.8 < hi and 0.70 < lo and hi < 0.87
    assert wilson_interval(0, 0) is None


def test_suite_details_breakdowns_skip_rare_and_universal_tags() -> None:
    rows = [_row(i, i % 4 != 0, 0.9, tags=("s", "hard" if i < 6 else "easy")) for i in range(12)]
    rows.append(_row(99, True, 0.9, tags=("s", "rare")))
    d = suite_details(rows, n_items=13, wall_seconds=2.0, n_errors=0, stopped_early=False, stop_reason="")
    assert d["n_scored"] == 13
    assert set(d["by_tag"]) == {"hard", "easy"}  # "s" is on every row, "rare" on one
    assert d["accuracy_ci95"][0] < d["n_correct"] / 13 < d["accuracy_ci95"][1]
    assert sum(b["n"] for b in d["reliability"]) == 13
    assert d["items_per_second"] == 6.5


def test_report_md_shows_run_header_and_hides_not_measured_metrics(tmp_path: Path) -> None:
    rows = [_row(i, i % 3 != 0, 0.8, kind="noul" if i % 2 else "choice", tags=("s", "t1")) for i in range(9)]
    rows += [_row(10 + i, True, 0.95, tags=("s", "t2")) for i in range(6)]
    metrics = Metrics(n=15, n_valid=15, n_unattempted=0, accuracy=0.8, ece=0.05, brier=0.2, nll=0.4,
                      coverage_at_risk=0.5, valid_vector_rate=1.0, latency_p50_ms=10.0, latency_p95_ms=12.0)
    details = {"s": suite_details(rows, n_items=15, wall_seconds=3.0, n_errors=0, stopped_early=False, stop_reason="")}
    run = {"started_at": "2026-09-28T01:00:00+00:00", "finished_at": "2026-09-28T01:02:03+00:00",
           "wall_seconds": 123.0, "command": "omj bench --backend mock --suite s", "options": {"suites": ["s"]}}
    json_path, md_path = write_report(tmp_path, suite_metrics={"s": metrics}, env=_env(), run=run, details=details)

    text = md_path.read_text(encoding="utf-8")
    assert "2026-09-28 01:02:03 UTC" in text and "2m 03s" in text
    assert "`omj bench --backend mock --suite s`" in text
    assert "Accuracy [95% CI]" in text and "0.8000 [" in text
    assert "By question type" in text and "By tag" in text and "Reliability" in text
    assert "N/A" not in text
    assert "Not measured for this suite:" in text and "`order_shift_max` (`--suite order`)" in text
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["suites"]["s"]["order_shift_max"] is None  # report.json keeps every field
    assert data["run"]["command"].startswith("omj bench") and "s" in data["details"]
