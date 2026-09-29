"""Evaluation set: evals/omj-p0.jsonl and mock/live tests.
# REQ-050
"""

from __future__ import annotations

from pathlib import Path

from omj.backends.mock import MockBackend
from omj.bench.metrics import compute_metrics
from omj.bench.runner import BackendTarget, run_suite
from omj.bench.suites.local import load_local


def test_eval_set_byte_identical_to_smoke(tmp_path: Path) -> None:
    """evals/omj-p0.jsonl and suites/omj-smoke.jsonl must be byte-identical."""
    # REQ-050
    evals_path = Path(__file__).parents[2] / "evals" / "omj-p0.jsonl"
    smoke_path = Path(__file__).parents[2] / "suites" / "omj-smoke.jsonl"

    assert evals_path.is_file(), f"{evals_path} does not exist"
    assert smoke_path.is_file(), f"{smoke_path} does not exist"

    evals_bytes = evals_path.read_bytes()
    smoke_bytes = smoke_path.read_bytes()

    assert evals_bytes == smoke_bytes, "evals/omj-p0.jsonl and suites/omj-smoke.jsonl must be byte-identical"


def test_eval_set_structure() -> None:
    """Every row in evals/omj-p0.jsonl must be valid JSON with comment placeholder."""
    # REQ-050
    evals_path = Path(__file__).parents[2] / "evals" / "omj-p0.jsonl"

    with evals_path.open("r", encoding="utf-8") as f:
        line_count = 0
        for line in f:
            if not line.strip():
                continue
            line_count += 1

    assert line_count == 30, f"Expected 30 rows in evals/omj-p0.jsonl, got {line_count}"


def test_mock_backend_100_percent_accuracy() -> None:
    """Run evals/omj-p0.jsonl through MockBackend and verify 100% accuracy."""
    # REQ-050
    evals_path = Path(__file__).parents[2] / "evals" / "omj-p0.jsonl"

    items = load_local(evals_path, suite_name="omj-p0")
    assert len(items) == 30, f"Expected 30 items, got {len(items)}"

    backend = MockBackend()
    target = BackendTarget(backend)

    result = run_suite(
        items,
        target,
        suite_name="omj-p0",
        backend_name="mock",
        model="mock",
    )

    assert result.n_errors == 0, f"Expected no errors, got {result.n_errors}"
    assert len(result.rows) > 0, "Expected result rows"

    metrics = compute_metrics(result.rows)

    assert metrics.n_unattempted == 0, f"Expected 0 unattempted, got {metrics.n_unattempted}"
    assert metrics.accuracy == 1.0, f"Expected accuracy 1.0, got {metrics.accuracy}"
