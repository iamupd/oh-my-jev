"""Live evaluation with TypeSafeBackend (requires JEV_KEY).
# REQ-050
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omj.backends.typesafe import TypeSafeBackend
from omj.bench.metrics import compute_metrics
from omj.bench.runner import BackendTarget, run_suite
from omj.bench.suites.local import load_local
from omj.config import BackendSection


@pytest.mark.live
def test_live_eval_set_typesafe_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run evals/omj-p0.jsonl through TypeSafeBackend and verify >= 90% accuracy.

    Requires environment variable JEV_KEY to be set. This test is marked as
    @pytest.mark.live and is skipped by default unless explicitly enabled.
    """
    # REQ-050
    jev_key = os.environ.get("JEV_KEY")
    if not jev_key:
        pytest.skip("JEV_KEY not set")

    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)

    evals_path = Path(__file__).parents[2] / "evals" / "omj-p0.jsonl"

    items = load_local(evals_path, suite_name="omj-p0")
    assert len(items) == 30, f"Expected 30 items, got {len(items)}"

    backend = TypeSafeBackend()
    backend.load(BackendSection(name="typesafe", provider="typesafe"))
    target = BackendTarget(backend)

    result = run_suite(
        items,
        target,
        suite_name="omj-p0",
        backend_name="typesafe",
        model="jev-latest",
    )

    assert not result.stopped_early, f"Run stopped early: {result.stop_reason}"
    assert result.stop_reason is None or result.stop_reason == ""
    assert result.n_errors == 0, f"Expected no errors, got {result.n_errors}"
    assert len(result.rows) > 0, "Expected result rows"

    metrics = compute_metrics(result.rows, input_tokens_total=result.input_tokens_total)

    assert metrics.accuracy is not None, "Expected accuracy to be computed"
    assert metrics.accuracy >= 0.90, f"Expected accuracy >= 0.90, got {metrics.accuracy}"

    print(f"Accuracy: {metrics.accuracy:.1%}")
    print(f"Total input tokens: {result.input_tokens_total}")
