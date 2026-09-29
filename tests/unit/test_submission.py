"""JevBench submission bundle contract tests. # REQ-042"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omj import __version__
from omj.bench.submission import write_submission
from omj.bench.types import ResultRow


def make_row(
    *,
    item_id: str = "dec-1",
    suite: str = "jevbench",
    qid: str = "decision",
    kind: str = "choice",
    keys: list[str] | None = None,
    probs: list[float] | None = None,
    logits: list[float] | None = None,
    expected: str | None = None,
    correct: bool | None = None,
    latency_ms: float = 120.0,
    cost_usd: float | None = None,
    backend: str = "mock",
    model: str = "mock-model",
    error: str | None = None,
) -> ResultRow:
    return ResultRow(
        item_id=item_id,
        suite=suite,
        qid=qid,
        kind=kind,
        keys=keys if keys is not None else ["A", "B"],
        probs=probs,
        logits=logits,
        expected=expected,
        correct=correct,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        backend=backend,
        model=model,
        error=error,
    )


class TestWriteSubmission:
    def test_creates_expected_files(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [
            make_row(item_id="dec-1", keys=["A", "B"], probs=[0.7, 0.3], expected="A", correct=True),
            make_row(
                item_id="dec-2",
                kind="noul",
                keys=["yes", "no"],
                probs=[0.6, 0.4],
                expected="yes",
                correct=True,
            ),
        ]
        submission_dir = write_submission(
            tmp_path, rows, model="mock-model", backend="mock", price_input_per_m=None
        )

        assert submission_dir == tmp_path / "submission"
        assert (submission_dir / "results.jsonl").exists()
        assert (submission_dir / "manifest.json").exists()
        assert (submission_dir / "ledger.jsonl").exists()
        assert (submission_dir / "raw").is_dir()
        assert (submission_dir / "raw" / "dec-1.json").exists()
        assert (submission_dir / "raw" / "dec-2.json").exists()

    def test_results_jsonl_schema(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [
            make_row(
                item_id="dec-1",
                keys=["A", "B", "C"],
                probs=[0.6, 0.3, 0.1],
                expected="A",
                correct=True,
                cost_usd=0.002,
            ),
        ]
        submission_dir = write_submission(
            tmp_path, rows, model="mock-model", backend="mock", price_input_per_m=0.5
        )

        lines = (submission_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])

        assert set(row.keys()) == {
            "decision_id",
            "probabilities",
            "latency_seconds",
            "cost_usd",
            "correct",
            "model",
            "adapter",
            "status",
            "error",
        }
        assert row["decision_id"] == "dec-1"
        assert row["probabilities"] == {"A": 0.6, "B": 0.3, "C": 0.1}
        assert row["latency_seconds"] == pytest.approx(0.12)
        assert row["cost_usd"] == 0.002
        assert row["correct"] is True
        assert row["model"] == "mock-model"
        assert row["adapter"] == "omj"
        assert row["status"] == "ok"
        assert row["error"] is None

    def test_noul_probabilities_are_yes_no(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [make_row(item_id="dec-1", kind="noul", keys=["yes", "no"], probs=[0.7, 0.3])]
        submission_dir = write_submission(
            tmp_path, rows, model="m", backend="mock", price_input_per_m=None
        )
        row = json.loads(
            (submission_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert row["probabilities"]["yes"] == pytest.approx(0.7)
        assert row["probabilities"]["no"] == pytest.approx(0.3)

    def test_error_row_status_error(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [make_row(item_id="dec-err", probs=None, error="E_BACKEND: timeout")]
        submission_dir = write_submission(
            tmp_path, rows, model="m", backend="mock", price_input_per_m=None
        )
        row = json.loads(
            (submission_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert row["status"] == "error"
        assert row["error"] == "E_BACKEND: timeout"
        assert row["probabilities"] == {}
        assert row["correct"] is None

    def test_cost_usd_null_when_row_has_no_cost(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [make_row(item_id="dec-1", cost_usd=None)]
        submission_dir = write_submission(
            tmp_path, rows, model="m", backend="mock", price_input_per_m=None
        )
        row = json.loads(
            (submission_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert row["cost_usd"] is None

    def test_manifest_json_fields(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [make_row(item_id="dec-1")]
        submission_dir = write_submission(
            tmp_path, rows, model="mock-model", backend="mock", price_input_per_m=None
        )
        manifest = json.loads((submission_dir / "manifest.json").read_text(encoding="utf-8"))

        assert set(manifest.keys()) == {
            "adapter",
            "model",
            "backend",
            "omj_version",
            "created_utc",
            "n_decisions",
            "price_input_per_m",
            "price_output_per_m",
            "notes",
        }
        assert manifest["adapter"] == "omj"
        assert manifest["model"] == "mock-model"
        assert manifest["backend"] == "mock"
        assert manifest["omj_version"] == __version__
        assert manifest["created_utc"]
        assert manifest["n_decisions"] == 1
        assert manifest["price_input_per_m"] is None
        assert manifest["price_output_per_m"] == 0

    def test_manifest_price_input_per_m_set(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [make_row(item_id="dec-1")]
        submission_dir = write_submission(
            tmp_path, rows, model="mock-model", backend="semif", price_input_per_m=1.25
        )
        manifest = json.loads((submission_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["price_input_per_m"] == 1.25
        assert manifest["price_output_per_m"] == 0

    def test_ledger_jsonl_null_when_no_cost(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [make_row(item_id="dec-1", cost_usd=None), make_row(item_id="dec-2", cost_usd=None)]
        submission_dir = write_submission(
            tmp_path, rows, model="m", backend="mock", price_input_per_m=None
        )
        ledger_lines = (submission_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(ledger_lines) == 1
        entry = json.loads(ledger_lines[0])
        assert entry["reserved_usd"] is None
        assert entry["settled_usd"] is None

    def test_ledger_jsonl_sums_known_costs(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [
            make_row(item_id="dec-1", cost_usd=0.01),
            make_row(item_id="dec-2", cost_usd=0.02),
        ]
        submission_dir = write_submission(
            tmp_path, rows, model="m", backend="semif", price_input_per_m=None
        )
        entry = json.loads(
            (submission_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert entry["reserved_usd"] is None
        assert entry["settled_usd"] == pytest.approx(0.03)

    def test_raw_files_contain_decision_id(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [make_row(item_id="dec-1", keys=["A", "B"], probs=[0.5, 0.5])]
        submission_dir = write_submission(
            tmp_path, rows, model="m", backend="mock", price_input_per_m=None
        )
        raw = json.loads((submission_dir / "raw" / "dec-1.json").read_text(encoding="utf-8"))
        assert raw["decision_id"] == "dec-1"

    def test_files_use_lf_line_endings(self, tmp_path: Path) -> None:
        # REQ-042
        rows = [make_row(item_id="dec-1")]
        submission_dir = write_submission(
            tmp_path, rows, model="m", backend="mock", price_input_per_m=None
        )
        for name in ("results.jsonl", "manifest.json", "ledger.jsonl"):
            raw_bytes = (submission_dir / name).read_bytes()
            assert b"\r\n" not in raw_bytes
