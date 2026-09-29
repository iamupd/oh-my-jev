"""Temperature fitting tests: split, golden-section NLL search, calibration report. REQ-039"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omj.bench.temperature import (
    apply_temperatures,
    fit_and_report,
    fit_temperature_for_kind,
    fit_temperatures,
    nll_at_temperature,
    stratified_split,
)
from omj.bench.types import ResultRow
from omj.gateway.assemble import softmax
from omj.gateway.calibration import load_calibration


def make_row(
    *,
    item_id: str = "item",
    suite: str = "synthetic",
    qid: str = "q",
    kind: str = "choice",
    keys: list[str],
    probs: list[float] | None,
    logits: list[float] | None = None,
    expected: str | None = None,
    tags: list[str] | None = None,
) -> ResultRow:
    return ResultRow(
        item_id=item_id,
        suite=suite,
        qid=qid,
        kind=kind,
        keys=keys,
        probs=probs,
        logits=logits,
        expected=expected,
        correct=None,
        latency_ms=10.0,
        cost_usd=None,
        backend="mock",
        model="mock-model",
        tags=tags or [],
    )


def make_synthetic_rows(
    n: int,
    seed: int,
    scale: float,
    noise_frac: float,
    keys: tuple[str, ...] = ("A", "B", "C", "D"),
) -> list[ResultRow]:
    """Generate choice rows whose true labels come from softmax(true_logits), with
    observed logits = true_logits * scale (over/under-confident vs. the true distribution)
    and a noise_frac chance of the label being reassigned uniformly at random.
    """
    rng = np.random.default_rng(seed)
    k = len(keys)
    rows: list[ResultRow] = []
    for i in range(n):
        true_logits = rng.normal(0.0, 2.0, size=k)
        true_probs = softmax(true_logits.tolist())
        expected_idx = int(rng.choice(k, p=true_probs))
        if rng.random() < noise_frac:
            other = [j for j in range(k) if j != expected_idx]
            expected_idx = int(rng.choice(other))
        observed_logits = (true_logits * scale).tolist()
        observed_probs = softmax(observed_logits)
        rows.append(
            make_row(
                item_id=f"item-{i}",
                qid=f"q{i}",
                kind="choice",
                keys=list(keys),
                probs=observed_probs,
                logits=observed_logits,
                expected=keys[expected_idx],
            )
        )
    return rows


class TestStratifiedSplit:
    def test_deterministic_and_stratum_proportions(self) -> None:
        rows: list[ResultRow] = []
        for i in range(20):
            rows.append(make_row(item_id=f"a{i}", keys=["A", "B"], probs=[0.5, 0.5], tags=["A"]))
        for i in range(30):
            rows.append(make_row(item_id=f"b{i}", keys=["A", "B"], probs=[0.5, 0.5], tags=["B"]))
        for i in range(50):
            rows.append(make_row(item_id=f"c{i}", keys=["A", "B"], probs=[0.5, 0.5], tags=["C"]))

        fit1, eval1 = stratified_split(rows, holdout_frac=0.2, seed=20260922)
        fit2, eval2 = stratified_split(rows, holdout_frac=0.2, seed=20260922)

        assert [r.item_id for r in fit1] == [r.item_id for r in fit2]
        assert [r.item_id for r in eval1] == [r.item_id for r in eval2]
        assert len(fit1) + len(eval1) == len(rows)
        assert set(r.item_id for r in fit1).isdisjoint(r.item_id for r in eval1)

        for tag, total in (("A", 20), ("B", 30), ("C", 50)):
            n_fit = sum(1 for r in fit1 if r.tags == [tag])
            assert n_fit == round(total * 0.2)

    def test_untagged_rows_form_none_stratum(self) -> None:
        rows = [make_row(item_id=f"x{i}", keys=["A", "B"], probs=[0.5, 0.5]) for i in range(10)]
        fit_rows, eval_rows = stratified_split(rows, seed=1)
        assert len(fit_rows) == 2
        assert len(eval_rows) == 8

    def test_different_seed_can_change_split(self) -> None:
        rows = [make_row(item_id=f"x{i}", keys=["A", "B"], probs=[0.5, 0.5]) for i in range(30)]
        fit_a, _ = stratified_split(rows, seed=1)
        fit_b, _ = stratified_split(rows, seed=2)
        assert [r.item_id for r in fit_a] != [r.item_id for r in fit_b]


class TestNllAtTemperature:
    def test_skips_rows_without_logits_or_expected(self) -> None:
        rows = [
            make_row(keys=["A", "B"], probs=[0.5, 0.5], logits=None, expected="A"),
            make_row(keys=["A", "B"], probs=[0.5, 0.5], logits=[1.0, -1.0], expected=None),
        ]
        assert nll_at_temperature(rows, 1.0) == 0.0

    def test_lower_temperature_lowers_nll_for_confident_correct_row(self) -> None:
        row = make_row(keys=["A", "B"], probs=[0.9, 0.1], logits=[4.0, 0.0], expected="A")
        nll_sharp = nll_at_temperature([row], 0.5)
        nll_flat = nll_at_temperature([row], 5.0)
        assert nll_sharp < nll_flat


class TestFitTemperatureForKind:
    def test_no_logits_returns_one(self) -> None:
        rows = [
            make_row(kind="choice", keys=["A", "B"], probs=[0.5, 0.5], logits=None, expected="A")
            for _ in range(10)
        ]
        assert fit_temperature_for_kind(rows, "choice") == 1.0

    def test_no_expected_returns_one(self) -> None:
        rows = [
            make_row(kind="choice", keys=["A", "B"], probs=[0.5, 0.5], logits=[1.0, -1.0], expected=None)
            for _ in range(10)
        ]
        assert fit_temperature_for_kind(rows, "choice") == 1.0

    def test_kind_mismatch_returns_one(self) -> None:
        rows = [
            make_row(kind="score", keys=["0", "1"], probs=[0.5, 0.5], logits=[1.0, -1.0], expected="0")
            for _ in range(10)
        ]
        assert fit_temperature_for_kind(rows, "choice") == 1.0


class TestFitTemperatures:
    def test_defaults_missing_kinds_to_one(self) -> None:
        rows = [
            make_row(kind="choice", keys=["A", "B"], probs=[0.5, 0.5], logits=[1.0, -1.0], expected="A")
        ]
        temps = fit_temperatures(rows)
        assert set(temps) == {"noul", "choice", "score"}
        assert temps["noul"] == 1.0
        assert temps["score"] == 1.0


class TestTemperatureDirection:
    def test_overconfident_logits_fit_temperature_above_one(self) -> None:
        rows = make_synthetic_rows(n=600, seed=1, scale=5.0, noise_frac=0.3)
        T = fit_temperature_for_kind(rows, "choice")
        assert T > 1.0

    def test_underconfident_logits_fit_temperature_below_one(self) -> None:
        rows = make_synthetic_rows(n=600, seed=2, scale=0.2, noise_frac=0.3)
        T = fit_temperature_for_kind(rows, "choice")
        assert T < 1.0


class TestApplyTemperatures:
    def test_rows_without_logits_unchanged(self) -> None:
        row = make_row(kind="choice", keys=["A", "B"], probs=[0.7, 0.3], logits=None, expected="A")
        result = apply_temperatures([row], {"choice": 2.0})
        assert result[0].probs == [0.7, 0.3]
        assert result[0] is not row

    def test_rows_with_logits_rescaled(self) -> None:
        row = make_row(kind="choice", keys=["A", "B"], probs=[0.9, 0.1], logits=[2.0, 0.0], expected="A")
        result = apply_temperatures([row], {"choice": 2.0})
        expected_probs = softmax([1.0, 0.0])
        assert result[0].probs == pytest.approx(expected_probs)

    def test_unknown_kind_defaults_to_temperature_one(self) -> None:
        row = make_row(kind="noul", keys=["yes", "no"], probs=[0.6, 0.4], logits=[0.4, -0.4], expected="yes")
        result = apply_temperatures([row], {"choice": 2.0})
        assert result[0].probs == pytest.approx(softmax([0.4, -0.4]))


class TestFitAndReport:
    def test_improves_ece_and_preserves_accuracy(self) -> None:
        rows = make_synthetic_rows(n=600, seed=3, scale=5.0, noise_frac=0.3)
        cal, report = fit_and_report(rows, backend="mock", model="mock-model", suite="synthetic")

        assert report["accuracy_before"] == report["accuracy_after"]
        assert report["ece_after"] < report["ece_before"]
        assert cal.temperatures["choice"] > 1.0
        assert report["n_fit"] + report["n_eval"] == len(rows)

    def test_writes_calibration_file_with_all_fields(self, tmp_path: Path) -> None:
        rows = make_synthetic_rows(n=200, seed=4, scale=5.0, noise_frac=0.3)
        out_path = tmp_path / "cal.json"

        cal, report = fit_and_report(
            rows, backend="mock", model="Qwen/Qwen3.5-2B", suite="synthetic", out_path=out_path
        )

        assert out_path.exists()
        loaded = load_calibration(out_path)
        assert loaded.version == 1
        assert loaded.backend == "mock"
        assert loaded.model == "Qwen/Qwen3.5-2B"
        assert loaded.temperatures == cal.temperatures
        assert loaded.fitted_on == {"suite": "synthetic", "n": report["n_fit"]}
        assert loaded.ece_before == report["ece_before"]
        assert loaded.ece_after == report["ece_after"]
        assert loaded.created_at

    def test_default_out_path_uses_omj_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("OMJ_HOME", str(home))
        rows = make_synthetic_rows(n=100, seed=5, scale=5.0, noise_frac=0.3)

        fit_and_report(rows, backend="semif", model="Qwen/Qwen3.5-2B", suite="synthetic")

        expected_path = home / "calibration" / "semif-Qwen__Qwen3.5-2B.json"
        assert expected_path.exists()

    def test_raises_when_accuracy_would_change(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dataclasses import replace

        import omj.bench.temperature as temperature_module
        from omj.errors import ErrorCode, OmjError

        rows = make_synthetic_rows(n=50, seed=6, scale=5.0, noise_frac=0.3)

        def fake_apply_temperatures(eval_rows: list[ResultRow], temps: dict[str, float]) -> list[ResultRow]:
            # Reverse only the first row's probs (keys/expected untouched) to flip its argmax
            # and simulate a calibration bug that changes accuracy.
            tampered = list(eval_rows)
            if tampered and tampered[0].probs:
                tampered[0] = replace(tampered[0], probs=list(reversed(tampered[0].probs)))
            return tampered

        monkeypatch.setattr(temperature_module, "apply_temperatures", fake_apply_temperatures)

        with pytest.raises(OmjError) as exc_info:
            fit_and_report(rows, backend="mock", model="mock-model", suite="synthetic")

        assert exc_info.value.code == ErrorCode.E_BACKEND
