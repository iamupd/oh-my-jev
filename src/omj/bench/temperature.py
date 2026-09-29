"""Temperature fitting: stratified split, golden-section NLL search, calibration report (design.md 5.6)."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from omj.bench.metrics import compute_metrics
from omj.bench.types import ResultRow
from omj.config import omj_home
from omj.errors import ErrorCode, OmjError
from omj.gateway.assemble import softmax
from omj.gateway.calibration import Calibration, save_calibration

KINDS: tuple[str, ...] = ("noul", "choice", "score")
LOG_T_MIN = math.log(0.05)
LOG_T_MAX = math.log(20.0)
GOLDEN_ITERATIONS = 60
INV_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def stratified_split(
    rows: list[ResultRow],
    holdout_frac: float = 0.2,
    seed: int = 20260922,
) -> tuple[list[ResultRow], list[ResultRow]]:
    """Split rows into (fit_rows, eval_rows), stratified by the first tag (or "none").

    Each stratum contributes ~holdout_frac of its rows to fit_rows; the remainder goes
    to eval_rows. Deterministic for a given row order, holdout_frac, and seed.
    """
    strata: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        key = row.tags[0] if row.tags else "none"
        strata.setdefault(key, []).append(i)

    rng = np.random.default_rng(seed)
    fit_indices: set[int] = set()
    # Sort stratum keys so the RNG is consumed in a fixed order regardless of dict iteration.
    for key in sorted(strata):
        indices = strata[key]
        order = rng.permutation(len(indices))
        n_fit = round(len(indices) * holdout_frac)
        for pos in order[:n_fit]:
            fit_indices.add(indices[pos])

    fit_rows = [row for i, row in enumerate(rows) if i in fit_indices]
    eval_rows = [row for i, row in enumerate(rows) if i not in fit_indices]
    return fit_rows, eval_rows


def nll_at_temperature(rows: list[ResultRow], T: float) -> float:
    """Mean NLL of the expected label under softmax(logits / T) (floor 1e-15).

    Rows without logits, without an expected label, or whose expected label is not in
    keys are skipped. Returns 0.0 if no row qualifies.
    """
    nlls: list[float] = []
    for row in rows:
        if row.logits is None or row.expected is None:
            continue
        try:
            y_index = row.keys.index(row.expected)
        except ValueError:
            continue
        scaled = [logit / T for logit in row.logits]
        probs = softmax(scaled)
        nlls.append(-math.log(max(probs[y_index], 1e-15)))
    if not nlls:
        return 0.0
    return float(np.mean(nlls))


def _golden_section_minimize(f: Callable[[float], float], lo: float, hi: float, iterations: int) -> float:
    """Minimize a unimodal scalar function f over [lo, hi] via golden-section search."""
    a, b = lo, hi
    c = b - INV_GOLDEN * (b - a)
    d = a + INV_GOLDEN * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - INV_GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + INV_GOLDEN * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def fit_temperature_for_kind(rows: list[ResultRow], kind: str) -> float:
    """Fit one temperature for `kind` via golden-section search minimizing NLL.

    Searches u = ln T over [ln 0.05, ln 20] for 60 iterations. Returns 1.0 if no row of
    this kind has both logits and an expected label present in keys.
    """
    kind_rows = [
        row
        for row in rows
        if row.kind == kind
        and row.logits is not None
        and row.expected is not None
        and row.expected in row.keys
    ]
    if not kind_rows:
        return 1.0

    def objective(log_t: float) -> float:
        return nll_at_temperature(kind_rows, float(math.exp(log_t)))

    best_log_t = _golden_section_minimize(objective, LOG_T_MIN, LOG_T_MAX, GOLDEN_ITERATIONS)
    return float(math.exp(best_log_t))


def fit_temperatures(rows: list[ResultRow]) -> dict[str, float]:
    """Fit a temperature for each of noul/choice/score."""
    return {kind: fit_temperature_for_kind(rows, kind) for kind in KINDS}


def apply_temperatures(rows: list[ResultRow], temps: dict[str, float]) -> list[ResultRow]:
    """Return copies of rows with probs = softmax(logits / T). Rows without logits are unchanged."""
    calibrated: list[ResultRow] = []
    for row in rows:
        if row.logits is None:
            calibrated.append(replace(row))
            continue
        T = temps.get(row.kind, 1.0)
        scaled = [logit / T for logit in row.logits]
        calibrated.append(replace(row, probs=softmax(scaled)))
    return calibrated


def fit_and_report(
    rows: list[ResultRow],
    *,
    backend: str,
    model: str,
    suite: str,
    out_path: str | Path | None = None,
) -> tuple[Calibration, dict[str, float | int | None]]:
    """Split, fit temperatures on the 20% holdout, evaluate before/after on the 80%, save, report.

    Raises OmjError(E_BACKEND) if temperature scaling changes accuracy (argmax should be
    invariant under positive temperature scaling; a change indicates a bug upstream).
    """
    fit_rows, eval_rows = stratified_split(rows)
    temperatures = fit_temperatures(fit_rows)

    metrics_before = compute_metrics(eval_rows)
    eval_rows_calibrated = apply_temperatures(eval_rows, temperatures)
    metrics_after = compute_metrics(eval_rows_calibrated)

    if metrics_before.accuracy != metrics_after.accuracy:
        raise OmjError(ErrorCode.E_BACKEND, "temperature changed argmax")

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    cal = Calibration(
        version=1,
        backend=backend,
        model=model,
        temperatures=temperatures,
        fitted_on={"suite": suite, "n": len(fit_rows)},
        ece_before=metrics_before.ece,
        ece_after=metrics_after.ece,
        created_at=created_at,
    )

    if out_path is not None:
        target_path = Path(out_path)
    else:
        model_safe = model.replace("/", "__")
        target_path = omj_home() / "calibration" / f"{backend}-{model_safe}.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(cal, target_path)

    report: dict[str, float | int | None] = {
        "ece_before": metrics_before.ece,
        "ece_after": metrics_after.ece,
        "accuracy_before": metrics_before.accuracy,
        "accuracy_after": metrics_after.accuracy,
        "n_fit": len(fit_rows),
        "n_eval": len(eval_rows),
    }
    return cal, report
