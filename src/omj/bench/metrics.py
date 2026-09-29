"""Metric computations for omj bench results (design.md 4.3)."""

from __future__ import annotations

import numpy as np

from omj.bench.types import Metrics, ResultRow, is_correct, predicted_key

STRICT_SUM_TOLERANCE = 1e-3
LENIENT_SUM_TOLERANCE = 2e-2
RISK_BUDGET_DEFAULT = 0.05
HIGH_CONF_CUTOFF = 0.9
HARD_TAG = "hard"


def is_valid_probability_vector(keys: list[str], probs: list[float] | None, tol: float) -> bool:
    """A vector is valid when it exists, aligns 1:1 with keys, stays in [0,1], and sums to ~1."""
    if probs is None:
        return False
    if len(probs) != len(keys):
        return False
    if any(p < 0.0 or p > 1.0 for p in probs):
        return False
    return abs(sum(probs) - 1.0) <= tol


def top_label_ece(confidences: list[float], corrects: list[bool], bins: int) -> float:
    """Top-label expected calibration error over `bins` equal-width bins; empty bins skipped."""
    n = len(confidences)
    if n == 0:
        return 0.0

    total_error = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        if b == bins - 1:
            # last bin is closed on both ends so a confidence of exactly 1.0 is counted.
            idxs = [i for i in range(n) if lo <= confidences[i] <= hi]
        else:
            idxs = [i for i in range(n) if lo <= confidences[i] < hi]
        if not idxs:
            continue
        bin_conf = sum(confidences[i] for i in idxs) / len(idxs)
        bin_acc = sum(1.0 for i in idxs if corrects[i]) / len(idxs)
        total_error += (len(idxs) / n) * abs(bin_conf - bin_acc)
    return total_error


def brier_score(probs: list[float], y_index: int) -> float:
    """Multiclass Brier score for one row: sum_k (p_k - y_k)^2 with y one-hot at y_index."""
    return sum((p - (1.0 if i == y_index else 0.0)) ** 2 for i, p in enumerate(probs))


def chance_corrected_score(correct: bool, k: int) -> float:
    chance = 1.0 / k
    return (float(correct) - chance) / (1.0 - chance)


def coverage_at_risk(sorted_corrects: list[bool], risk_budget: float = RISK_BUDGET_DEFAULT) -> float:
    """Largest prefix fraction (by confidence-descending order) whose error rate <= risk_budget."""
    n = len(sorted_corrects)
    if n == 0:
        return 0.0

    best_len = 0
    wrong = 0
    for i, correct in enumerate(sorted_corrects, start=1):
        if not correct:
            wrong += 1
        if wrong / i <= risk_budget:
            best_len = i
    return best_len / n


def order_shift_for_group(group_rows: list[ResultRow]) -> float:
    """Max over answer keys of (max prob - min prob) across shuffled-order rows of one item/qid."""
    key_probs: dict[str, list[float]] = {}
    for row in group_rows:
        if row.probs is None:
            continue
        for key, p in zip(row.keys, row.probs):
            key_probs.setdefault(key, []).append(p)
    shifts = [max(ps) - min(ps) for ps in key_probs.values() if len(ps) > 1]
    return max(shifts) if shifts else 0.0


def _ko_en_pair_metrics(valid_rows: list[ResultRow]) -> tuple[float | None, float | None]:
    """Pair over the same valid-probability-vector rows that drive accuracy.

    ``valid_rows`` (not the wider ``attempted`` set) so a row with a malformed
    probability vector never contributes to the gap/agreement pair either.
    """
    ko_rows = {(r.pair_id, r.qid): r for r in valid_rows if r.suite == "massive-ko" and r.pair_id is not None}
    en_rows = {(r.pair_id, r.qid): r for r in valid_rows if r.suite == "massive-en" and r.pair_id is not None}
    shared = ko_rows.keys() & en_rows.keys()
    if not shared:
        return None, None

    ko_corrects: list[bool] = []
    en_corrects: list[bool] = []
    agreements: list[bool] = []
    for key in shared:
        ko_row = ko_rows[key]
        en_row = en_rows[key]
        if ko_row.expected is not None and en_row.expected is not None:
            ko_corrects.append(bool(is_correct(ko_row)))
            en_corrects.append(bool(is_correct(en_row)))
        ko_key = predicted_key(ko_row)
        en_key = predicted_key(en_row)
        # Two unpredictable rows agreeing that neither has a prediction is not
        # agreement -- skip pairs where either side has no predicted_key.
        if ko_key is not None and en_key is not None:
            agreements.append(ko_key == en_key)

    gap = None
    if ko_corrects and en_corrects:
        gap = (sum(en_corrects) / len(en_corrects)) - (sum(ko_corrects) / len(ko_corrects))
    agreement = (sum(1 for a in agreements if a) / len(agreements)) if agreements else None
    return gap, agreement


def compute_metrics(
    rows: list[ResultRow],
    *,
    gpu_cost_per_hour: float | None = None,
    total_wall_seconds: float | None = None,
    price_input_per_mtok: float | None = None,
    input_tokens_total: int | None = None,
    order_groups: dict[str, list[ResultRow]] | None = None,
) -> Metrics:
    n = len(rows)
    attempted = [r for r in rows if r.error is None]
    n_unattempted = n - len(attempted)
    n_attempted = len(attempted)

    valid_rows = [r for r in attempted if is_valid_probability_vector(r.keys, r.probs, STRICT_SUM_TOLERANCE)]
    valid_rows_lenient = [
        r for r in attempted if is_valid_probability_vector(r.keys, r.probs, LENIENT_SUM_TOLERANCE)
    ]
    n_valid = len(valid_rows)

    valid_vector_rate = (n_valid / n_attempted) if n_attempted else None
    valid_vector_rate_lenient = (len(valid_rows_lenient) / n_attempted) if n_attempted else None

    scored_rows = [r for r in valid_rows if r.expected is not None]
    corrects = [bool(is_correct(r)) for r in scored_rows]
    accuracy = (sum(corrects) / len(corrects)) if scored_rows else None

    chance_corrected_values = [
        chance_corrected_score(c, len(r.keys)) for r, c in zip(scored_rows, corrects) if len(r.keys) > 1
    ]
    chance_corrected_accuracy = (
        sum(chance_corrected_values) / len(chance_corrected_values) if chance_corrected_values else None
    )

    hard_corrects = [bool(is_correct(r)) for r in scored_rows if HARD_TAG in r.tags]
    hard_accuracy = (sum(hard_corrects) / len(hard_corrects)) if hard_corrects else None

    confidences = [max(r.probs) for r in scored_rows]
    ece = top_label_ece(confidences, corrects, 15) if scored_rows else None
    ece_jevbench = top_label_ece(confidences, corrects, 10) if scored_rows else None

    briers: list[float] = []
    nlls: list[float] = []
    for r in scored_rows:
        try:
            y_index = r.keys.index(r.expected)  # type: ignore[arg-type]
        except ValueError:
            continue
        briers.append(brier_score(r.probs, y_index))  # type: ignore[arg-type]
        nlls.append(float(-np.log(max(r.probs[y_index], 1e-15))))  # type: ignore[index]
    brier = (sum(briers) / len(briers)) if briers else None
    nll = (sum(nlls) / len(nlls)) if nlls else None

    coverage_sorted = sorted(scored_rows, key=lambda r: max(r.probs), reverse=True)  # type: ignore[arg-type]
    coverage_value = (
        coverage_at_risk([bool(is_correct(r)) for r in coverage_sorted]) if scored_rows else None
    )

    order_shift_max = None
    if order_groups:
        shifts = [order_shift_for_group(g) for g in order_groups.values()]
        order_shift_max = max(shifts) if shifts else None

    underdetermined_rows = [r for r in attempted if r.expected_shape is not None and r.probs is not None]
    overconfidence_rate = None
    mean_max_prob = None
    if underdetermined_rows:
        max_probs = [max(r.probs) for r in underdetermined_rows]  # type: ignore[arg-type]
        overconfidence_rate = sum(1 for p in max_probs if p >= HIGH_CONF_CUTOFF) / len(max_probs)
        mean_max_prob = sum(max_probs) / len(max_probs)

    uniform_rows = [r for r in attempted if r.expected_shape == "uniform" and r.probs is not None]
    uniform_deviation = None
    if uniform_rows:
        deviations = [abs(max(r.probs) - 1.0 / len(r.keys)) for r in uniform_rows]  # type: ignore[arg-type]
        uniform_deviation = sum(deviations) / len(deviations)

    ko_en_gap, pair_agreement = _ko_en_pair_metrics(valid_rows)

    latencies = [r.latency_ms for r in attempted]
    latency_p50_ms = float(np.percentile(latencies, 50)) if latencies else None
    latency_p95_ms = float(np.percentile(latencies, 95)) if latencies else None

    cost_per_1k = None
    if n_attempted > 0:
        if gpu_cost_per_hour is not None and total_wall_seconds is not None:
            cost_per_1k = gpu_cost_per_hour * (total_wall_seconds / 3600.0) / n_attempted * 1000.0
        elif price_input_per_mtok is not None and input_tokens_total is not None:
            cost_per_1k = input_tokens_total * price_input_per_mtok / 1e6 / n_attempted * 1000.0

    return Metrics(
        n=n,
        n_valid=n_valid,
        n_unattempted=n_unattempted,
        accuracy=accuracy,
        chance_corrected_accuracy=chance_corrected_accuracy,
        hard_accuracy=hard_accuracy,
        ece=ece,
        ece_jevbench=ece_jevbench,
        brier=brier,
        nll=nll,
        coverage_at_risk=coverage_value,
        valid_vector_rate=valid_vector_rate,
        valid_vector_rate_lenient=valid_vector_rate_lenient,
        order_shift_max=order_shift_max,
        overconfidence_rate=overconfidence_rate,
        mean_max_prob=mean_max_prob,
        uniform_deviation=uniform_deviation,
        ko_en_gap=ko_en_gap,
        pair_agreement=pair_agreement,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        cost_per_1k=cost_per_1k,
    )
