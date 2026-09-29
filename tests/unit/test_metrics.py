"""Hand-computed expectations for every metric in design.md 4.3."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omj.bench.metrics import _ko_en_pair_metrics, compute_metrics, coverage_at_risk, top_label_ece
from omj.bench.types import ResultRow, is_correct, predicted_key


def make_row(
    *,
    item_id: str = "item-1",
    suite: str = "synthetic",
    qid: str = "q1",
    kind: str = "choice",
    keys: list[str],
    probs: list[float] | None,
    expected: str | None = None,
    latency_ms: float = 10.0,
    cost_usd: float | None = None,
    backend: str = "mock",
    model: str = "mock-model",
    tags: list[str] | None = None,
    expected_shape: str | None = None,
    pair_id: str | None = None,
    error: str | None = None,
) -> ResultRow:
    return ResultRow(
        item_id=item_id,
        suite=suite,
        qid=qid,
        kind=kind,
        keys=keys,
        probs=probs,
        logits=None,
        expected=expected,
        correct=None,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        backend=backend,
        model=model,
        tags=tags or [],
        expected_shape=expected_shape,
        pair_id=pair_id,
        error=error,
    )


# REQ-036: predicted_key / is_correct tie-break and threshold rules that accuracy relies on.
def test_predicted_key_choice_ties_pick_earliest_index() -> None:
    row = make_row(keys=["A", "B", "C"], probs=[0.3, 0.3, 0.4], expected="C")
    assert predicted_key(row) == "C"

    tie_row = make_row(keys=["A", "B", "C"], probs=[0.4, 0.4, 0.2], expected="B")
    assert predicted_key(tie_row) == "A"
    assert is_correct(tie_row) is False


# REQ-036: noul uses a 0.5 threshold on p_yes, not argmax parity, and ties go to "no".
def test_predicted_key_noul_threshold() -> None:
    yes_row = make_row(kind="noul", keys=["yes", "no"], probs=[0.51, 0.49], expected="yes")
    assert predicted_key(yes_row) == "yes"

    tie_row = make_row(kind="noul", keys=["yes", "no"], probs=[0.5, 0.5], expected="no")
    assert predicted_key(tie_row) == "no"
    assert is_correct(tie_row) is True


# REQ-036: accuracy and chance_corrected_accuracy, hand-computed for k=4 (choice).
def test_accuracy_and_chance_corrected_k4() -> None:
    rows = [
        make_row(keys=["A", "B", "C", "D"], probs=[0.7, 0.1, 0.1, 0.1], expected="A"),  # correct
        make_row(keys=["A", "B", "C", "D"], probs=[0.1, 0.6, 0.2, 0.1], expected="B"),  # correct
        make_row(keys=["A", "B", "C", "D"], probs=[0.25, 0.25, 0.25, 0.25], expected="C"),  # wrong (tie->A)
        make_row(keys=["A", "B", "C", "D"], probs=[0.1, 0.1, 0.1, 0.7], expected="D"),  # correct
    ]
    m = compute_metrics(rows)
    assert m.accuracy == pytest.approx(0.75)
    # (1-1/4)/(3/4)=1.0 thrice, (0-1/4)/(3/4)=-1/3 once -> mean = (3 - 1/3)/4 = 2/3
    assert m.chance_corrected_accuracy == pytest.approx(2.0 / 3.0)


# REQ-036: same metrics for k=2 (noul), where the chance term is 1/2.
def test_accuracy_and_chance_corrected_k2() -> None:
    rows = [
        make_row(kind="noul", keys=["yes", "no"], probs=[0.8, 0.2], expected="yes"),  # correct
        make_row(kind="noul", keys=["yes", "no"], probs=[0.3, 0.7], expected="yes"),  # wrong
        make_row(kind="noul", keys=["yes", "no"], probs=[0.5, 0.5], expected="no"),  # correct (tie->no)
        make_row(kind="noul", keys=["yes", "no"], probs=[0.6, 0.4], expected="no"),  # wrong
    ]
    m = compute_metrics(rows)
    assert m.accuracy == pytest.approx(0.5)
    assert m.chance_corrected_accuracy == pytest.approx(0.0)


# REQ-036: hard_accuracy only over rows tagged "hard"; None when no such rows exist.
def test_hard_accuracy_subset_and_none_when_absent() -> None:
    rows = [
        make_row(keys=["A", "B"], probs=[0.9, 0.1], expected="A", tags=["hard"]),  # correct, hard
        make_row(keys=["A", "B"], probs=[0.1, 0.9], expected="A", tags=["hard"]),  # wrong, hard
        make_row(keys=["A", "B"], probs=[0.9, 0.1], expected="A"),  # correct, not hard
    ]
    m = compute_metrics(rows)
    assert m.accuracy == pytest.approx(2.0 / 3.0)
    assert m.hard_accuracy == pytest.approx(0.5)

    no_hard = compute_metrics(rows[2:])
    assert no_hard.hard_accuracy is None


# REQ-036: top_label_ece hand-computed directly, showing 15 vs 10 bins group points differently.
def test_top_label_ece_helper_15_vs_10_bins() -> None:
    confidences = [0.62, 0.68, 0.95, 0.95]
    corrects = [True, False, True, False]
    assert top_label_ece(confidences, corrects, 15) == pytest.approx(0.49)
    assert top_label_ece(confidences, corrects, 10) == pytest.approx(0.3)


# REQ-036: same fixture wired through compute_metrics to check ece/ece_jevbench plumbing.
def test_ece_and_ece_jevbench_via_compute_metrics() -> None:
    rows = [
        make_row(keys=["A", "B"], probs=[0.62, 0.38], expected="A"),  # correct, conf 0.62
        make_row(keys=["A", "B"], probs=[0.68, 0.32], expected="B"),  # wrong, conf 0.68
        make_row(keys=["A", "B"], probs=[0.95, 0.05], expected="A"),  # correct, conf 0.95
        make_row(keys=["A", "B"], probs=[0.95, 0.05], expected="B"),  # wrong, conf 0.95
    ]
    m = compute_metrics(rows)
    assert m.ece == pytest.approx(0.49)
    assert m.ece_jevbench == pytest.approx(0.3)


# REQ-036: brier and nll exact values for a 3-way choice.
def test_brier_and_nll_exact_values() -> None:
    rows = [
        make_row(keys=["A", "B", "C"], probs=[0.5, 0.3, 0.2], expected="A"),
        make_row(keys=["A", "B", "C"], probs=[0.2, 0.5, 0.3], expected="B"),
        make_row(keys=["A", "B", "C"], probs=[0.1, 0.1, 0.8], expected="C"),
    ]
    m = compute_metrics(rows)
    expected_brier = (0.38 + 0.38 + 0.06) / 3.0
    expected_nll = (-math.log(0.5) - math.log(0.5) - math.log(0.8)) / 3.0
    assert m.brier == pytest.approx(expected_brier)
    assert m.nll == pytest.approx(expected_nll)


# REQ-036: coverage_at_risk helper picks the largest satisfying prefix, not a monotone one.
def test_coverage_at_risk_largest_satisfying_prefix() -> None:
    # 20 rows sorted by confidence descending; wrong at position 5 and 15 (1-indexed).
    corrects = [i not in (5, 15) for i in range(1, 21)]
    assert coverage_at_risk(corrects) == pytest.approx(0.2)


# REQ-036: coverage_at_risk is 0.0 when the very first (highest-confidence) row is wrong.
def test_coverage_at_risk_zero_when_first_row_wrong() -> None:
    assert coverage_at_risk([False, True, True]) == pytest.approx(0.0)


# REQ-036: coverage_at_risk wired through compute_metrics on a matching ResultRow fixture.
def test_coverage_at_risk_via_compute_metrics() -> None:
    rows = []
    for i in range(20):
        conf = 0.99 - i * 0.01
        wrong = (i + 1) in (5, 15)
        expected = "B" if wrong else "A"
        rows.append(make_row(keys=["A", "B"], probs=[conf, 1 - conf], expected=expected))
    m = compute_metrics(rows)
    assert m.coverage_at_risk == pytest.approx(0.2)


# REQ-036: order_shift_max is the max (max-min) probability spread per key across shuffled orders.
def test_order_shift_max_across_groups() -> None:
    group1 = [
        make_row(item_id="i1", qid="q1", keys=["A", "B", "C"], probs=[0.5, 0.3, 0.2]),
        make_row(item_id="i1", qid="q1", keys=["B", "A", "C"], probs=[0.25, 0.55, 0.2]),
        make_row(item_id="i1", qid="q1", keys=["C", "A", "B"], probs=[0.15, 0.65, 0.2]),
    ]
    group2 = [
        make_row(item_id="i2", qid="q1", keys=["A", "B"], probs=[0.5, 0.5]),
        make_row(item_id="i2", qid="q1", keys=["A", "B"], probs=[0.5, 0.5]),
    ]
    order_groups = {"i1-q1": group1, "i2-q1": group2}
    m = compute_metrics(group1 + group2, order_groups=order_groups)
    assert m.order_shift_max == pytest.approx(0.15)


# REQ-036: order_shift_max is None when no order groups are supplied.
def test_order_shift_max_none_when_no_groups() -> None:
    rows = [make_row(keys=["A", "B"], probs=[0.5, 0.5])]
    m = compute_metrics(rows)
    assert m.order_shift_max is None

    m_empty_groups = compute_metrics(rows, order_groups={})
    assert m_empty_groups.order_shift_max is None


# REQ-037: underdetermined set yields overconfidence_rate/mean_max_prob but no accuracy metrics.
def test_underdetermined_overconfidence_and_mean_max_prob() -> None:
    rows = [
        make_row(keys=["A", "B", "C", "D"], probs=[0.4, 0.3, 0.2, 0.1], expected_shape="uniform"),
        make_row(keys=["A", "B"], probs=[0.6, 0.4], expected_shape="uniform"),
        make_row(keys=["A", "B"], probs=[0.95, 0.05], expected_shape="unknown"),
        make_row(keys=["A", "B", "C"], probs=[0.5, 0.3, 0.2], expected_shape="conflict"),
    ]
    m = compute_metrics(rows)
    assert m.overconfidence_rate == pytest.approx(0.25)
    assert m.mean_max_prob == pytest.approx(0.6125)
    assert m.accuracy is None
    assert m.ece is None


# REQ-037: uniform_deviation only over expected_shape=="uniform" rows; None when absent.
def test_uniform_deviation() -> None:
    rows = [
        make_row(keys=["A", "B", "C", "D"], probs=[0.4, 0.3, 0.2, 0.1], expected_shape="uniform"),
        make_row(keys=["A", "B"], probs=[0.6, 0.4], expected_shape="uniform"),
        make_row(keys=["A", "B"], probs=[0.95, 0.05], expected_shape="unknown"),
    ]
    m = compute_metrics(rows)
    assert m.uniform_deviation == pytest.approx((0.15 + 0.1) / 2.0)

    no_uniform = compute_metrics(rows[2:])
    assert no_uniform.uniform_deviation is None


# REQ-051: ko_en_gap (en accuracy - ko accuracy) and pair_agreement over shared pair_id/qid.
def test_ko_en_gap_and_pair_agreement() -> None:
    rows = [
        make_row(suite="massive-ko", pair_id="p1", qid="decision", keys=["A", "B"], probs=[0.6, 0.4], expected="A"),
        make_row(suite="massive-en", pair_id="p1", qid="decision", keys=["A", "B"], probs=[0.3, 0.7], expected="A"),
        make_row(suite="massive-ko", pair_id="p2", qid="decision", keys=["A", "B"], probs=[0.2, 0.8], expected="B"),
        make_row(suite="massive-en", pair_id="p2", qid="decision", keys=["A", "B"], probs=[0.9, 0.1], expected="B"),
        make_row(suite="massive-ko", pair_id="p3", qid="decision", keys=["A", "B"], probs=[0.7, 0.3], expected="A"),
        make_row(suite="massive-en", pair_id="p3", qid="decision", keys=["A", "B"], probs=[0.65, 0.35], expected="A"),
    ]
    m = compute_metrics(rows)
    # ko correct: p1 T, p2 T, p3 T -> 1.0 ; en correct: p1 F, p2 F, p3 T -> 1/3
    assert m.ko_en_gap == pytest.approx((1.0 / 3.0) - 1.0)
    # agreement: p1 A vs B (no), p2 B vs A (no), p3 A vs A (yes) -> 1/3
    assert m.pair_agreement == pytest.approx(1.0 / 3.0)


# REQ-051: an invalid-probability-vector row must be excluded from the ko/en
# pair just like it is excluded from accuracy -- pairing over `attempted`
# (rather than the valid-vector rows accuracy uses) let a malformed vector
# still drive ko_en_gap/pair_agreement.
def test_ko_en_pair_metrics_excludes_invalid_probability_vector_rows() -> None:
    rows = [
        # ko side: probs sum to 1.8, invalid under both strict and lenient tolerance.
        make_row(suite="massive-ko", pair_id="p1", qid="q1", keys=["A", "B"], probs=[0.9, 0.9], expected="A"),
        make_row(suite="massive-en", pair_id="p1", qid="q1", keys=["A", "B"], probs=[0.6, 0.4], expected="A"),
    ]
    m = compute_metrics(rows)
    assert m.ko_en_gap is None
    assert m.pair_agreement is None


# REQ-051: a pair where neither side has a predicted_key (no probability
# vector) must not count as "agreement" just because None == None.
def test_ko_en_pair_metrics_skips_pairs_where_either_side_has_no_prediction() -> None:
    unpredictable_ko = make_row(
        suite="massive-ko", pair_id="p1", qid="q1", keys=["A", "B"], probs=None, expected="A"
    )
    unpredictable_en = make_row(
        suite="massive-en", pair_id="p1", qid="q1", keys=["A", "B"], probs=None, expected="A"
    )
    # A real, disagreeing pair: ko predicts "A", en predicts "B".
    real_ko = make_row(
        suite="massive-ko", pair_id="p2", qid="q1", keys=["A", "B"], probs=[0.9, 0.1], expected="A"
    )
    real_en = make_row(
        suite="massive-en", pair_id="p2", qid="q1", keys=["A", "B"], probs=[0.1, 0.9], expected="A"
    )

    gap, agreement = _ko_en_pair_metrics([unpredictable_ko, unpredictable_en, real_ko, real_en])

    # Only p2 contributes a real prediction comparison, and it disagrees, so
    # agreement must be 0.0 -- not inflated to 0.5 by counting None == None
    # for the unpredictable p1 pair as agreement.
    assert agreement == pytest.approx(0.0)


# REQ-051: None when there is no massive-ko/massive-en pair overlap.
def test_ko_en_gap_none_when_no_pairs() -> None:
    rows = [
        make_row(suite="massive-ko", pair_id="p1", qid="decision", keys=["A", "B"], probs=[0.6, 0.4], expected="A"),
        make_row(suite="synthetic", pair_id="p1", qid="decision", keys=["A", "B"], probs=[0.6, 0.4], expected="A"),
    ]
    m = compute_metrics(rows)
    assert m.ko_en_gap is None
    assert m.pair_agreement is None


# REQ-036: latency percentiles via numpy over attempted rows only.
def test_latency_percentiles() -> None:
    rows = [make_row(keys=["A", "B"], probs=[0.5, 0.5], latency_ms=float(i)) for i in range(1, 101)]
    m = compute_metrics(rows)
    assert m.latency_p50_ms == pytest.approx(float(np.percentile(list(range(1, 101)), 50)))
    assert m.latency_p95_ms == pytest.approx(float(np.percentile(list(range(1, 101)), 95)))
    assert m.latency_p50_ms == pytest.approx(50.5)
    assert m.latency_p95_ms == pytest.approx(95.05)


# REQ-041: local-backend cost_per_1k = gpu_cost_per_hour * hours / n_attempted * 1000.
def test_cost_per_1k_local_gpu_branch() -> None:
    rows = [make_row(keys=["A", "B"], probs=[0.5, 0.5]) for _ in range(10)]
    m = compute_metrics(rows, gpu_cost_per_hour=2.0, total_wall_seconds=7200.0)
    assert m.cost_per_1k == pytest.approx(400.0)


# REQ-041: typesafe-backend cost_per_1k = input_tokens_total * price_per_mtok / 1e6 / n * 1000.
def test_cost_per_1k_typesafe_token_branch() -> None:
    rows = [make_row(keys=["A", "B"], probs=[0.5, 0.5]) for _ in range(10)]
    price_per_mtok = 0.042
    token_count = 10**6
    m = compute_metrics(rows, price_input_per_mtok=price_per_mtok, input_tokens_total=token_count)
    assert m.cost_per_1k == pytest.approx(4.2)


# REQ-041: cost_per_1k is None when no pricing inputs are supplied.
def test_cost_per_1k_none_when_unspecified() -> None:
    rows = [make_row(keys=["A", "B"], probs=[0.5, 0.5]) for _ in range(10)]
    m = compute_metrics(rows)
    assert m.cost_per_1k is None


# REQ-036: invalid probability vectors are excluded from accuracy but still counted in n
# and drive valid_vector_rate (strict) / valid_vector_rate_lenient.
def test_invalid_vectors_excluded_from_accuracy_but_counted_in_n() -> None:
    rows = [
        make_row(keys=["A", "B"], probs=[0.5, 0.5], expected="A"),  # valid, correct
        make_row(keys=["A", "B"], probs=[0.5, 0.49], expected="B"),  # sum off by 0.01: lenient-only
        make_row(keys=["A", "B"], probs=[0.5, 0.2], expected="A"),  # sum off by 0.3: invalid both
        make_row(keys=["A", "B"], probs=[0.3, 0.3, 0.4], expected="A"),  # key/prob length mismatch
        make_row(keys=["A", "B"], probs=[0.2, 0.8], expected="B"),  # valid, correct
    ]
    m = compute_metrics(rows)
    assert m.n == 5
    assert m.n_valid == 2
    assert m.n_unattempted == 0
    assert m.valid_vector_rate == pytest.approx(0.4)
    assert m.valid_vector_rate_lenient == pytest.approx(0.6)
    assert m.accuracy == pytest.approx(1.0)


# REQ-036: rows with `error` set are unattempted -- excluded from every metric, counted separately.
def test_unattempted_rows_excluded_and_counted() -> None:
    rows = [
        make_row(keys=["A", "B"], probs=[0.9, 0.1], expected="A"),  # correct
        make_row(keys=["A", "B"], probs=[0.1, 0.9], expected="A"),  # wrong
        make_row(keys=["A", "B"], probs=[0.9, 0.1], expected="A"),  # correct
        make_row(keys=["A", "B"], probs=None, expected=None, error="E_BACKEND: timeout"),
        make_row(keys=["A", "B"], probs=None, expected=None, error="E_NET: connection reset"),
    ]
    m = compute_metrics(rows)
    assert m.n == 5
    assert m.n_unattempted == 2
    assert m.n_valid == 3
    assert m.accuracy == pytest.approx(2.0 / 3.0)
    assert m.valid_vector_rate == pytest.approx(1.0)


# REQ-036: to_dict() exposes every Metrics field for serialization.
def test_metrics_to_dict_round_trip() -> None:
    rows = [make_row(keys=["A", "B"], probs=[0.5, 0.5], expected="A")]
    m = compute_metrics(rows)
    d = m.to_dict()
    assert d["n"] == 1
    assert d["accuracy"] == pytest.approx(1.0)
    assert "cost_per_1k" in d and d["cost_per_1k"] is None
