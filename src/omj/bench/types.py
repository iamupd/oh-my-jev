"""Result rows produced by bench runners and the aggregate Metrics report (design.md 3, 4.3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class ResultRow:
    item_id: str
    suite: str
    qid: str
    kind: Literal["noul", "choice", "score"]
    keys: list[str]
    probs: list[float] | None
    logits: list[float] | None
    expected: str | None
    correct: bool | None
    latency_ms: float
    cost_usd: float | None
    backend: str
    model: str
    tags: list[str] = field(default_factory=list)
    expected_shape: str | None = None
    pair_id: str | None = None
    error: str | None = None


def predicted_key(row: ResultRow) -> str | None:
    """Return the predicted answer key for a row, or None if it has no probability vector."""
    if row.probs is None or not row.keys or len(row.probs) != len(row.keys):
        return None

    if row.kind == "noul":
        try:
            yes_index = row.keys.index("yes")
        except ValueError:
            yes_index = 0
        p_yes = row.probs[yes_index]
        return "yes" if p_yes > 0.5 else "no"

    best_index = 0
    best_prob = row.probs[0]
    for i in range(1, len(row.probs)):
        # strict '>' keeps the earliest index on ties, matching the stable-sort tie-break rule.
        if row.probs[i] > best_prob:
            best_prob = row.probs[i]
            best_index = i
    return row.keys[best_index]


def is_correct(row: ResultRow) -> bool | None:
    """Return whether the row's prediction matches its expected key, or None if unscorable."""
    if row.expected is None:
        return None
    key = predicted_key(row)
    if key is None:
        return None
    return key == row.expected


@dataclass
class Metrics:
    n: int
    n_valid: int
    n_unattempted: int
    accuracy: float | None = None
    chance_corrected_accuracy: float | None = None
    hard_accuracy: float | None = None
    ece: float | None = None
    ece_jevbench: float | None = None
    brier: float | None = None
    nll: float | None = None
    coverage_at_risk: float | None = None
    valid_vector_rate: float | None = None
    valid_vector_rate_lenient: float | None = None
    order_shift_max: float | None = None
    overconfidence_rate: float | None = None
    mean_max_prob: float | None = None
    uniform_deviation: float | None = None
    ko_en_gap: float | None = None
    pair_agreement: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    cost_per_1k: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)
