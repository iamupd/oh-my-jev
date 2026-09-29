"""Per-suite run details for the bench report: breakdowns, reliability bins and run facts.

`Metrics` holds the headline numbers every report shares; this module adds the
context a reader needs to trust them: how many items were scored, a 95%
confidence interval for accuracy, accuracy by question type and by item tag,
the reliability table behind ECE, and the suite's wall time and failures.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from omj.bench.metrics import STRICT_SUM_TOLERANCE, is_valid_probability_vector
from omj.bench.types import ResultRow, is_correct

RELIABILITY_BINS = 10
MIN_TAG_ITEMS = 5
MAX_TAGS = 25


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; None when n == 0."""
    if n == 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _scored(rows: list[ResultRow]) -> list[ResultRow]:
    return [
        r
        for r in rows
        if r.error is None
        and r.expected is not None
        and is_valid_probability_vector(r.keys, r.probs, STRICT_SUM_TOLERANCE)
    ]


def _group_stats(rows: list[ResultRow]) -> dict[str, Any]:
    corrects = [bool(is_correct(r)) for r in rows]
    confidences = [max(r.probs) for r in rows]  # type: ignore[arg-type]
    return {
        "n": len(rows),
        "accuracy": sum(corrects) / len(rows),
        "mean_confidence": sum(confidences) / len(rows),
    }


def _reliability(rows: list[ResultRow], n_bins: int = RELIABILITY_BINS) -> list[dict[str, Any]]:
    buckets: list[list[ResultRow]] = [[] for _ in range(n_bins)]
    for row in rows:
        confidence = max(row.probs)  # type: ignore[arg-type]
        buckets[min(int(confidence * n_bins), n_bins - 1)].append(row)
    table = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        stats = _group_stats(bucket)
        table.append(
            {
                "range": [index / n_bins, (index + 1) / n_bins],
                "n": stats["n"],
                "mean_confidence": stats["mean_confidence"],
                "accuracy": stats["accuracy"],
            }
        )
    return table


def suite_details(
    rows: list[ResultRow],
    *,
    n_items: int,
    wall_seconds: float,
    n_errors: int,
    stopped_early: bool,
    stop_reason: str,
) -> dict[str, Any]:
    """Everything report.md shows for one suite beyond its `Metrics`."""
    scored = _scored(rows)
    n_correct = sum(1 for r in scored if is_correct(r))
    ci = wilson_interval(n_correct, len(scored))
    attempted_items = len({r.item_id for r in rows})

    by_kind: dict[str, list[ResultRow]] = defaultdict(list)
    by_tag: dict[str, list[ResultRow]] = defaultdict(list)
    for row in scored:
        by_kind[row.kind].append(row)
        for tag in set(row.tags):
            by_tag[tag].append(row)
    # A tag on every scored row (e.g. the suite's own name) says nothing.
    tags = {
        tag: group
        for tag, group in by_tag.items()
        if MIN_TAG_ITEMS <= len(group) < len(scored)
    }
    top_tags = sorted(tags.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:MAX_TAGS]

    return {
        "n_items": n_items,
        "n_attempted_items": attempted_items,
        "n_scored": len(scored),
        "n_correct": n_correct,
        "accuracy_ci95": list(ci) if ci else None,
        "n_errors": n_errors,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "wall_seconds": wall_seconds,
        "items_per_second": (attempted_items / wall_seconds) if wall_seconds > 0 else None,
        "by_kind": {kind: _group_stats(group) for kind, group in sorted(by_kind.items())},
        "by_tag": {tag: _group_stats(group) for tag, group in top_tags},
        "reliability": _reliability(scored),
    }
