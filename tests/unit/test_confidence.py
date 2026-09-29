"""Confidence formula parity with typesafe-ai/system-one-adapter-python reference.
# REQ-014
"""

from __future__ import annotations

import pytest

from omj.gateway.assemble import choice_confidence as omj_choice_confidence
from omj.gateway.assemble import score_confidence as omj_score_confidence

# Reference implementation inlined verbatim for parity testing.
# Source: typesafe-ai/system-one-adapter-python
#   src/system_one_adapter/_utils/confidence_metrics.py
#   commit fb52b1030b7fc1f4f1cf39910afa5da54f9835e3


def _ref_score_confidence(probs: list[float]) -> float:
    """Measure score concentration around its modal score."""
    if len(probs) == 1:
        return 1.0

    normalized_probs = _ref_normalize(probs)
    mode_index = max(range(len(normalized_probs)), key=normalized_probs.__getitem__)
    distance_from_mode = sum(
        probability * abs(index - mode_index) for index, probability in enumerate(normalized_probs)
    )
    uniform_center = (len(normalized_probs) - 1) / 2
    uniform_mean_absolute_deviation = sum(
        abs(index - uniform_center) for index in range(len(normalized_probs))
    ) / len(normalized_probs)
    return max(0.0, 1.0 - distance_from_mode / uniform_mean_absolute_deviation)


def _ref_choice_confidence(probs: list[float]) -> float:
    """Scale peak choice probability from uniform to certainty."""
    if len(probs) == 1:
        return 1.0

    normalized_probs = _ref_normalize(probs)
    uniform_probability = 1.0 / len(normalized_probs)
    return (max(normalized_probs) - uniform_probability) / (1.0 - uniform_probability)


def _ref_normalize(probs: list[float]) -> list[float]:
    """Normalize confidence inputs, using uniform probabilities for zero totals."""
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [probability / total for probability in probs]


DISTRIBUTIONS = [
    ("uniform_3", [1 / 3, 1 / 3, 1 / 3]),
    ("one_hot_4", [0.0, 0.0, 1.0, 0.0]),
    ("two_way", [0.7, 0.3]),
    ("five_way_ordered", [0.05, 0.1, 0.2, 0.3, 0.35]),
    ("zero_total", [0.0, 0.0, 0.0]),
    ("single_option", [1.0]),
]


@pytest.mark.parametrize("name,probs", DISTRIBUTIONS, ids=[d[0] for d in DISTRIBUTIONS])
def test_choice_confidence_matches_reference(name: str, probs: list[float]) -> None:
    # REQ-014
    assert omj_choice_confidence(probs) == pytest.approx(_ref_choice_confidence(probs))


@pytest.mark.parametrize("name,probs", DISTRIBUTIONS, ids=[d[0] for d in DISTRIBUTIONS])
def test_score_confidence_matches_reference(name: str, probs: list[float]) -> None:
    # REQ-014
    assert omj_score_confidence(probs) == pytest.approx(_ref_score_confidence(probs))
