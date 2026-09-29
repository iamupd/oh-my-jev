"""Answer assembly (noul/choice/score) and confidence metrics.

The `score_confidence`, `choice_confidence`, and `_normalize` functions below are
ported verbatim (same arithmetic) from the official TypeSafe reference implementation:
  typesafe-ai/system-one-adapter-python
  src/system_one_adapter/_utils/confidence_metrics.py
  commit fb52b1030b7fc1f4f1cf39910afa5da54f9835e3
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from omj.gateway.schema import Question, score_legend


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


def score_confidence(probs: list[float]) -> float:
    """Measure score concentration around its modal score."""
    if len(probs) == 1:
        return 1.0

    normalized_probs = _normalize(probs)
    mode_index = max(range(len(normalized_probs)), key=normalized_probs.__getitem__)
    distance_from_mode = sum(
        probability * abs(index - mode_index) for index, probability in enumerate(normalized_probs)
    )
    uniform_center = (len(normalized_probs) - 1) / 2
    uniform_mean_absolute_deviation = sum(
        abs(index - uniform_center) for index in range(len(normalized_probs))
    ) / len(normalized_probs)
    return max(0.0, 1.0 - distance_from_mode / uniform_mean_absolute_deviation)


def choice_confidence(probs: list[float]) -> float:
    """Scale peak choice probability from uniform to certainty."""
    if len(probs) == 1:
        return 1.0

    normalized_probs = _normalize(probs)
    uniform_probability = 1.0 / len(normalized_probs)
    return (max(normalized_probs) - uniform_probability) / (1.0 - uniform_probability)


def _normalize(probs: list[float]) -> list[float]:
    """Normalize confidence inputs, using uniform probabilities for zero totals."""
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [probability / total for probability in probs]


def assemble_answer(q: Question, keys: list[str], probs: list[float]) -> dict:
    total = sum(probs)
    if total == 0:
        normalized = [1.0 / len(probs)] * len(probs)
    else:
        normalized = [p / total for p in probs]

    if q.type == "noul":
        key_to_prob = dict(zip(keys, normalized))
        return {"type": "noul", "noul": key_to_prob.get("yes", 0.0)}

    if q.type == "choice":
        probabilities = dict(zip(keys, normalized))
        best_key = keys[0]
        best_prob = normalized[0]
        for key, prob in zip(keys[1:], normalized[1:]):
            if prob > best_prob:
                best_key = key
                best_prob = prob
        return {
            "type": "choice",
            "choice": best_key,
            "probabilities": probabilities,
            "confidence": choice_confidence(normalized),
        }

    if q.type == "score":
        probabilities = dict(zip(keys, normalized))
        score_value = sum(i * p for i, p in enumerate(normalized))
        return {
            "type": "score",
            "score": score_value,
            "legend": score_legend(q),
            "probabilities": probabilities,
            "confidence": score_confidence(normalized),
        }

    raise ValueError(f"unsupported question type: {q.type}")
