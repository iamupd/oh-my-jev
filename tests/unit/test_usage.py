"""Usage accounting: zero mode returns 0 output tokens, typesafe mode approximates via JSON length.
# REQ-016
"""

from __future__ import annotations

import json
import math

from omj.gateway.usage import compute_usage


def test_zero_mode_returns_zero_output_tokens() -> None:
    # REQ-016
    answers = {"q1": {"type": "noul", "noul": 0.5}}
    usage = compute_usage(input_tokens=42, answers=answers, mode="zero")
    assert usage == {"input_tokens": 42, "output_tokens": 0}


def test_typesafe_mode_approximates_tokens_from_serialized_json() -> None:
    # REQ-016
    answers = {
        "q1": {
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.6, "b": 0.4},
            "confidence": 0.2,
        }
    }
    usage = compute_usage(input_tokens=10, answers=answers, mode="typesafe")
    serialized_len = len(json.dumps(answers, separators=(",", ":")))
    assert usage["output_tokens"] == math.ceil(serialized_len / 4)
    assert usage["input_tokens"] == 10
