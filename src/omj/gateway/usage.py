"""Usage accounting for SystemOneResponse.usage (REQ-016)."""

from __future__ import annotations

import json
import math
from typing import Literal


def compute_usage(input_tokens: int, answers: dict, mode: Literal["zero", "typesafe"]) -> dict:
    if mode == "zero":
        output_tokens = 0
    elif mode == "typesafe":
        serialized = json.dumps(answers, separators=(",", ":"))
        output_tokens = math.ceil(len(serialized) / 4)
    else:
        raise ValueError(f"unsupported usage mode: {mode}")

    return {"input_tokens": input_tokens, "output_tokens": output_tokens}
