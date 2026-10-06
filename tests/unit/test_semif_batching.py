"""semif batching under a token budget, and recovery from device out-of-memory errors.

A batch is padded to its longest prompt, so sixteen long prompts can need far more memory than sixteen short ones.
Batches are therefore capped by padded tokens as well as by count, and a batch that still runs out of memory is
retried in halves; a single prompt that does not fit is refused as a capacity limit instead of failing the server.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from omj.backends.semif import (  # noqa: E402
    forward_with_oom_split,
    token_budget_batches,
)
from omj.errors import ErrorCode, OmjError  # noqa: E402


def test_short_prompts_are_batched_by_count() -> None:
    assert token_budget_batches([10] * 20, max_batch=16, max_tokens=1000) == [list(range(16)), [16, 17, 18, 19]]


def test_long_prompts_are_batched_by_padded_tokens() -> None:
    assert token_budget_batches([300] * 5, max_batch=16, max_tokens=1000) == [[0, 1, 2], [3, 4]]


def test_a_longer_prompt_closes_the_batch_when_padding_would_exceed_the_budget() -> None:
    assert token_budget_batches([100, 600], max_batch=16, max_tokens=1000) == [[0], [1]]


def test_a_prompt_over_the_budget_runs_alone() -> None:
    assert token_budget_batches([5000, 10, 10], max_batch=16, max_tokens=1000) == [[0], [1, 2]]


def test_batches_keep_order_and_cover_every_prompt() -> None:
    lengths = [50, 900, 10, 10, 400, 400, 400, 3000, 5]
    batches = token_budget_batches(lengths, max_batch=4, max_tokens=1000)
    assert [index for batch in batches for index in batch] == list(range(len(lengths)))
    assert token_budget_batches([], max_batch=4, max_tokens=1000) == []


class FakeOOM(RuntimeError):
    pass


def test_out_of_memory_batch_is_retried_in_halves() -> None:
    calls: list[list[str]] = []
    releases: list[int] = []

    def forward(chunk: list[str]) -> dict[str, str]:
        calls.append(list(chunk))
        if len(chunk) > 1:
            raise FakeOOM("out of memory")
        return {item: item.upper() for item in chunk}

    answers = forward_with_oom_split(
        ["a", "b", "c", "d"], forward, oom_errors=(FakeOOM,), release=lambda: releases.append(1)
    )

    assert answers == {"a": "A", "b": "B", "c": "C", "d": "D"}
    assert calls == [["a", "b", "c", "d"], ["a", "b"], ["a"], ["b"], ["c", "d"], ["c"], ["d"]]
    assert len(releases) == 3


def test_a_single_prompt_that_does_not_fit_is_refused_as_a_capacity_limit() -> None:
    def forward(chunk: list[str]) -> dict[str, str]:
        raise FakeOOM("out of memory")

    with pytest.raises(OmjError) as exc_info:
        forward_with_oom_split(["only"], forward, oom_errors=(FakeOOM,), release=lambda: None)
    assert exc_info.value.code == ErrorCode.E_BACKEND
    assert "maximum context length" in exc_info.value.message


def test_other_errors_are_not_retried() -> None:
    def forward(chunk: list[str]) -> dict[str, str]:
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        forward_with_oom_split(["a", "b"], forward, oom_errors=(FakeOOM,), release=lambda: None)
