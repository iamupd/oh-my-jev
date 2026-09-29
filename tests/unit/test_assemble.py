"""Answer assembly: probability sums, argmax tie-break, score arithmetic, legend, zero-total.
# REQ-013
"""

from __future__ import annotations

from omj.gateway.assemble import assemble_answer, softmax
from omj.gateway.schema import Question


def _choice_question(n: int) -> Question:
    return Question(
        type="choice",
        instructions="pick one",
        criteria={f"opt{i}": f"Option {i}" for i in range(n)},
    )


def _score_question(n: int) -> Question:
    return Question(
        type="score",
        instructions="rate it",
        criteria=[f"level {i}" for i in range(n)],
    )


def _noul_question() -> Question:
    return Question(type="noul", instructions="yes or no?")


def test_softmax_sums_to_one() -> None:
    # REQ-013
    probs = softmax([1.0, 2.0, 3.0])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(p > 0 for p in probs)


def test_choice_probabilities_sum_to_one_within_tolerance() -> None:
    # REQ-013
    q = _choice_question(3)
    keys = ["opt0", "opt1", "opt2"]
    answer = assemble_answer(q, keys, [1.0, 2.0, 3.0])
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-6


def test_choice_argmax_tie_breaks_to_earliest_key() -> None:
    # REQ-013
    q = _choice_question(3)
    keys = ["opt0", "opt1", "opt2"]
    answer = assemble_answer(q, keys, [1.0, 1.0, 0.5])
    assert answer["choice"] == "opt0"


def test_score_arithmetic_matches_sum_index_times_probability() -> None:
    # REQ-013
    q = _score_question(4)
    keys = ["0", "1", "2", "3"]
    probs = [0.1, 0.2, 0.3, 0.4]
    answer = assemble_answer(q, keys, probs)
    expected = sum(i * p for i, p in enumerate(probs))
    assert abs(answer["score"] - expected) < 1e-6


def test_score_legend_keys_match_index_strings() -> None:
    # REQ-013
    q = _score_question(3)
    keys = ["0", "1", "2"]
    answer = assemble_answer(q, keys, [1.0, 1.0, 1.0])
    assert set(answer["legend"].keys()) == {"0", "1", "2"}
    assert answer["legend"]["0"] == "level 0"


def test_zero_total_probs_fall_back_to_uniform() -> None:
    # REQ-013
    q = _choice_question(4)
    keys = ["opt0", "opt1", "opt2", "opt3"]
    answer = assemble_answer(q, keys, [0.0, 0.0, 0.0, 0.0])
    for p in answer["probabilities"].values():
        assert abs(p - 0.25) < 1e-9


def test_noul_answer_uses_yes_key_as_probability() -> None:
    # REQ-013
    q = _noul_question()
    answer = assemble_answer(q, ["yes", "no"], [3.0, 1.0])
    assert answer["type"] == "noul"
    assert abs(answer["noul"] - 0.75) < 1e-9
