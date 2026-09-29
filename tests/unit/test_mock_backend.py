"""MockBackend determinism, the omj-smoke suite (30/30), and Korean tokenization.
# REQ-028
"""

from __future__ import annotations

import json
from pathlib import Path

from omj.backends.mock import MockBackend

SUITE_PATH = Path(__file__).resolve().parents[2] / "suites" / "omj-smoke.jsonl"


def _load_smoke_rows() -> list[dict]:
    rows = []
    with SUITE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _argmax_label(kind: str, keys: list[str], logits: list[float]) -> str:
    if kind == "noul":
        return "yes" if logits[0] > logits[1] else "no"
    best_index = 0
    best_value = logits[0]
    for index, value in enumerate(logits):
        if value > best_value:
            best_value = value
            best_index = index
    return keys[best_index]


def test_decide_is_deterministic() -> None:
    # REQ-028
    backend = MockBackend()
    state = "The app crashes every time I open the settings page."
    questions = {
        "q1": {
            "type": "choice",
            "instructions": "route",
            "criteria": {"billing": "invoice charges", "technical": "bug crash outage"},
        }
    }
    first = backend.decide(state, questions)
    second = backend.decide(state, questions)
    assert first["q1"].logits == second["q1"].logits
    assert first["q1"].keys == second["q1"].keys


def test_smoke_suite_30_of_30() -> None:
    # REQ-028, basis of REQ-050
    backend = MockBackend()
    rows = _load_smoke_rows()
    assert len(rows) == 30

    total = 0
    mismatches = []
    for row in rows:
        answers = backend.decide(row["state"], row["questions"])
        for qid, expected_label in row["expected"].items():
            answer = answers[qid]
            predicted = _argmax_label(answer.kind, answer.keys, answer.logits)
            total += 1
            if predicted != expected_label:
                mismatches.append((row["id"], qid, expected_label, predicted))

    assert not mismatches, f"{len(mismatches)}/{total} mismatches: {mismatches}"
    assert total == 30


def test_noul_question_without_criteria_key_returns_two_logit_raw_answer() -> None:
    # REQ-028: suites/underdetermined.jsonl rows (e.g. und-003 "will_convert") omit
    # `criteria` entirely for noul questions; decide() must fall back to the
    # documented defaults instead of raising KeyError.
    backend = MockBackend()
    state = "A customer signed up for the free trial today."
    questions = {
        "will_convert": {
            "type": "noul",
            "instructions": "Will this customer convert to a paid plan?",
        }
    }
    answers = backend.decide(state, questions)
    answer = answers["will_convert"]
    assert answer.kind == "noul"
    assert answer.keys == ["yes", "no"]
    assert answer.logits is not None
    assert len(answer.logits) == 2
    assert all(isinstance(value, float) for value in answer.logits)


def test_count_tokens_with_string_instructions() -> None:
    # REQ-028: count_tokens whitespace-splits the state plus every question's
    # instructions when instructions is a plain string.
    backend = MockBackend()
    state = "one two three"
    questions = {
        "q1": {"type": "noul", "instructions": "four five six", "criteria": {}},
    }
    assert backend.count_tokens(state, questions) == 6


def test_count_tokens_with_dict_instructions() -> None:
    # REQ-028: render_state() supports dict/list instructions too, so a
    # question whose "instructions" is a nested payload (not a plain string)
    # must not raise and must count its rendered tokens.
    backend = MockBackend()
    state = "one two three"
    questions = {
        "q1": {
            "type": "choice",
            "instructions": {"ko": "안내", "en": "guide"},
            "criteria": {"a": "desc"},
        },
    }
    count = backend.count_tokens(state, questions)
    assert isinstance(count, int)
    assert count > 3  # at least the 3 state tokens plus the rendered dict tokens


def test_korean_state_and_criteria_tokenize_and_score() -> None:
    # REQ-028: Korean text must tokenize via \\w+ (Unicode letters), not just ASCII words.
    backend = MockBackend()
    state = "고객이 중복 결제로 환불을 요청했습니다."
    questions = {
        "refund": {
            "type": "noul",
            "instructions": "고객이 환불을 요청하는가?",
            "criteria": {
                "true": "고객이 환불이나 결제 취소를 명확히 요청함",
                "false": "환불 요청이 없고 단순 문의임",
            },
        }
    }
    answers = backend.decide(state, questions)
    answer = answers["refund"]
    assert answer.keys == ["yes", "no"]
    assert answer.logits is not None
    assert answer.logits[0] > answer.logits[1]
