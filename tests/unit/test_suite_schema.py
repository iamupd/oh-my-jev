"""DecisionItem schema validation and the two synthetic suites. # REQ-034"""

from __future__ import annotations

from dataclasses import replace

import pytest

from omj.bench.suites import load_suite
from omj.bench.suites.base import DecisionItem, validate_item
from omj.bench.suites.local import SUITES_DIR, load_local
from omj.errors import ErrorCode, OmjError

VALID_SHAPES = {"uniform", "unknown", "conflict"}


def _item(**overrides: object) -> DecisionItem:
    base = DecisionItem(
        id="x-1",
        suite="unit",
        state="some state",
        questions={
            "q": {
                "type": "noul",
                "instructions": "Is it so?",
                "criteria": {"true": "yes it is", "false": "no it is not"},
            }
        },
        expected={"q": "yes"},
        tags=["unit"],
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_suites_dir_points_at_repo_suites() -> None:
    # REQ-034
    assert SUITES_DIR.is_dir()
    assert (SUITES_DIR / "omj-smoke.jsonl").is_file()
    assert (SUITES_DIR / "underdetermined.jsonl").is_file()


def test_smoke_suite_loads_thirty_items_with_expected_labels() -> None:
    # REQ-034
    items = load_suite("omj-smoke")
    assert len(items) == 30
    assert {item.suite for item in items} == {"omj-smoke"}
    for item in items:
        assert item.id
        assert item.state
        assert item.questions
        assert item.tags
        assert item.expected is not None
        assert set(item.expected) <= set(item.questions)
        assert item.expected_shape is None


def test_underdetermined_suite_loads_twenty_items_with_expected_shape() -> None:
    # REQ-034
    items = load_suite("underdetermined")
    assert len(items) == 20
    assert {item.suite for item in items} == {"underdetermined"}
    for item in items:
        assert item.expected is None
        assert item.expected_shape in VALID_SHAPES
    assert {item.expected_shape for item in items} == VALID_SHAPES


def test_both_synthetic_suites_cover_all_three_question_types() -> None:
    # REQ-034
    for name in ("omj-smoke", "underdetermined"):
        kinds = {q["type"] for item in load_suite(name) for q in item.questions.values()}
        assert kinds == {"noul", "choice", "score"}


def test_unknown_suite_name_is_a_schema_error() -> None:
    # REQ-034
    with pytest.raises(OmjError) as excinfo:
        load_suite("no-such-suite")
    assert excinfo.value.code is ErrorCode.E_SCHEMA


def test_local_loader_reports_the_offending_line(tmp_path) -> None:
    # REQ-034
    path = tmp_path / "broken.jsonl"
    good = '{"id": "a", "tags": [], "state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "criteria": {}}}}'
    bad = '{"id": "", "tags": [], "state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "criteria": {}}}}'
    path.write_text(good + "\n" + bad + "\n", encoding="utf-8")
    with pytest.raises(OmjError) as excinfo:
        load_local(path, "broken")
    assert excinfo.value.code is ErrorCode.E_SCHEMA
    assert "line 2" in excinfo.value.message


def test_local_loader_rejects_malformed_json(tmp_path) -> None:
    # REQ-034
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(OmjError) as excinfo:
        load_local(path, "bad")
    assert excinfo.value.code is ErrorCode.E_SCHEMA


def test_validate_item_accepts_each_question_kind() -> None:
    # REQ-034
    validate_item(_item())
    validate_item(
        _item(
            questions={"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}}},
            expected={"q": "b"},
        )
    )
    validate_item(
        _item(
            questions={"q": {"type": "score", "instructions": "i", "criteria": ["low", "mid", "high"]}},
            expected={"q": "2"},
        )
    )
    validate_item(_item(expected=None, expected_shape="conflict"))


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("empty id", {"id": ""}),
        (
            "unknown question type",
            {"questions": {"q": {"type": "ranking", "instructions": "i", "criteria": {}}}, "expected": None},
        ),
        (
            "choice criteria not a dict",
            {"questions": {"q": {"type": "choice", "instructions": "i", "criteria": ["a", "b"]}}, "expected": None},
        ),
        (
            "choice criteria with one key",
            {"questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A"}}}, "expected": None},
        ),
        (
            "score criteria not a list",
            {"questions": {"q": {"type": "score", "instructions": "i", "criteria": {"0": "a"}}}, "expected": None},
        ),
        (
            "score criteria too short",
            {"questions": {"q": {"type": "score", "instructions": "i", "criteria": ["only"]}}, "expected": None},
        ),
        (
            "score criteria too long",
            {
                "questions": {"q": {"type": "score", "instructions": "i", "criteria": [str(i) for i in range(11)]}},
                "expected": None,
            },
        ),
        ("noul expected not an answer key", {"expected": {"q": "maybe"}}),
        (
            "choice expected not an answer key",
            {"questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}}}, "expected": {"q": "c"}},
        ),
        (
            "score expected out of range",
            {"questions": {"q": {"type": "score", "instructions": "i", "criteria": ["a", "b", "c"]}}, "expected": {"q": "3"}},
        ),
        ("expected for an unknown question", {"expected": {"nope": "yes"}}),
        ("no questions at all", {"questions": {}, "expected": None}),
        ("bad expected_shape", {"expected": None, "expected_shape": "bimodal"}),
    ],
)
def test_validate_item_rejects_bad_rows(label: str, overrides: dict) -> None:
    # REQ-034
    with pytest.raises(OmjError) as excinfo:
        validate_item(_item(**overrides))
    assert excinfo.value.code is ErrorCode.E_SCHEMA, label
    assert excinfo.value.message, label
