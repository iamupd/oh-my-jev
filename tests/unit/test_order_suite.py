"""Tests for the order suite: shuffle criteria key order and measure variability."""
# REQ-035

from __future__ import annotations

from pathlib import Path

import pytest

from omj.bench.suites.base import DecisionItem
from omj.bench.suites.order import make_order_suite


@pytest.fixture
def sample_item_with_choice():
    """One item with a 4-option choice question and a noul question."""
    return DecisionItem(
        id="test-item-001",
        suite="smoke",
        state={"text": "sample state"},
        questions={
            "quality": {
                "type": "choice",
                "instructions": "Rate the quality",
                "criteria": {
                    "excellent": "Excellent quality",
                    "good": "Good quality",
                    "fair": "Fair quality",
                    "poor": "Poor quality",
                },
            },
            "helpful": {
                "type": "noul",
                "instructions": "Is it helpful?",
            },
        },
        expected={"quality": "good", "helpful": "yes"},
        tags=["test"],
    )


def test_order_suite_generates_copies(sample_item_with_choice):
    """REQ-035: make_order_suite generates shuffles copies (5 by default)."""
    items, groups = make_order_suite([sample_item_with_choice])

    assert len(items) == 5, "Should generate 5 copies for 1 choice question (default shuffles=5)"

    for i, item in enumerate(items):
        assert item.id == f"test-item-001#order{i}", f"Copy {i} should have id with order index"
        assert item.suite == "order"
        assert "order" in item.tags
        assert "test" in item.tags


def test_order_suite_different_permutations(sample_item_with_choice):
    """REQ-035: Different copies have different permutations of criteria keys."""
    items, groups = make_order_suite([sample_item_with_choice])

    # Extract the key order from each copy's criteria dict
    key_orders = []
    for item in items:
        keys = tuple(item.questions["quality"]["criteria"].keys())
        key_orders.append(keys)

    # Check that we have at least 2 different orderings
    unique_orders = set(key_orders)
    assert len(unique_orders) >= 2, f"Should have at least 2 different orderings, got {unique_orders}"


def test_order_suite_expected_unchanged(sample_item_with_choice):
    """REQ-035: expected field should be unchanged."""
    items, groups = make_order_suite([sample_item_with_choice])

    for item in items:
        assert item.expected == {"quality": "good", "helpful": "yes"}


def test_order_suite_groups_mapping(sample_item_with_choice):
    """REQ-035: groups dict maps group_id to list of copy ids."""
    items, groups = make_order_suite([sample_item_with_choice])

    # For 1 item with 1 choice question, should have 1 group
    expected_group_id = "test-item-001/quality"
    assert expected_group_id in groups

    # Group should contain all 5 copy ids
    expected_ids = [f"test-item-001#order{i}" for i in range(5)]
    assert groups[expected_group_id] == expected_ids


def test_order_suite_pair_id(sample_item_with_choice):
    """REQ-035: pair_id should be set to {item_id}/{qid}."""
    items, groups = make_order_suite([sample_item_with_choice])

    for item in items:
        assert item.pair_id == "test-item-001/quality"


def test_order_suite_noul_questions_skipped(sample_item_with_choice):
    """REQ-035: noul questions should not generate separate items (only choice questions do)."""
    items, groups = make_order_suite([sample_item_with_choice])

    # Should only have 1 group (for the choice question), not 2
    assert len(groups) == 1
    assert "test-item-001/helpful" not in groups


def test_order_suite_seed_determinism(sample_item_with_choice):
    """REQ-035: same seed produces identical output."""
    seed = 20260922
    items1, groups1 = make_order_suite([sample_item_with_choice], seed=seed)
    items2, groups2 = make_order_suite([sample_item_with_choice], seed=seed)

    # Compare ids and questions
    for i1, i2 in zip(items1, items2):
        assert i1.id == i2.id
        assert i1.questions == i2.questions


def test_order_suite_item_without_choice_is_skipped():
    """REQ-035: Items without choice questions should be skipped."""
    item = DecisionItem(
        id="noul-only",
        suite="test",
        state="state",
        questions={
            "q1": {"type": "noul", "instructions": "noul question"}
        },
        expected=None,
    )

    items, groups = make_order_suite([item])

    assert len(items) == 0, "Item without choice questions should not produce copies"
    assert len(groups) == 0


def test_order_suite_multiple_choice_questions():
    """REQ-035: item with multiple choice questions generates separate groups."""
    item = DecisionItem(
        id="multi-choice",
        suite="test",
        state="state",
        questions={
            "q1": {
                "type": "choice",
                "instructions": "first choice",
                "criteria": {"a": "A", "b": "B"},
            },
            "q2": {
                "type": "choice",
                "instructions": "second choice",
                "criteria": {"x": "X", "y": "Y"},
            },
        },
        expected=None,
    )

    items, groups = make_order_suite([item], shuffles=3)

    # Should have 3 copies for q1 and 3 copies for q2 = 6 total
    assert len(items) == 6

    # Should have 2 groups
    assert len(groups) == 2
    assert "multi-choice/q1" in groups
    assert "multi-choice/q2" in groups


def test_order_suite_custom_shuffles():
    """REQ-035: custom shuffles parameter."""
    item = DecisionItem(
        id="test",
        suite="test",
        state="state",
        questions={
            "q": {"type": "choice", "instructions": "choice", "criteria": {"a": "A", "b": "B"}}
        },
        expected=None,
    )

    items, groups = make_order_suite([item], shuffles=2)

    assert len(items) == 2
    assert groups["test/q"] == ["test#order0", "test#order1"]
