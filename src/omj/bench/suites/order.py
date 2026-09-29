"""Generate test variants by shuffling the order of choice question criteria keys.

This module creates multiple copies of benchmark items, where each copy has the
criteria dictionary keys for choice questions permuted in a deterministic but
varied way. This tests whether model confidence is sensitive to option ordering.
"""

from __future__ import annotations

import random
from copy import deepcopy

from omj.bench.suites.base import DecisionItem


def make_order_suite(
    items: list[DecisionItem],
    *,
    shuffles: int = 5,
    seed: int = 20260922,
) -> tuple[list[DecisionItem], dict[str, list[str]]]:
    """Generate permuted copies of items for choice questions.

    For every choice question in every item, produces `shuffles` copies with
    deterministically permuted criteria key order. Uses random.Random(seed + k)
    for permutation k to ensure determinism and consistency per item/question.

    Args:
        items: List of DecisionItems to expand.
        shuffles: Number of permutations per item/question pair (default 5).
        seed: Base seed for random number generation (default 20260922).

    Returns:
        (items, groups) where:
        - items: expanded list of DecisionItems with ids like "orig_id#orderK"
        - groups: dict mapping "orig_id/qid" to list of ["orig_id#order0", ...]
    """
    all_items: list[DecisionItem] = []
    groups: dict[str, list[str]] = {}

    for item in items:
        # Find all choice questions
        choice_qids = [
            qid for qid, q in item.questions.items()
            if q.get("type") == "choice"
        ]

        # Skip items without choice questions
        if not choice_qids:
            continue

        # For each choice question, generate shuffles copies
        for qid in choice_qids:
            group_id = f"{item.id}/{qid}"
            group_ids: list[str] = []

            for k in range(shuffles):
                copy = deepcopy(item)
                copy.id = f"{item.id}#order{k}"
                copy.suite = "order"
                copy.tags = item.tags + ["order"]
                copy.pair_id = group_id

                # Permute the criteria keys for this question
                question = copy.questions[qid]
                original_criteria = question["criteria"]

                # Get the keys and permute them
                keys = list(original_criteria.keys())
                rng = random.Random(seed + k)
                rng.shuffle(keys)

                # Rebuild criteria dict with permuted key order
                question["criteria"] = {k: original_criteria[k] for k in keys}

                all_items.append(copy)
                group_ids.append(copy.id)

            groups[group_id] = group_ids

    return all_items, groups
