"""Brier loss term, surface-perturbation augmentation, and the omj-holdout suite."""

from __future__ import annotations

import collections
import random

import pytest
import torch

from omj.bench.suites import load_suite
from omj.train.loss import restricted_brier
from omj.train.sources import DecisionRecord, add_surface_augmentations, augment_surface


def test_brier_is_zero_for_a_confident_correct_answer_and_respects_the_mask() -> None:
    logits = torch.full((1, 10), -50.0)
    logits[0, 3] = 50.0
    ids = torch.tensor([[3, 4, 5]])
    assert restricted_brier(logits, ids, torch.tensor([0])).item() == pytest.approx(0.0, abs=1e-6)
    assert restricted_brier(logits, ids, torch.tensor([1])).item() == pytest.approx(2.0, abs=1e-6)
    uniform = torch.zeros((1, 10))
    masked = restricted_brier(uniform, ids, torch.tensor([0]), label_mask=torch.tensor([[True, True, False]]))
    assert masked.item() == pytest.approx(0.5, abs=1e-6)  # (0.5-1)^2 + 0.5^2


def _rec() -> DecisionRecord:
    q = {"type": "choice", "instructions": "Which?", "criteria": {"alarm_set": "set an alarm", "weather_query": "ask the weather"}}
    return DecisionRecord("massive:1/intent", "massive", {"utterance": "wake me at 7", "lang": "en"}, q, "alarm_set", 7)


def test_augment_surface_keeps_the_meaning_of_the_answer() -> None:
    renamed = 0
    for seed in range(60):
        a = augment_surface(_rec(), random.Random(seed))
        assert a.id.endswith(":aug") and a.expected in a.question["criteria"]
        assert a.question["criteria"][a.expected] == "set an alarm"
        renamed += a.expected != "alarm_set"
    assert 10 < renamed < 50


def test_surface_augmentation_fraction_and_determinism() -> None:
    recs = [DecisionRecord(f"r{i}", "x", {"a": i, "b": 2}, {"type": "noul", "instructions": "ok?"}, "yes", i) for i in range(2000)]
    out = add_surface_augmentations(recs, 0.15, seed=7)
    assert out[:2000] == recs
    assert 0.10 < (len(out) - 2000) / 2000 < 0.20
    assert [r.id for r in out] == [r.id for r in add_surface_augmentations(recs, 0.15, seed=7)]
    assert add_surface_augmentations(recs, 0.0, seed=7) == recs


def test_holdout_suite_loads_with_licenses() -> None:
    items = load_suite("omj-holdout")
    assert len(items) == 300
    assert collections.Counter(i.tags[1] for i in items) == {"boolq": 75, "arc": 75, "commonsense_qa": 75, "svamp": 75}
    assert all(i.license for i in items)
