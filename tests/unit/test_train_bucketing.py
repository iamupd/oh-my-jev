"""Length bucketing in the training epoch order must be deterministic and length-sorted per chunk.

# REQ-017
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

torch = pytest.importorskip("torch")

from omj.train.loop import Trainer  # noqa: E402
from omj.train.recipe import DataSection, LoraSection, Recipe, TrainSection  # noqa: E402


@dataclass
class _Ex:
    id: str
    row_id: int
    qid: str
    input_ids: list[int]
    label_token_ids: list[int]
    target_index: int
    keys: list[str] = field(default_factory=lambda: ["a", "b"])
    n_tokens: int = 0


def _recipe(batch_size: int) -> Recipe:
    return Recipe(
        name="t",
        base_model="x",
        data=DataSection(questions=["scenario"]),
        lora=LoraSection(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        train=TrainSection(
            lr=1e-3, epochs=1, batch_size=batch_size, grad_accum=1, max_seq_len=128,
            label_smoothing=0.0, seed=7, eval_every_steps=5,
        ),
    )


def _examples(n: int) -> list[_Ex]:
    return [_Ex(id=str(i), row_id=i, qid="scenario", input_ids=[1] * (10 + (i % 5) * 20),
                label_token_ids=[3, 4], target_index=i % 2, n_tokens=10 + (i % 5) * 20) for i in range(n)]


def _order(batch_size: int, n: int, seed_epoch: int = 0) -> list[int]:
    trainer = Trainer.__new__(Trainer)
    trainer.train_examples = _examples(n)
    trainer.batch_size = batch_size
    trainer.recipe = _recipe(batch_size)
    return Trainer._epoch_order(trainer, seed_epoch)


def test_epoch_order_is_deterministic_and_a_permutation() -> None:
    a = _order(2, 100)
    b = _order(2, 100)
    assert a == b
    assert sorted(a) == list(range(100))


def test_each_chunk_is_sorted_by_length_descending() -> None:
    batch_size = 2
    order = _order(batch_size, 100)
    examples = _examples(100)
    chunk = batch_size * 16
    for start in range(0, len(order), chunk):
        lengths = [examples[i].n_tokens for i in order[start : start + chunk]]
        assert lengths == sorted(lengths, reverse=True)


def test_different_epochs_give_different_orders() -> None:
    assert _order(2, 100, 0) != _order(2, 100, 1)
