"""Turn dev-split predictions into a P0 calibration file (temperature scaling).

The trainer holds restricted label logits for every dev example. Wrapping them in the
same ``ResultRow`` the bench runners emit lets the P0 temperature fitter run unchanged,
so the adapter ships with a calibration.json the gateway already knows how to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from omj.bench.temperature import fit_and_report
from omj.bench.types import ResultRow
from omj.gateway.assemble import softmax
from omj.gateway.calibration import Calibration

DEV_SUITE = "massive-ko-dev"


def dev_result_rows(
    dev_examples: Sequence[Any],
    probs_or_logits: Sequence[Sequence[float]],
    *,
    backend: str,
    model: str,
) -> list[ResultRow]:
    """One ResultRow per dev example, carrying its restricted logits and softmax probs."""
    if len(dev_examples) != len(probs_or_logits):
        raise ValueError(
            f"got {len(probs_or_logits)} logit rows for {len(dev_examples)} dev examples"
        )

    rows: list[ResultRow] = []
    for example, values in zip(dev_examples, probs_or_logits):
        keys = list(example.keys)
        logits = [float(value) for value in values][: len(keys)]
        probs = softmax(logits)
        target = int(example.target_index)
        predicted = max(range(len(probs)), key=lambda i: probs[i]) if probs else target
        rows.append(
            ResultRow(
                item_id=str(example.id),
                suite=DEV_SUITE,
                qid=str(example.qid),
                kind="choice",
                keys=keys,
                probs=probs,
                logits=logits,
                expected=keys[target],
                correct=predicted == target,
                latency_ms=0.0,
                cost_usd=None,
                backend=backend,
                model=model,
                tags=[str(example.qid)],
            )
        )
    return rows


def calibrate_from_dev(
    dev_examples: Sequence[Any],
    probs_or_logits: Sequence[Sequence[float]],
    *,
    backend: str,
    model: str,
    out_path: Path,
) -> Calibration:
    """Fit per-kind temperatures on the dev predictions and write ``out_path``."""
    rows = dev_result_rows(dev_examples, probs_or_logits, backend=backend, model=model)
    calibration, _report = fit_and_report(
        rows, backend=backend, model=model, suite=DEV_SUITE, out_path=out_path
    )
    return calibration
