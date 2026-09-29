"""JevBench submission bundle writer per the fixed adapter contract (design.md 0/3; REQ-042)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from omj import __version__
from omj.bench.types import ResultRow

_ADAPTER_NAME = "omj"


def _probabilities_for_row(row: ResultRow) -> dict[str, float]:
    if row.error is not None:
        return {}
    if row.probs is None or not row.keys or len(row.probs) != len(row.keys):
        return {}
    if row.kind == "noul":
        try:
            yes_index = row.keys.index("yes")
        except ValueError:
            yes_index = 0
        p_yes = row.probs[yes_index]
        return {"yes": p_yes, "no": 1.0 - p_yes}
    return dict(zip(row.keys, row.probs))


def write_submission(
    out_dir: Path,
    rows: list[ResultRow],
    *,
    model: str,
    backend: str,
    price_input_per_m: float | None,
) -> Path:
    """Write a JevBench-adapter submission bundle under `out_dir/submission/` (REQ-042).

    Creates results.jsonl (one line per decision), raw/<decision_id>.json
    (the raw answer produced for that decision), manifest.json, and
    ledger.jsonl, matching the contract verified from the jevbench-public
    adapter loader.
    """
    submission_dir = out_dir / "submission"
    raw_dir = submission_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0
    any_cost = False

    results_path = submission_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            status = "error" if row.error else "ok"

            result_row = {
                "decision_id": row.item_id,
                "probabilities": _probabilities_for_row(row),
                "latency_seconds": row.latency_ms / 1000.0,
                "cost_usd": row.cost_usd,
                "correct": row.correct if status == "ok" else None,
                "model": row.model,
                "adapter": _ADAPTER_NAME,
                "status": status,
                "error": row.error,
            }
            f.write(json.dumps(result_row, ensure_ascii=False) + "\n")

            if row.cost_usd is not None:
                total_cost += row.cost_usd
                any_cost = True

            raw_answer = {
                "decision_id": row.item_id,
                "suite": row.suite,
                "qid": row.qid,
                "kind": row.kind,
                "keys": row.keys,
                "probs": row.probs,
                "logits": row.logits,
                "expected": row.expected,
                "correct": row.correct,
                "error": row.error,
            }
            raw_path = raw_dir / f"{row.item_id}.json"
            raw_path.write_text(
                json.dumps(raw_answer, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    manifest = {
        "adapter": _ADAPTER_NAME,
        "model": model,
        "backend": backend,
        "omj_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_decisions": len(rows),
        "price_input_per_m": price_input_per_m,
        "price_output_per_m": 0,
        "notes": "",
    }
    manifest_path = submission_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    ledger_entry = {
        "reserved_usd": None,
        "settled_usd": total_cost if any_cost else None,
    }
    ledger_path = submission_dir / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(ledger_entry, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return submission_dir
