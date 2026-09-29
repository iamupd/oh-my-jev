"""Build suites/omj-holdout.jsonl: an out-of-distribution decision suite for choosing model versions.

None of these datasets is used by the bundled training sources, so the suite stands in
for a sealed set: select checkpoints and recipes on it instead of on the public
JevBench items. 75 items per source, sampled deterministically.

Sources (Hugging Face dataset cards, checked 2026-09-24):
- google/boolq validation           CC-BY-SA-3.0  passage yes/no
- allenai/ai2_arc ARC-Challenge test CC-BY-SA-4.0 science 4-way choice
- tau/commonsense_qa validation     MIT           commonsense 5-way choice
- ChilleD/SVAMP test                MIT           math word problem, 4-way numeric choice

MathQA is deliberately not used: it re-annotates AQuA-RAT problems, a common training source,
so its items overlap with AQuA-trained models.

Usage: uv run python scripts/make_holdout_suite.py
"""

from __future__ import annotations

import ast
import io
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

from omj.train.sources import fetch_cached

HF = "https://huggingface.co/api/datasets"
PER_SOURCE = 75
OVERSAMPLE = 120  # candidates per source before the contamination filter
SEED = 20260924
OUT = Path(__file__).resolve().parents[1] / "suites" / "omj-holdout.jsonl"


def _rows(url: str, name: str) -> list[dict]:
    return pq.read_table(io.BytesIO(fetch_cached(url, name))).to_pylist()


def _pick(rows: list[dict], key: str) -> list[dict]:
    rows = sorted(rows, key=lambda r: str(r[key]))
    return random.Random(SEED).sample(rows, min(OVERSAMPLE, len(rows)))


TRAIN_SOURCES = ("massive", "massive-en", "openjev-business", "banking77", "klue-ynat", "klue-nli", "nsmc")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", s.lower()).strip()


def _grams(s: str, n: int = 8) -> set[str]:
    w = _norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _item_text(item: dict) -> str:
    return json.dumps(item["state"], ensure_ascii=False) + " " + " ".join(q["instructions"] for q in item["questions"].values())


def _train_grams() -> set[str]:
    from omj.train.sources import load_source

    grams: set[str] = set()
    for name in TRAIN_SOURCES:
        for split in ("train", "dev"):
            for r in load_source(name, split):
                grams |= _grams(json.dumps(r.state, ensure_ascii=False) + " " + str(r.question["instructions"]))
    return grams


def _clean(items: list[dict], train: set[str]) -> list[dict]:
    """Drop items sharing >= 3 8-grams with any training source, then keep the first PER_SOURCE."""
    kept = [it for it in items if len(_grams(_item_text(it)) & train) < 3]
    for n, it in enumerate(kept[:PER_SOURCE]):
        it["id"] = re.sub(r"-\d{3}$", f"-{n:03d}", it["id"])
    return kept[:PER_SOURCE]


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else ast.literal_eval(value)


def boolq() -> list[dict]:
    out = []
    for i, r in enumerate(_pick(_rows(f"{HF}/google/boolq/parquet/default/validation/0.parquet", "holdout-boolq.bin"), "question")):
        q = r["question"].strip().rstrip("?") + "?"
        out.append({"id": f"holdout-boolq-{i:03d}", "tags": ["holdout", "boolq"], "state": r["passage"],
                    "questions": {"answer": {"type": "noul", "instructions": q[0].upper() + q[1:]}},
                    "expected": {"answer": "yes" if str(r["answer"]).lower() == "true" else "no"},
                    "source": "google/boolq", "license": "CC-BY-SA-3.0"})
    return out


def arc() -> list[dict]:
    out = []
    for i, r in enumerate(_pick(_rows(f"{HF}/allenai/ai2_arc/parquet/ARC-Challenge/test/0.parquet", "holdout-arc.bin"), "id")):
        ch = _as_dict(r["choices"])
        crit = dict(zip(ch["label"], ch["text"]))
        if r["answerKey"] not in crit:
            continue
        out.append({"id": f"holdout-arc-{i:03d}", "tags": ["holdout", "arc"], "state": "",
                    "questions": {"answer": {"type": "choice", "instructions": r["question"], "criteria": crit}},
                    "expected": {"answer": r["answerKey"]}, "source": "allenai/ai2_arc", "license": "CC-BY-SA-4.0"})
    return out


def csqa() -> list[dict]:
    out = []
    for i, r in enumerate(_pick(_rows(f"{HF}/tau/commonsense_qa/parquet/default/validation/0.parquet", "holdout-csqa.bin"), "id")):
        ch = _as_dict(r["choices"])
        crit = dict(zip(ch["label"], ch["text"]))
        out.append({"id": f"holdout-csqa-{i:03d}", "tags": ["holdout", "commonsense_qa"], "state": "",
                    "questions": {"answer": {"type": "choice", "instructions": r["question"], "criteria": crit}},
                    "expected": {"answer": r["answerKey"]}, "source": "tau/commonsense_qa", "license": "MIT"})
    return out


def _perturb(value: str, rng: random.Random) -> list[str]:
    """Plausible wrong numbers for a numeric answer: small offsets, doubling, halving, x10."""
    try:
        x = float(value)
    except ValueError:
        return []
    out = []
    for c in {x + d for d in (-10, -2, -1, 1, 2, 10)} | {x * 2, x / 2, x * 10}:
        if c == x or c < 0:
            continue
        s = str(int(c)) if float(c).is_integer() else f"{c:.2f}".rstrip("0").rstrip(".")
        if s != value:
            out.append(s)
    rng.shuffle(out)
    return out


def svamp() -> list[dict]:
    out = []
    for i, r in enumerate(_pick(_rows(f"{HF}/ChilleD/SVAMP/parquet/default/test/0.parquet", "holdout-svamp.bin"), "ID")):
        gold = str(r["Answer"]).rstrip("0").rstrip(".") if "." in str(r["Answer"]) else str(r["Answer"])
        rng = random.Random(SEED + i)
        wrong = _perturb(gold, rng)
        if len(wrong) < 3:
            continue
        opts = [gold] + wrong[:3]
        rng.shuffle(opts)
        crit = {k: v for k, v in zip("ABCD", opts)}
        out.append({"id": f"holdout-svamp-{i:03d}", "tags": ["holdout", "svamp"], "state": r["Body"],
                    "questions": {"answer": {"type": "choice", "instructions": r["Question"], "criteria": crit}},
                    "expected": {"answer": "ABCD"[opts.index(gold)]}, "source": "ChilleD/SVAMP", "license": "MIT"})
    return out


def main() -> None:
    train = _train_grams()
    rows = []
    for build in (boolq, arc, csqa, svamp):
        cand = build()
        clean = _clean(cand, train)
        print(f"{build.__name__}: {len(cand)} candidates, {len(clean)} kept after the contamination filter")
        rows += clean
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} items to {OUT}")


if __name__ == "__main__":
    main()
