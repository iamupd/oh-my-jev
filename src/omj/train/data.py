"""Training data build for ``omj train``: load MASSIVE rows, render/tokenize
examples via the same path the semif backend serves with, dev sampling, the
test-leak guard, and the label/count statistics recorded into ``train.json``.

REQ-003: build one example per (train row, question) and refuse to proceed if
any of those rows collide with the pinned bench test id list.
REQ-004: examples are rendered through ``render_prompt_text_with``, the exact
function the semif backend uses at inference time.
REQ-005: when configured, permute each example's option order deterministically
from the recipe seed, the row id, and the question id.
REQ-017: two builds of the same recipe over the same rows produce an identical
example list (order included).
REQ-020: the returned ``DataStats`` carries the example count, label
distribution, and test-id overlap for the run's audit trail.
"""

from __future__ import annotations

import json
import random
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from omj.backends.semif import render_prompt_text_with
from omj.bench.suites.base import Downloader
from omj.bench.suites.local import SUITES_DIR
from omj.bench.suites.massive import (
    CRITERIA_FILE,
    DEFAULT_IDS_FILE,
    INSTRUCTIONS,
    INTENTS,
    LOCALES,
    SCENARIOS,
    download_massive,
    read_ids_file,
)
from omj.errors import ErrorCode, OmjError
from omj.train.recipe import Recipe

_LOCALE_TAGS: set[str] = set(LOCALES.values())


@dataclass
class TrainExample:
    """One rendered, tokenized (row, question) pair ready for the training loop."""

    id: str
    row_id: int
    qid: str
    input_ids: list[int]
    label_token_ids: list[int]
    target_index: int
    keys: list[str]
    n_tokens: int


@dataclass
class DataStats:
    """Build-time counters recorded into ``train.json``'s data section (REQ-020)."""

    n_rows: int
    n_examples: int
    n_dropped_too_long: int
    label_counts: dict[str, dict[str, int]]
    test_overlap: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_examples": self.n_examples,
            "n_dropped_too_long": self.n_dropped_too_long,
            "label_counts": self.label_counts,
            "test_overlap": self.test_overlap,
            "seed": self.seed,
        }


def _short_locale(locale: str) -> str:
    """Map a recipe locale (``ko-KR``/``ko``) to MASSIVE's short code (``ko``)."""
    if locale in LOCALES:
        return locale
    for code, tag in LOCALES.items():
        if tag == locale:
            return code
    raise OmjError(
        ErrorCode.E_SCHEMA,
        f"unknown MASSIVE locale {locale!r} (expected one of "
        f"{', '.join(sorted(_LOCALE_TAGS))})",
    )


def _qid_hash(qid: str) -> int:
    """A hash of a question id that is stable across processes and Python versions.

    Python's builtin ``hash()`` is salted per interpreter (``PYTHONHASHSEED``),
    which would make the option-shuffle seed --- and therefore the built
    examples --- irreproducible across separate ``omj train`` runs (REQ-017).
    """
    return zlib.crc32(qid.encode("utf-8"))


def load_massive_rows(
    locale: str,
    split: Literal["train", "dev", "test"],
    *,
    cache_dir: Path | str | None = None,
    downloader: Downloader | None = None,
) -> list[dict]:
    """Rows of one MASSIVE locale file narrowed to a single partition.

    ``locale`` is the recipe's IETF tag (``ko-KR``/``en-US``), matching the
    file names ``download_massive`` extracts the tarball into.
    """
    if locale not in _LOCALE_TAGS:
        raise OmjError(
            ErrorCode.E_SCHEMA,
            f"unknown MASSIVE locale {locale!r} (expected one of "
            f"{', '.join(sorted(_LOCALE_TAGS))})",
        )
    root = download_massive(cache_dir=cache_dir, downloader=downloader)
    path = root / "data" / f"{locale}.jsonl"
    if not path.is_file():
        raise OmjError(ErrorCode.E_DOWNLOAD, f"{path} is missing from the MASSIVE cache")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OmjError(ErrorCode.E_SCHEMA, f"{path} line {lineno}: {exc}") from exc
            if row.get("partition") != split:
                continue
            rows.append(row)
    return rows


def load_criteria(locale: str, criteria_file: str = "") -> dict:
    """The authored scenario/intent option descriptions for one language.

    ``criteria_file`` overrides the default ``suites/massive-criteria.<lang>.json``
    (relative paths resolve under ``suites/``), mirroring
    ``Recipe.data.criteria_file``.
    """
    code = _short_locale(locale)
    if criteria_file:
        candidate = Path(criteria_file)
        path = candidate if candidate.is_absolute() else SUITES_DIR / candidate
    else:
        path = SUITES_DIR / CRITERIA_FILE.format(locale=code)
    if not path.is_file():
        raise OmjError(ErrorCode.E_SCHEMA, f"MASSIVE criteria file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    for group, labels in (("scenario", SCENARIOS), ("intent", INTENTS)):
        options = data.get(group)
        if not isinstance(options, dict) or set(options) != set(labels):
            raise OmjError(
                ErrorCode.E_SCHEMA,
                f"{path}: '{group}' must describe exactly the {len(labels)} MASSIVE {group} labels",
            )
    return data


def question_dict(qid: str, criteria: dict, locale: str) -> dict:
    """The question object ``render_prompt_text_with`` expects for one qid (REQ-004)."""
    code = _short_locale(locale)
    return {
        "type": "choice",
        "instructions": INSTRUCTIONS[code][qid],
        "criteria": criteria[qid],
    }


def sample_dev(rows: list[dict], n: int, seed: int) -> list[dict]:
    """A deterministic sample of up to ``n`` rows, independent of input row order."""
    ordered = sorted(rows, key=lambda row: int(row["id"]))
    count = min(n, len(ordered))
    return random.Random(seed).sample(ordered, count)


def check_no_leak(
    row_ids: Iterable[int], ids_file: Path | None = None, *, strict: bool = True
) -> int:
    """Count ids in ``row_ids`` that are pinned bench test ids; raise E_CONFIG when strict (REQ-003).

    The count is what ``DataStats.test_overlap`` records (REQ-020), so ``strict=False``
    lets a caller measure the leak instead of aborting on it.
    """
    target = DEFAULT_IDS_FILE if ids_file is None else ids_file
    test_ids = set(read_ids_file(target))
    overlap = sorted(set(int(row_id) for row_id in row_ids) & test_ids)
    if overlap and strict:
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"{len(overlap)} training/dev row id(s) overlap with the MASSIVE test id list "
            f"{target} (first offending id: {overlap[0]})",
        )
    return len(overlap)


def build_examples(
    rows: list[dict],
    recipe: Recipe,
    tokenizer: Any,
    criteria: dict,
    *,
    split: Literal["train", "dev"],
) -> tuple[list[TrainExample], DataStats]:
    """Render, tokenize, and (for train) shuffle every row x question (REQ-003/004/005/017/020)."""
    locale = recipe.data.locale
    max_seq_len = recipe.train.max_seq_len
    seed = recipe.train.seed
    shuffle = split == "train" and recipe.data.shuffle_options

    row_ids = [int(row["id"]) for row in rows]
    test_overlap = check_no_leak(row_ids)

    label_counts: dict[str, dict[str, int]] = {qid: {} for qid in recipe.data.questions}
    n_dropped_too_long = 0
    built: list[tuple[int, str, TrainExample]] = []

    for row in sorted(rows, key=lambda r: int(r["id"])):
        row_id = int(row["id"])
        for qid in recipe.data.questions:
            label = row[qid]
            label_counts[qid][label] = label_counts[qid].get(label, 0) + 1

            question = question_dict(qid, criteria, locale)
            option_order: list[int] | None = None
            if shuffle:
                n_options = len(question["criteria"])
                option_order = random.Random(
                    seed * 1_000_003 + row_id * 7 + _qid_hash(qid)
                ).sample(range(n_options), n_options)

            rendered = render_prompt_text_with(
                tokenizer, row["utt"], question, option_order=option_order
            )
            input_ids = tokenizer(rendered.text, add_special_tokens=False)["input_ids"]
            n_tokens = len(input_ids)
            if n_tokens > max_seq_len:
                n_dropped_too_long += 1
                continue

            built.append(
                (
                    row_id,
                    qid,
                    TrainExample(
                        id=f"{row_id}/{qid}",
                        row_id=row_id,
                        qid=qid,
                        input_ids=list(input_ids),
                        label_token_ids=list(rendered.label_token_ids),
                        target_index=rendered.keys.index(label),
                        keys=list(rendered.keys),
                        n_tokens=n_tokens,
                    ),
                )
            )

    built.sort(key=lambda triple: (triple[0], triple[1]))
    examples = [example for _, _, example in built]
    if split == "train":
        random.Random(seed).shuffle(examples)

    stats = DataStats(
        n_rows=len(rows),
        n_examples=len(examples),
        n_dropped_too_long=n_dropped_too_long,
        label_counts=label_counts,
        test_overlap=test_overlap,
        seed=seed,
    )
    return examples, stats


def build_examples_from_records(
    records: list[Any],
    recipe: Recipe,
    tokenizer: Any,
    *,
    split: Literal["train", "dev"],
) -> tuple[list[TrainExample], DataStats]:
    """Render/tokenize mixed-source ``DecisionRecord``s the same way as ``build_examples``.

    ``label_counts`` is keyed by source name; ``test_overlap`` counts MASSIVE
    rows that collide with the pinned test id list (always 0 for other sources).
    """
    max_seq_len = recipe.train.max_seq_len
    seed = recipe.train.seed
    shuffle = split == "train" and recipe.data.shuffle_options

    massive_ids = [
        int(r.id.split(":", 1)[1].split("/", 1)[0]) for r in records if r.source in ("massive", "massive-en")
    ]
    test_overlap = check_no_leak(massive_ids) if massive_ids else 0

    label_counts: dict[str, dict[str, int]] = {}
    n_dropped_too_long = 0
    built: list[TrainExample] = []
    for index, record in enumerate(sorted(records, key=lambda r: r.id)):
        counts = label_counts.setdefault(record.source, {})
        counts[record.expected] = counts.get(record.expected, 0) + 1

        option_order: list[int] | None = None
        if shuffle and record.question["type"] == "choice":
            n_options = len(record.question["criteria"])
            option_order = random.Random(seed * 1_000_003 + record.group).sample(range(n_options), n_options)

        rendered = render_prompt_text_with(tokenizer, record.state, record.question, option_order=option_order)
        input_ids = tokenizer(rendered.text, add_special_tokens=False)["input_ids"]
        if len(input_ids) > max_seq_len:
            n_dropped_too_long += 1
            continue
        if record.expected not in rendered.keys:
            raise OmjError(
                ErrorCode.E_SCHEMA,
                f"{record.id}: expected answer {record.expected!r} is not one of {rendered.keys}",
            )
        built.append(
            TrainExample(
                id=record.id,
                row_id=index,
                qid=record.source,
                input_ids=list(input_ids),
                label_token_ids=list(rendered.label_token_ids),
                target_index=rendered.keys.index(record.expected),
                keys=list(rendered.keys),
                n_tokens=len(input_ids),
            )
        )

    if split == "train":
        random.Random(seed).shuffle(built)
    stats = DataStats(
        n_rows=len(records),
        n_examples=len(built),
        n_dropped_too_long=n_dropped_too_long,
        label_counts=label_counts,
        test_overlap=test_overlap,
        seed=seed,
    )
    return built, stats
