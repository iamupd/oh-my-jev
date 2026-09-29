"""Loader for the synthetic JSONL suites committed under the repo's suites/ directory."""

from __future__ import annotations

import json
from pathlib import Path

from omj.bench.suites.base import DecisionItem, validate_item
from omj.errors import ErrorCode, OmjError

_HERE = Path(__file__).resolve()
# A source checkout keeps suites/ at the repo root (src/omj/bench/suites/local.py);
# an installed wheel has to ship it next to the omj package, which is one level up.
_SUITES_CANDIDATES = (_HERE.parents[4] / "suites", _HERE.parents[3] / "suites")


def _resolve_suites_dir() -> Path:
    for candidate in _SUITES_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return _SUITES_CANDIDATES[0]


SUITES_DIR = _resolve_suites_dir()

LOCAL_SUITES = ("omj-smoke", "underdetermined", "omj-holdout")


def local_suite_path(suite_name: str) -> Path:
    return SUITES_DIR / f"{suite_name}.jsonl"


def load_local(path: Path | str, suite_name: str) -> list[DecisionItem]:
    """Read a JSONL suite file and validate every row."""
    target = Path(path)
    if not target.is_file():
        raise OmjError(ErrorCode.E_SCHEMA, f"suite file not found: {target}")

    items: list[DecisionItem] = []
    with target.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OmjError(ErrorCode.E_SCHEMA, f"{target} line {lineno}: {exc}") from exc
            if not isinstance(row, dict):
                raise OmjError(ErrorCode.E_SCHEMA, f"{target} line {lineno}: row must be a JSON object")

            item = DecisionItem(
                id=str(row.get("id", "")),
                suite=suite_name,
                state=row.get("state", ""),
                questions=row.get("questions") or {},
                expected=row.get("expected"),
                tags=list(row.get("tags") or []),
                expected_shape=row.get("expected_shape"),
                pair_id=row.get("pair_id"),
                source=row.get("source", "omj-synthetic"),
                license=row.get("license", "CC0-1.0"),
            )
            try:
                validate_item(item)
            except OmjError as exc:
                raise OmjError(ErrorCode.E_SCHEMA, f"{target} line {lineno}: {exc.message}") from exc
            items.append(item)

    return items
