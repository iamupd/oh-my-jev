"""Loader for the public JevBench split, pinned to one upstream commit."""

from __future__ import annotations

import json
from pathlib import Path

from omj.bench.suites.base import (
    DecisionItem,
    Downloader,
    cache_root,
    resolve_downloader,
    validate_item,
)
from omj.errors import ErrorCode, OmjError

SUITE_NAME = "jevbench-public"
JEVBENCH_REPO = "fstandhartinger/jevbench"
JEVBENCH_COMMIT = "75e6224ed8103bbc3485ca74820a2eaf7ce8abe0"
JEVBENCH_RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/datasets/public/{name}.jsonl"

#: file tag -> pinned raw URL. The tag is also the difficulty tag put on every row.
JEVBENCH_FILES: dict[str, str] = {
    name: JEVBENCH_RAW_URL.format(repo=JEVBENCH_REPO, commit=JEVBENCH_COMMIT, name=name)
    for name in ("original", "easy", "hard")
}

DEFAULT_LICENSE = "MIT"
QUESTION_ID = "decision"


def jevbench_cache_dir(cache_dir: Path | str | None = None) -> Path:
    return cache_root(cache_dir) / "jevbench" / JEVBENCH_COMMIT


def _fetch_file(name: str, url: str, target: Path, downloader: Downloader) -> str:
    if target.is_file():
        return target.read_text(encoding="utf-8")

    payload = downloader(url)
    if not payload:
        raise OmjError(ErrorCode.E_DOWNLOAD, f"jevbench {name}.jsonl: empty response from {url}")
    text = payload.decode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write through a temp file so an interrupted run never leaves a half file
    # that the next run would happily treat as a cache hit.
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(target)
    return text


def _to_item(row: dict, file_tag: str) -> DecisionItem:
    question = row.get("question")
    if not isinstance(question, dict):
        raise OmjError(ErrorCode.E_SCHEMA, f"jevbench {row.get('id')!r}: missing question object")

    tags = [file_tag]
    family = row.get("family")
    if family and family not in tags:
        tags.append(str(family))

    provenance = row.get("provenance") or {}
    expected = row.get("expected")

    return DecisionItem(
        id=str(row.get("id", "")),
        suite=SUITE_NAME,
        state=row.get("state", ""),
        questions={QUESTION_ID: question},
        expected={QUESTION_ID: str(expected)} if expected is not None else None,
        tags=tags,
        source=f"{JEVBENCH_REPO}@{JEVBENCH_COMMIT}",
        license=str(provenance.get("license") or DEFAULT_LICENSE),
    )


def load_jevbench_public(
    cache_dir: Path | str | None = None,
    downloader: Downloader | None = None,
) -> list[DecisionItem]:
    """Download (once) and convert the three public JevBench files into DecisionItems."""
    fetch = resolve_downloader(downloader)
    target_dir = jevbench_cache_dir(cache_dir)

    items: list[DecisionItem] = []
    for name, url in JEVBENCH_FILES.items():
        text = _fetch_file(name, url, target_dir / f"{name}.jsonl", fetch)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OmjError(ErrorCode.E_SCHEMA, f"jevbench {name}.jsonl line {lineno}: {exc}") from exc
            item = _to_item(row, name)
            validate_item(item)
            items.append(item)

    return items
