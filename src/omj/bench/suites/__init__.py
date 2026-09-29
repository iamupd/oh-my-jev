"""Benchmark suite loaders: synthetic local JSONL, JevBench public, and MASSIVE."""

from __future__ import annotations

from pathlib import Path

from omj.bench.suites.base import (
    DecisionItem,
    Downloader,
    answer_keys,
    cache_root,
    http_fetch,
    validate_item,
)
from omj.bench.suites.jevbench import load_jevbench_public
from omj.bench.suites.local import LOCAL_SUITES, SUITES_DIR, load_local, local_suite_path
from omj.bench.suites.massive import load_massive
from omj.errors import ErrorCode, OmjError

MASSIVE_SUITES: dict[str, str] = {"massive-ko": "ko", "massive-en": "en"}
JEVBENCH_SUITE = "jevbench-public"

SUITE_NAMES: tuple[str, ...] = (*LOCAL_SUITES, JEVBENCH_SUITE, *MASSIVE_SUITES)

__all__ = [
    "DecisionItem",
    "Downloader",
    "JEVBENCH_SUITE",
    "LOCAL_SUITES",
    "MASSIVE_SUITES",
    "SUITES_DIR",
    "SUITE_NAMES",
    "answer_keys",
    "cache_root",
    "http_fetch",
    "load_jevbench_public",
    "load_local",
    "load_massive",
    "load_suite",
    "local_suite_path",
    "validate_item",
]


def load_suite(
    name: str,
    *,
    cache_dir: Path | str | None = None,
    downloader: Downloader | None = None,
) -> list[DecisionItem]:
    """Load one benchmark suite by name, downloading external sets on first use."""
    if name in LOCAL_SUITES:
        return load_local(local_suite_path(name), name)
    if name == JEVBENCH_SUITE:
        return load_jevbench_public(cache_dir=cache_dir, downloader=downloader)
    if name in MASSIVE_SUITES:
        return load_massive(MASSIVE_SUITES[name], cache_dir=cache_dir, downloader=downloader)
    raise OmjError(
        ErrorCode.E_SCHEMA,
        f"unknown suite {name!r} (expected one of {', '.join(SUITE_NAMES)})",
    )
