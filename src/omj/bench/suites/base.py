"""DecisionItem, its schema validation, and the shared download plumbing."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from omj.config import omj_home
from omj.errors import ErrorCode, OmjError

QUESTION_TYPES = ("noul", "choice", "score")
NOUL_ANSWER_KEYS = ("yes", "no")
EXPECTED_SHAPES = ("uniform", "unknown", "conflict")

MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

HTTP_TIMEOUT_SECONDS = 60.0

#: A downloader turns a URL into the raw bytes at that URL. Loaders take one so
#: that tests can serve fixtures without touching the network.
Downloader = Callable[[str], bytes]


@dataclass
class DecisionItem:
    """One benchmark row: a state plus the questions to ask about it."""

    id: str
    suite: str
    state: str | dict | list
    questions: dict[str, dict]
    expected: dict[str, str] | None
    tags: list[str] = field(default_factory=list)
    expected_shape: str | None = None
    pair_id: str | None = None
    source: str = ""
    license: str = ""


def _fail(message: str) -> OmjError:
    return OmjError(ErrorCode.E_SCHEMA, message)


def answer_keys(question: dict[str, Any]) -> list[str]:
    """The answer keys a backend may return for one question, derived from its criteria."""
    kind = question.get("type")
    if kind == "noul":
        return list(NOUL_ANSWER_KEYS)
    if kind == "choice":
        return list(question.get("criteria") or {})
    if kind == "score":
        return [str(i) for i in range(len(question.get("criteria") or []))]
    raise _fail(f"unknown question type {kind!r}")


def _validate_question(item_id: str, qid: str, question: Any) -> None:
    if not isinstance(question, dict):
        raise _fail(f"{item_id}: question {qid!r} must be an object")

    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        raise _fail(f"{item_id}: question {qid!r} has unknown type {kind!r} (expected one of {', '.join(QUESTION_TYPES)})")

    if not question.get("instructions"):
        raise _fail(f"{item_id}: question {qid!r} has no instructions")

    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise _fail(f"{item_id}: question {qid!r} (noul) criteria must be an object")
    elif kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < MIN_CHOICE_OPTIONS:
            raise _fail(
                f"{item_id}: question {qid!r} (choice) criteria must be an object with at least "
                f"{MIN_CHOICE_OPTIONS} keys"
            )
    elif kind == "score":
        if not isinstance(criteria, list) or not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise _fail(
                f"{item_id}: question {qid!r} (score) criteria must be a list of "
                f"{MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels"
            )


def validate_item(item: DecisionItem) -> None:
    """Raise OmjError(E_SCHEMA) unless the item is a well-formed decision row."""
    if not item.id:
        raise _fail(f"suite {item.suite!r}: an item has an empty id")

    if not isinstance(item.questions, dict) or not item.questions:
        raise _fail(f"{item.id}: questions must be a non-empty object")

    for qid, question in item.questions.items():
        _validate_question(item.id, qid, question)

    if item.expected is not None:
        if not isinstance(item.expected, dict):
            raise _fail(f"{item.id}: expected must be an object keyed by question id")
        for qid, label in item.expected.items():
            question = item.questions.get(qid)
            if question is None:
                raise _fail(f"{item.id}: expected refers to unknown question {qid!r}")
            keys = answer_keys(question)
            if label not in keys:
                raise _fail(
                    f"{item.id}: expected[{qid!r}] = {label!r} is not an answer key ({', '.join(keys)})"
                )

    if item.expected_shape is not None and item.expected_shape not in EXPECTED_SHAPES:
        raise _fail(
            f"{item.id}: expected_shape {item.expected_shape!r} is not one of {', '.join(EXPECTED_SHAPES)}"
        )


def cache_root(cache_dir: Path | str | None = None) -> Path:
    """Where downloaded suites are cached: the override, else $OMJ_HOME/cache."""
    if cache_dir is not None:
        return Path(cache_dir)
    return omj_home() / "cache"


def http_fetch(url: str) -> bytes:
    """Default Downloader: GET the URL and return its body."""
    if os.environ.get("OMJ_NO_NETWORK") == "1":
        raise OmjError(ErrorCode.E_DOWNLOAD, "network disabled (OMJ_NO_NETWORK=1)")

    try:
        response = httpx.get(url, timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OmjError(ErrorCode.E_NET, f"{url}: {exc}") from exc
    return response.content


def resolve_downloader(downloader: Downloader | None) -> Downloader:
    return http_fetch if downloader is None else downloader
