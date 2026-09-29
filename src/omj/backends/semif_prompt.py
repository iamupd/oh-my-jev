"""Prompt rendering and option-letter labels for the semif backend.

Pure string work only: no torch or transformers import, so the label table and
the prompt body can be unit-tested without the `semif` extra installed.
"""

from __future__ import annotations

from omj.backends.base import Kind, answer_keys, render_state

MAX_LABELS = 255

ASSISTANT_PREFIX = "Answer:"

NOUL_TRUE_FALLBACK = "The statement is true."
NOUL_FALSE_FALLBACK = "The statement is false."

_ALPHABET = [chr(ord("A") + offset) for offset in range(26)]


def _build_labels(limit: int = MAX_LABELS) -> list[str]:
    labels = list(_ALPHABET)
    for prefix in _ALPHABET:
        for suffix in _ALPHABET:
            if len(labels) >= limit:
                return labels
            labels.append(prefix + suffix)
    return labels[:limit]


LABELS: list[str] = _build_labels()


def question_options(question: dict) -> list[tuple[str, str]]:
    """Return the (answer key, option text) pairs a question offers, in wire order."""
    kind: Kind = question["type"]
    criteria = question.get("criteria")

    if kind == "noul":
        # noul is rendered as a two-way choice so the readout path is the same for
        # every question kind; the gateway turns p(yes) back into the noul field.
        pairs = criteria if isinstance(criteria, dict) else {}
        return [
            ("yes", pairs.get("true") or NOUL_TRUE_FALLBACK),
            ("no", pairs.get("false") or NOUL_FALSE_FALLBACK),
        ]
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("choice criteria must be an object mapping option id to description")
        return [(key, text) for key, text in criteria.items()]
    if kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("score criteria must be a list of level descriptions")
        return [(str(index), text) for index, text in enumerate(criteria)]

    raise ValueError(f"unknown question kind: {kind!r}")


def render_question(
    state: str | dict | list,
    question: dict,
    labels: list[str] | None = None,
) -> tuple[str, list[str], list[str]]:
    """Render one question into its own prompt body.

    Returns the prompt body (no chat template applied), the answer keys in wire
    order, and the option labels used for them.
    """
    available = LABELS if labels is None else labels
    options = question_options(question)
    if len(options) > len(available):
        raise ValueError(
            f"question has {len(options)} options but only {len(available)} usable labels"
        )

    kind: Kind = question["type"]
    keys = answer_keys(kind, question.get("criteria") or {})
    labels_used = list(available[: len(options)])

    lines = [
        "State:",
        render_state(state),
        "",
        f"Question: {render_state(question['instructions'])}",
        "Options:",
    ]
    for label, (key, text) in zip(labels_used, options):
        lines.append(f"{label}. {key}: {text}")
    lines.append("Answer with the letter only.")

    return "\n".join(lines), keys, labels_used


def build_messages(prompt_body: str) -> list[dict]:
    """Chat-template input: a single user turn, no system message."""
    return [{"role": "user", "content": prompt_body}]
