"""Deterministic keyword-overlap backend: no GPU or network required.

Scores each answer option by how many of its own distinct keyword tokens
appear in the state text. This is intentionally crude (not a language model)
so that unit tests and CLI smoke checks run anywhere without downloads.
"""

from __future__ import annotations

import re

from omj.backends.base import Capabilities, Health, Kind, RawAnswer, answer_keys, render_state
from omj.config import BackendSection

_TOKEN_RE = re.compile(r"\w+")

# Small closed-class English function words; excluding them keeps overlap
# scoring focused on content words instead of grammar glue.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to",
    "of", "in", "on", "at", "by", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "as", "with",
    "from", "into", "about", "not", "no", "so", "do", "does", "did", "has",
    "have", "had", "you", "your", "my", "me", "we", "our", "they", "their",
    "he", "she", "his", "her", "him", "will", "would", "can", "could",
    "should", "just", "please", "now", "all", "any", "one", "out", "up",
    "only", "here", "some", "very",
}


def _tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    }


def _option_text(kind: Kind, key: str, index: int, criteria: dict | list) -> str:
    if kind == "noul":
        criteria_dict = criteria or {}
        if key == "yes":
            desc = criteria_dict.get("true") or "true correct"
            return f"yes {desc}"
        desc = criteria_dict.get("false") or "not never"
        return f"no {desc}"
    if kind == "choice":
        desc = criteria[key]
        key_part = re.sub(r"[_-]+", " ", key)
        return f"{key_part} {desc}"
    if kind == "score":
        return str(criteria[index])
    raise ValueError(f"unknown question kind: {kind!r}")


class MockBackend:
    name = "mock"
    model_id = "mock-keyword-v1"
    capabilities = Capabilities(
        max_options=255,
        max_state_tokens=65536,
        supports_batch=True,
        calibrated=False,
        device="cpu",
    )

    def load(self, cfg: BackendSection) -> None:
        return None

    def decide(self, state: str | dict | list, questions: dict[str, dict]) -> dict[str, RawAnswer]:
        state_tokens = _tokenize(render_state(state))
        answers: dict[str, RawAnswer] = {}
        for qid, question in questions.items():
            kind: Kind = question["type"]
            criteria = question.get("criteria") or {}
            keys = answer_keys(kind, criteria)
            logits = []
            for index, key in enumerate(keys):
                option_tokens = _tokenize(_option_text(kind, key, index, criteria))
                logits.append(float(len(option_tokens & state_tokens)))
            answers[qid] = RawAnswer(qid=qid, kind=kind, keys=keys, logits=logits)
        return answers

    def health(self) -> Health:
        return Health(ok=True)

    def count_tokens(self, state: str | dict | list, questions: dict[str, dict]) -> int:
        total = len(render_state(state).split())
        for question in questions.values():
            total += len(render_state(question["instructions"]).split())
        return total
