"""Backend protocol and the small value types every adapter exchanges with the gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from omj.config import BackendSection

Kind = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class Capabilities:
    max_options: int
    max_state_tokens: int
    supports_batch: bool
    calibrated: bool
    device: str


@dataclass
class RawAnswer:
    qid: str
    kind: Kind
    keys: list[str]
    logits: list[float] | None = None
    probs: list[float] | None = None
    calibrated: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # `assert` is stripped when Python runs with -O, which would let an
        # invalid RawAnswer silently reach the gateway instead of failing
        # fast; raise a real exception so this invariant always holds.
        has_logits = self.logits is not None
        has_probs = self.probs is not None
        if has_logits == has_probs:
            raise ValueError("RawAnswer requires exactly one of logits or probs")
        values = self.logits if has_logits else self.probs
        if len(values) != len(self.keys):
            raise ValueError("RawAnswer values length must match keys length")
        if self.kind == "noul" and self.keys != ["yes", "no"]:
            raise ValueError('noul RawAnswer keys must be ["yes", "no"]')


@dataclass(frozen=True)
class Health:
    ok: bool
    detail: str = ""


@runtime_checkable
class Backend(Protocol):
    name: str
    model_id: str
    capabilities: Capabilities

    def load(self, cfg: BackendSection) -> None: ...

    def decide(self, state: str | dict | list, questions: dict[str, dict]) -> dict[str, RawAnswer]: ...

    def health(self) -> Health: ...

    def count_tokens(self, state: str | dict | list, questions: dict[str, dict]) -> int: ...


def answer_keys(kind: Kind, criteria: dict | list) -> list[str]:
    if kind == "noul":
        return ["yes", "no"]
    if kind == "choice":
        return list(criteria.keys())
    if kind == "score":
        return [str(i) for i in range(len(criteria))]
    raise ValueError(f"unknown question kind: {kind!r}")


def render_state(state: str | dict | list) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=False)
