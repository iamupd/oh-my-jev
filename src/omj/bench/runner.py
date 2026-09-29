"""Bench execution: send every decision item to a target and collect ResultRows (design.md 1; REQ-032).

Two targets are supported. `BackendTarget` calls a loaded `Backend` in-process,
applying the same calibration/softmax path the gateway uses, and `EndpointTarget`
posts the TypeSafe wire request to an already running gateway (this one or a
vendor's). `run_suite` is transport-agnostic: it times each item, turns failures
into error rows, and applies the JevBench house stop rules.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

from omj.backends.base import Backend
from omj.bench.suites.base import DecisionItem, answer_keys
from omj.bench.types import ResultRow, is_correct
from omj.errors import ErrorCode, OmjError
from omj.gateway.assemble import softmax
from omj.gateway.calibration import Calibration, apply_temperature, temperature_for

#: One parsed answer: (kind, keys, probs, logits or None, meta).
Decision = tuple[str, list[str], list[float], list[float] | None, dict[str, Any]]

DEFAULT_MODEL_ALIAS = "jev-latest"
ENDPOINT_PATH = "/v1/systemone"
HTTP_TIMEOUT_SECONDS = 120.0
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
#: House rule: a run gives up after this many infrastructure failures in a row.
MAX_CONSECUTIVE_NET_ERRORS = 3


def _wire_questions(item: DecisionItem) -> dict[str, dict]:
    """Questions in the shape the gateway hands a backend: `criteria` is always present.

    Suite rows may omit `criteria` for noul questions, while the gateway always dumps
    the full Question model, so backends are entitled to read the key unconditionally.
    """
    return {
        qid: ({**question, "criteria": None} if "criteria" not in question else question)
        for qid, question in item.questions.items()
    }


@runtime_checkable
class Target(Protocol):
    """Anything that can answer one decision item's questions."""

    def decide_item(self, item: DecisionItem) -> dict[str, Decision]: ...


@dataclass
class RunResult:
    rows: list[ResultRow]
    wall_seconds: float
    input_tokens_total: int
    n_errors: int
    stopped_early: bool = False
    stop_reason: str = ""


class BackendTarget:
    """In-process target: `Backend.decide` -> temperature -> softmax."""

    def __init__(self, backend: Backend, calibration: Calibration | None = None) -> None:
        self.backend = backend
        self.calibration = calibration

    def decide_item(self, item: DecisionItem) -> dict[str, Decision]:
        raw_answers = self.backend.decide(item.state, _wire_questions(item))
        decisions: dict[str, Decision] = {}
        for qid, question in item.questions.items():
            raw = raw_answers.get(qid)
            if raw is None:
                raise OmjError(
                    ErrorCode.E_BACKEND,
                    f"backend {self.backend.name} returned no answer for question {qid!r}",
                )

            keys = list(raw.keys) if raw.keys else answer_keys(question)
            logits = list(raw.logits) if raw.logits is not None else None
            if logits is not None:
                scaled = logits
                if self.calibration is not None and not raw.calibrated:
                    scaled = apply_temperature(logits, temperature_for(self.calibration, raw.kind))
                probs = softmax(scaled)
            else:
                probs = [float(p) for p in (raw.probs or [])]
            decisions[qid] = (raw.kind, keys, probs, logits, dict(raw.meta))
        return decisions

    def input_tokens(self, item: DecisionItem) -> int | None:
        """Prompt tokens the backend counts for this item, or None when it cannot say."""
        try:
            return int(self.backend.count_tokens(item.state, item.questions))
        except Exception:
            return None


def _is_loopback(url: str) -> bool:
    host = urlparse(url).hostname
    return host is not None and host.lower() in LOOPBACK_HOSTS


class EndpointTarget:
    """HTTP target: POST the TypeSafe request to `{base_url}/v1/systemone`."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        *,
        model: str = DEFAULT_MODEL_ALIAS,
        timeout: float = HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._last_input_tokens: int | None = None

        # A loopback endpoint is the local gateway, not the internet, so the
        # test-suite network block does not apply to it.
        if os.environ.get("OMJ_NO_NETWORK") == "1" and not _is_loopback(self.base_url):
            raise OmjError(
                ErrorCode.E_NET,
                f"network disabled (OMJ_NO_NETWORK=1): refusing to call {self.base_url}",
            )

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    def decide_item(self, item: DecisionItem) -> dict[str, Decision]:
        url = f"{self.base_url}{ENDPOINT_PATH}"
        payload = {"state": item.state, "model": self.model, "questions": _wire_questions(item)}
        try:
            response = self.client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise OmjError(ErrorCode.E_NET, f"{url}: {exc}") from exc

        self._raise_for_status(url, response)

        try:
            body = response.json()
            answers = body["answers"]
            usage = body.get("usage") or {}
        except Exception as exc:
            raise OmjError(ErrorCode.E_SCHEMA, f"{url}: unreadable response body: {exc}") from exc

        self._last_input_tokens = usage.get("input_tokens")
        return {
            qid: _parse_endpoint_answer(item.id, qid, question, answers.get(qid))
            for qid, question in item.questions.items()
        }

    def input_tokens(self, item: DecisionItem) -> int | None:
        """Prompt tokens reported by the endpoint for the item just decided."""
        return self._last_input_tokens

    @staticmethod
    def _raise_for_status(url: str, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        if status in (401, 403):
            raise OmjError(ErrorCode.E_AUTH, f"{url}: endpoint rejected the credentials (HTTP {status})")
        if status == 422:
            raise OmjError(ErrorCode.E_SCHEMA, f"{url}: endpoint rejected the request (HTTP 422)")
        raise OmjError(ErrorCode.E_NET, f"{url}: endpoint returned HTTP {status}")


def _probs_in_key_order(item_id: str, qid: str, keys: list[str], probabilities: Any) -> list[float]:
    if not isinstance(probabilities, dict):
        raise OmjError(
            ErrorCode.E_SCHEMA,
            f"{item_id}/{qid}: answer has no probabilities object",
        )
    missing = [key for key in keys if key not in probabilities]
    if missing:
        raise OmjError(
            ErrorCode.E_SCHEMA,
            f"{item_id}/{qid}: answer is missing probabilities for {', '.join(missing)}",
        )
    return [float(probabilities[key]) for key in keys]


def _parse_endpoint_answer(item_id: str, qid: str, question: dict, answer: Any) -> Decision:
    if not isinstance(answer, dict):
        raise OmjError(ErrorCode.E_SCHEMA, f"{item_id}/{qid}: endpoint returned no answer")

    kind = question["type"]
    keys = answer_keys(question)
    if kind == "noul":
        try:
            p_yes = float(answer["noul"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OmjError(ErrorCode.E_SCHEMA, f"{item_id}/{qid}: answer has no noul value") from exc
        probs = [p_yes, 1.0 - p_yes]
    else:
        probs = _probs_in_key_order(item_id, qid, keys, answer.get("probabilities"))

    meta: dict[str, Any] = {}
    if "confidence" in answer:
        meta["confidence"] = answer["confidence"]
    return kind, keys, probs, None, meta


def _keys_for(question: dict) -> list[str]:
    try:
        return answer_keys(question)
    except Exception:
        return []


def _error_rows(
    item: DecisionItem,
    message: str,
    latency_ms: float,
    *,
    suite_name: str,
    backend_name: str,
    model: str,
) -> list[ResultRow]:
    expected = item.expected or {}
    rows: list[ResultRow] = []
    for qid, question in item.questions.items():
        rows.append(
            ResultRow(
                item_id=item.id,
                suite=suite_name,
                qid=qid,
                kind=question.get("type", "choice"),
                keys=_keys_for(question),
                probs=None,
                logits=None,
                expected=expected.get(qid),
                correct=None,
                latency_ms=latency_ms,
                cost_usd=None,
                backend=backend_name,
                model=model,
                tags=list(item.tags),
                expected_shape=item.expected_shape,
                pair_id=item.pair_id,
                error=message,
            )
        )
    return rows


def _result_rows(
    item: DecisionItem,
    decisions: dict[str, Decision],
    latency_ms: float,
    *,
    suite_name: str,
    backend_name: str,
    model: str,
) -> list[ResultRow]:
    expected = item.expected or {}
    rows: list[ResultRow] = []
    for qid in item.questions:
        decision = decisions.get(qid)
        if decision is None:
            raise OmjError(ErrorCode.E_BACKEND, f"{item.id}: no answer for question {qid!r}")
        kind, keys, probs, logits, _meta = decision
        row = ResultRow(
            item_id=item.id,
            suite=suite_name,
            qid=qid,
            kind=kind,
            keys=list(keys),
            probs=list(probs),
            logits=list(logits) if logits is not None else None,
            expected=expected.get(qid),
            correct=None,
            latency_ms=latency_ms,
            cost_usd=None,
            backend=backend_name,
            model=model,
            tags=list(item.tags),
            expected_shape=item.expected_shape,
            pair_id=item.pair_id,
        )
        row.correct = is_correct(row)
        rows.append(row)
    return rows


def _input_tokens_for(target: Any, item: DecisionItem) -> int:
    counter = getattr(target, "input_tokens", None)
    if counter is None:
        return 0
    try:
        value = counter(item)
    except Exception:
        return 0
    return int(value) if value else 0


def run_suite(
    items: list[DecisionItem],
    target: Target,
    *,
    suite_name: str,
    backend_name: str,
    model: str,
    stop_on_auth: bool = True,
    on_item: Callable[[bool], None] | None = None,
) -> RunResult:
    """Send every item to `target`, timing each one and recording failures as error rows.

    `on_item(ok)` is called after every attempted item (e.g. to drive a progress bar).

    The run stops early (leaving the remaining items unattempted, i.e. absent from
    `rows`) on an E_AUTH failure or after MAX_CONSECUTIVE_NET_ERRORS E_NET failures
    in a row, so a bad key or a dead upstream cannot burn a whole suite.
    """
    rows: list[ResultRow] = []
    input_tokens_total = 0
    n_errors = 0
    consecutive_net_errors = 0
    stopped_early = False
    stop_reason = ""

    run_started = time.perf_counter()
    for item in items:
        item_started = time.perf_counter()
        try:
            decisions = target.decide_item(item)
            latency_ms = (time.perf_counter() - item_started) * 1000.0
            item_rows = _result_rows(
                item,
                decisions,
                latency_ms,
                suite_name=suite_name,
                backend_name=backend_name,
                model=model,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - item_started) * 1000.0
            rows.extend(
                _error_rows(
                    item,
                    str(exc),
                    latency_ms,
                    suite_name=suite_name,
                    backend_name=backend_name,
                    model=model,
                )
            )
            n_errors += 1
            if on_item is not None:
                on_item(False)

            code = getattr(exc, "code", None)
            if code is ErrorCode.E_AUTH and stop_on_auth:
                stopped_early = True
                stop_reason = "E_AUTH"
                break
            if code is ErrorCode.E_NET:
                consecutive_net_errors += 1
                if consecutive_net_errors >= MAX_CONSECUTIVE_NET_ERRORS:
                    stopped_early = True
                    stop_reason = f"{MAX_CONSECUTIVE_NET_ERRORS} consecutive E_NET errors"
                    break
            else:
                consecutive_net_errors = 0
            continue

        consecutive_net_errors = 0
        rows.extend(item_rows)
        input_tokens_total += _input_tokens_for(target, item)
        if on_item is not None:
            on_item(True)

    return RunResult(
        rows=rows,
        wall_seconds=time.perf_counter() - run_started,
        input_tokens_total=input_tokens_total,
        n_errors=n_errors,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )


def order_groups_from_rows(rows: list[ResultRow]) -> dict[str, list[ResultRow]]:
    """Group order-suite rows by their originating item/question, for order_shift_max."""
    groups: dict[str, list[ResultRow]] = {}
    for row in rows:
        if row.pair_id is None or row.probs is None:
            continue
        groups.setdefault(f"{row.pair_id}::{row.qid}", []).append(row)
    return groups


__all__ = [
    "BackendTarget",
    "Decision",
    "EndpointTarget",
    "MAX_CONSECUTIVE_NET_ERRORS",
    "RunResult",
    "Target",
    "order_groups_from_rows",
    "run_suite",
]
