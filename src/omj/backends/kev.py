"""KevBackend: HTTP proxy to local kev server at /v1/systemone."""

from __future__ import annotations

import json
import os

import httpx

from omj.backends.base import Capabilities, Health, Kind, RawAnswer, render_state
from omj.config import BackendSection
from omj.errors import ErrorCode, OmjError


def _require_probabilities(qid: str, upstream_answer: dict) -> dict:
    probabilities = upstream_answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise OmjError(
            ErrorCode.E_BACKEND, f"upstream answer missing 'probabilities' for {qid!r}"
        )
    return probabilities


class KevBackend:
    name = "kev"
    model_id: str = "kev-latest"
    capabilities = Capabilities(
        max_options=255,
        max_state_tokens=32768,
        supports_batch=True,
        calibrated=True,
        device="remote",
    )

    def __init__(self) -> None:
        self._base_url: str = "http://127.0.0.1:8009"
        self._http_client: httpx.Client | None = None

    def load(self, cfg: BackendSection, http_client: httpx.Client | None = None) -> None:
        base_url = cfg.kev_base_url.rstrip("/")
        self._base_url = base_url
        self._http_client = http_client or httpx.Client(timeout=60.0)
        if cfg.model:
            self.model_id = cfg.model
        else:
            self.model_id = "kev-latest"

    def decide(self, state: str | dict | list, questions: dict[str, dict]) -> dict[str, RawAnswer]:
        if os.environ.get("OMJ_NO_NETWORK") == "1":
            raise OmjError(ErrorCode.E_NET, "network disabled (OMJ_NO_NETWORK=1)")

        state_str = render_state(state)
        request_body = {
            "state": state,
            "model": self.model_id,
            "questions": questions,
        }

        try:
            response = self._http_client.post(
                f"{self._base_url}/v1/systemone",
                json=request_body,
            )
        except httpx.ConnectError as exc:
            raise OmjError(ErrorCode.E_NET, f"connection error: {exc}")

        if 400 <= response.status_code < 500:
            detail = ""
            try:
                data = response.json()
                detail = data.get("detail", "")
            except Exception:
                pass
            raise OmjError(ErrorCode.E_BACKEND, f"upstream error {response.status_code}: {detail}")

        if response.status_code >= 500:
            raise OmjError(ErrorCode.E_NET, f"upstream server error {response.status_code}")

        if response.status_code != 200:
            raise OmjError(ErrorCode.E_NET, f"unexpected status code {response.status_code}")

        data = response.json()

        if "model" in data:
            self.model_id = data["model"]

        answers: dict[str, RawAnswer] = {}
        upstream_model = data.get("model", "unknown")
        latency_ms = data.get("latency_ms", 0)
        upstream_answers = data.get("answers")
        if not isinstance(upstream_answers, dict):
            upstream_answers = {}

        for qid, question in questions.items():
            kind: Kind = question["type"]
            criteria = question.get("criteria") or {}
            upstream_answer = upstream_answers.get(qid)
            # A missing or malformed answer must surface as an error: filling it
            # in would hand the caller a fabricated, "calibrated" distribution.
            if not isinstance(upstream_answer, dict):
                raise OmjError(ErrorCode.E_BACKEND, f"upstream answer missing for {qid!r}")

            if kind == "noul":
                if "noul" not in upstream_answer:
                    raise OmjError(
                        ErrorCode.E_BACKEND, f"upstream answer missing 'noul' for {qid!r}"
                    )
                p = upstream_answer["noul"]
                answers[qid] = RawAnswer(
                    qid=qid,
                    kind="noul",
                    keys=["yes", "no"],
                    probs=[p, 1.0 - p],
                    calibrated=True,
                    meta={
                        "upstream_model": upstream_model,
                        "latency_ms": latency_ms,
                    },
                )

            elif kind == "choice":
                keys = list(criteria.keys())
                upstream_probs = _require_probabilities(qid, upstream_answer)
                probs = [upstream_probs.get(k, 0.0) for k in keys]
                confidence = upstream_answer.get("confidence", 0.0)
                answers[qid] = RawAnswer(
                    qid=qid,
                    kind="choice",
                    keys=keys,
                    probs=probs,
                    calibrated=True,
                    meta={
                        "upstream_model": upstream_model,
                        "latency_ms": latency_ms,
                        "confidence": confidence,
                    },
                )

            elif kind == "score":
                keys = [str(i) for i in range(len(criteria))]
                upstream_probs = _require_probabilities(qid, upstream_answer)
                probs = [upstream_probs.get(k, 0.0) for k in keys]
                confidence = upstream_answer.get("confidence", 0.0)
                answers[qid] = RawAnswer(
                    qid=qid,
                    kind="score",
                    keys=keys,
                    probs=probs,
                    calibrated=True,
                    meta={
                        "upstream_model": upstream_model,
                        "latency_ms": latency_ms,
                        "confidence": confidence,
                    },
                )

        return answers

    def health(self) -> Health:
        if os.environ.get("OMJ_NO_NETWORK") == "1":
            return Health(ok=False, detail="network disabled (OMJ_NO_NETWORK=1)")

        try:
            response = self._http_client.get(f"{self._base_url}/health")
            if response.status_code == 200:
                return Health(ok=True)
            return Health(ok=False, detail=f"health check returned {response.status_code}")
        except Exception as exc:
            return Health(ok=False, detail=f"health check failed: {exc}")

    def count_tokens(self, state: str | dict | list, questions: dict[str, dict]) -> int:
        total = len(render_state(state).split())
        for question in questions.values():
            total += len(render_state(question["instructions"]).split())
        return total
