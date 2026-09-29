"""Adapter that forwards SystemOne decisions to the hosted TypeSafe Jev API.

Supports the `typesafe` provider (direct) and the `openrouter` provider (proxied),
converting their SystemOneResponse answers into RawAnswer probability vectors.
"""

from __future__ import annotations

import os

import httpx

from omj.backends.base import Capabilities, Health, Kind, RawAnswer, answer_keys, render_state
from omj.config import BackendSection
from omj.errors import ErrorCode, OmjError

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "typesafe": {
        "base_url": "https://api.typesafe.ai",
        "model": "jev-latest",
        "key_env": "JEV_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api",
        "model": "typesafe/jev-1.13",
        "key_env": "OPENROUTER_KEY",
    },
}

_SYSTEMONE_PATH = "/v1/systemone"

_RESPONSE_KNOWN_FIELDS = {"model", "answers", "usage"}


def _extract_upstream_message(response: httpx.Response, status: int) -> str:
    try:
        data = response.json()
    except ValueError:
        return f"upstream {status}"
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])
        if "message" in data:
            return str(data["message"])
    return f"upstream {status}"


def _convert_answer(kind: Kind, criteria: dict | list | None, raw: dict) -> tuple[list[str], list[float]]:
    if kind == "noul":
        p = float(raw["noul"])
        return ["yes", "no"], [p, 1.0 - p]

    # choice/score both derive their key order from the request's own criteria
    # rather than upstream dict ordering, so results are stable regardless of
    # how the upstream server serializes its JSON object.
    keys = answer_keys(kind, criteria)
    probabilities = raw.get("probabilities") or {}
    probs = [float(probabilities.get(key, 0.0)) for key in keys]
    return keys, probs


class TypeSafeBackend:
    name = "typesafe"
    capabilities = Capabilities(
        max_options=255,
        max_state_tokens=65536,
        supports_batch=True,
        calibrated=True,
        device="remote",
    )

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client if client is not None else httpx.Client(timeout=30.0)
        self._provider = "typesafe"
        self._base_url = _PROVIDER_DEFAULTS["typesafe"]["base_url"]
        self._model = _PROVIDER_DEFAULTS["typesafe"]["model"]
        self._api_key = ""
        self.model_id = self._model

    def load(self, cfg: BackendSection) -> None:
        provider = cfg.provider
        defaults = _PROVIDER_DEFAULTS[provider]

        self._provider = provider
        self._base_url = defaults["base_url"]
        self._model = cfg.model if cfg.model else defaults["model"]
        self.model_id = self._model

        key_env = cfg.api_key_env if cfg.api_key_env else defaults["key_env"]
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise OmjError(
                ErrorCode.E_AUTH,
                f"missing API key: set environment variable {key_env}",
            )
        self._api_key = api_key

    def decide(self, state: str | dict | list, questions: dict[str, dict]) -> dict[str, RawAnswer]:
        if os.environ.get("OMJ_NO_NETWORK") == "1":
            raise OmjError(ErrorCode.E_NET, "network disabled (OMJ_NO_NETWORK=1)")

        url = f"{self._base_url}{_SYSTEMONE_PATH}"
        body = {"state": state, "model": self._model, "questions": questions}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            response = self._client.post(url, json=body, headers=headers)
        except httpx.RequestError as exc:
            raise OmjError(ErrorCode.E_NET, f"connection error: {exc}") from exc

        status = response.status_code
        if status in (401, 403):
            raise OmjError(ErrorCode.E_AUTH, f"upstream authentication failed (status {status})")
        if status == 422:
            raise OmjError(ErrorCode.E_BACKEND, _extract_upstream_message(response, status))
        if status == 429 or status >= 500:
            raise OmjError(ErrorCode.E_NET, f"upstream {status}")
        if status >= 400:
            raise OmjError(ErrorCode.E_BACKEND, f"upstream {status}")

        data = response.json()
        self.model_id = data.get("model", self._model)
        top_extra = {k: v for k, v in data.items() if k not in _RESPONSE_KNOWN_FIELDS}

        answers: dict[str, RawAnswer] = {}
        for qid, question in questions.items():
            kind: Kind = question["type"]
            criteria = question.get("criteria")
            raw_answer = data["answers"][qid]
            keys, probs = _convert_answer(kind, criteria, raw_answer)

            meta: dict = {
                "upstream_model": data.get("model"),
                "usage": data.get("usage"),
                "confidence": raw_answer.get("confidence"),
            }
            if top_extra:
                meta["upstream_extra"] = top_extra

            answers[qid] = RawAnswer(qid=qid, kind=kind, keys=keys, probs=probs, calibrated=True, meta=meta)

        return answers

    def health(self) -> Health:
        # Deliberately makes no network call: a real health probe would cost a
        # billed upstream request, so this only confirms the backend loaded.
        return Health(ok=True)

    def count_tokens(self, state: str | dict | list, questions: dict[str, dict]) -> int:
        total = len(render_state(state).split())
        for question in questions.values():
            total += len(render_state(question["instructions"]).split())
        return total
