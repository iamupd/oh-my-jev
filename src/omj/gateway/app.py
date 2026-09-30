"""FastAPI gateway: TypeSafe wire routes over any Backend, with auth, limits, and logging."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from starlette.background import BackgroundTask

from omj.backends.base import Backend
from omj.config import Config
from omj.errors import ErrorCode, OmjError
from omj.gateway.assemble import assemble_answer, softmax
from omj.gateway.calibration import Calibration, apply_temperature, temperature_for
from omj.gateway.decision_log import DecisionLogger, DecisionLogRow, state_sha256
from omj.gateway.schema import SchemaError, Question, answer_keys, check_token_budget, parse_request
from omj.gateway.usage import compute_usage
from omj.logging_setup import setup_logging
from omj import __version__

logger = logging.getLogger("omj.gateway")

SYSTEMONE_PATHS = ("/v1/systemone", "/api/jev", "/v1/inference")
MODEL_ALIASES = ["jev-latest"]
OVERLOADED_STATUS = 529
RETRY_AFTER_HEADERS = {"Retry-After": "2"}


def _error_body(error_type: str, message: str) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": message}}


@dataclass
class GatewayResult:
    """Transport-agnostic handler outcome, so the hot path can be timed without HTTP."""

    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    log_row: DecisionLogRow | None = None


def _bearer_matches(authorization: str | None, api_key: str) -> bool:
    if not authorization:
        return False
    scheme, _, token = authorization.partition(" ")
    return scheme.lower() == "bearer" and token.strip() == api_key


def create_app(
    backend: Backend,
    config: Config,
    *,
    calibration: Calibration | None = None,
    decision_logger: DecisionLogger | None = None,
) -> FastAPI:
    setup_logging()

    started_at = time.monotonic()
    api_key = config.serve.api_key
    usage_mode = config.usage.output_tokens_mode
    max_concurrency = max(1, config.serve.max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)

    if decision_logger is None:
        decision_logger = DecisionLogger(
            log_dir=config.log.dir or None,
            include_state=config.log.state,
        )

    def _overloaded(message: str) -> GatewayResult:
        return GatewayResult(
            OVERLOADED_STATUS,
            _error_body("overloaded_error", message),
            dict(RETRY_AFTER_HEADERS),
        )

    def _backend_failure(exc: Exception, fallback: str) -> GatewayResult:
        # An upstream 4xx is the caller's problem, not server overload, so it
        # must not be answered with 529 + Retry-After (REQ-018).
        if isinstance(exc, OmjError):
            if exc.code is ErrorCode.E_AUTH:
                # Never echo the upstream text: it can carry key material.
                return GatewayResult(
                    401, _error_body("authentication_error", "upstream authentication failed")
                )
            if exc.code is ErrorCode.E_BACKEND:
                return GatewayResult(422, _error_body("validation_error", exc.message))
        return _overloaded(fallback)

    async def handle_systemone(body: bytes, authorization: str | None = None) -> GatewayResult:
        if api_key and not _bearer_matches(authorization, api_key):
            return GatewayResult(
                401,
                _error_body(
                    "authentication_error",
                    "missing or invalid Authorization bearer token",
                ),
            )

        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return GatewayResult(
                422, _error_body("validation_error", "request body is not valid JSON")
            )

        try:
            request = parse_request(payload)
        except SchemaError as exc:
            return GatewayResult(exc.status, exc.to_body())

        questions = {qid: q.model_dump() for qid, q in request.questions.items()}

        try:
            # Off the event loop: a backend may tokenize or call upstream here.
            input_tokens = await run_in_threadpool(backend.count_tokens, request.state, questions)
            check_token_budget(input_tokens, backend.capabilities.max_state_tokens)
        except SchemaError as exc:
            return GatewayResult(exc.status, exc.to_body())
        except Exception as exc:
            logger.error("backend %s failed to count tokens: %s", backend.name, exc, exc_info=True)
            return _backend_failure(
                exc, f"backend {backend.name} could not measure the request"
            )

        # Refuse rather than queue: a caller waiting behind a full pool is worse
        # than a 429 it can retry against another replica.
        if semaphore.locked():
            return GatewayResult(
                429,
                _error_body(
                    "rate_limit_error",
                    f"server is at its concurrency limit of {max_concurrency}",
                ),
            )

        await semaphore.acquire()
        try:
            # health() can block on a remote probe, which would freeze the loop
            # for every other request while this one holds the permit.
            health = await run_in_threadpool(backend.health)
            if not health.ok:
                detail = health.detail or "backend reported an unhealthy state"
                logger.error("backend %s health check failed: %s", backend.name, detail)
                return _overloaded(detail)

            started = time.perf_counter()
            raw_answers = await run_in_threadpool(backend.decide, request.state, questions)
            latency_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            logger.error("backend %s failed to decide: %s", backend.name, exc, exc_info=True)
            return _backend_failure(exc, f"backend {backend.name} failed to produce a decision")
        finally:
            semaphore.release()

        try:
            answers, upstream_extra, calibrated = _assemble(
                request.questions, raw_answers, calibration
            )
        except Exception as exc:
            logger.error("backend %s returned an unusable answer: %s", backend.name, exc, exc_info=True)
            return _overloaded(f"backend {backend.name} returned an unusable answer")

        row = DecisionLogRow(
            ts=datetime.now(UTC).isoformat(),
            request_id=str(uuid.uuid4()),
            backend=backend.name,
            model=backend.model_id,
            state_sha256=state_sha256(request.state),
            questions=questions,
            answers=answers,
            latency_ms=latency_ms,
            calibrated=calibrated,
            upstream_extra=upstream_extra,
            state=request.state,
        )

        return GatewayResult(
            200,
            {
                "model": backend.model_id,
                "answers": answers,
                "usage": compute_usage(input_tokens, answers, usage_mode),
            },
            {
                "x-omj-backend": backend.name,
                "x-omj-model": backend.model_id,
                "x-omj-calibrated": "true" if calibrated else "false",
                "x-omj-latency-ms": f"{latency_ms:.1f}",
            },
            row,
        )

    def _assemble(
        questions: dict[str, Question],
        raw_answers: dict[str, Any],
        cal: Calibration | None,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        answers: dict[str, Any] = {}
        upstream_extra: dict[str, Any] = {}
        applied_temperature = False
        all_precalibrated = True

        for qid, question in questions.items():
            raw = raw_answers.get(qid)
            if raw is None:
                raise ValueError(f"no answer for question {qid!r}")
            if not raw.calibrated:
                all_precalibrated = False

            keys = raw.keys or answer_keys(question)
            if raw.logits is not None:
                logits = raw.logits
                if cal is not None and not raw.calibrated:
                    logits = apply_temperature(logits, temperature_for(cal, raw.kind))
                    applied_temperature = True
                probs = softmax(logits)
            else:
                if cal is not None and not raw.calibrated:
                    logger.warning(
                        "calibration is configured but backend %s returned probabilities for %r; "
                        "temperature scaling was skipped",
                        backend.name,
                        qid,
                    )
                probs = list(raw.probs or [])

            answers[qid] = assemble_answer(question, keys, probs)
            if raw.meta:
                upstream_extra.update(raw.meta)

        return answers, upstream_extra, applied_temperature or all_precalibrated

    def _write_decision_log(row: DecisionLogRow) -> None:
        # Runs after the response is sent, so a log failure must never surface
        # to the caller as an ASGI error.
        try:
            decision_logger.write(row)
        except Exception as exc:
            logger.error("failed to write decision log row %s: %s", row.request_id, exc)

    async def systemone_route(request: Request) -> Response:
        result = await handle_systemone(
            await request.body(), request.headers.get("authorization")
        )
        background = (
            BackgroundTask(_write_decision_log, result.log_row)
            if result.log_row is not None
            else None
        )
        return JSONResponse(
            result.body,
            status_code=result.status,
            headers=result.headers,
            background=background,
        )

    async def health_route() -> Response:
        try:
            health = await run_in_threadpool(backend.health)
            ok, detail = health.ok, health.detail
        except Exception as exc:
            ok, detail = False, str(exc)

        body = {
            "status": "ok" if ok else "degraded",
            "backend": backend.name,
            "model": backend.model_id,
            "calibrated": calibration is not None or backend.capabilities.calibrated,
            "uptime_s": round(time.monotonic() - started_at, 3),
        }
        if ok:
            return JSONResponse(body)

        logger.error("backend %s health check failed: %s", backend.name, detail)
        return JSONResponse(body, status_code=OVERLOADED_STATUS, headers=dict(RETRY_AFTER_HEADERS))

    async def models_route() -> Response:
        return JSONResponse(
            {
                "models": [
                    {
                        "id": backend.model_id,
                        "backend": backend.name,
                        "aliases": list(MODEL_ALIASES),
                    }
                ]
            }
        )

    app = FastAPI(title="oh-my-jev gateway", version=__version__)
    for index, path in enumerate(SYSTEMONE_PATHS):
        app.add_api_route(
            path,
            systemone_route,
            methods=["POST"],
            name=f"systemone_{index}",
            include_in_schema=index == 0,
        )
    app.add_api_route("/health", health_route, methods=["GET"], name="health")
    app.add_api_route("/v1/models", models_route, methods=["GET"], name="models")

    app.state.backend = backend
    app.state.config = config
    app.state.calibration = calibration
    app.state.decision_logger = decision_logger
    app.state.semaphore = semaphore
    app.state.handle_systemone = handle_systemone
    return app
