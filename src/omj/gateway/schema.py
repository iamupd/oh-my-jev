"""SystemOneRequest/Question schema, 422 error shape, and token budget check."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class SchemaError(Exception):
    """Wire-level error carrying the `{"error": {"type", "message"}}` response body."""

    def __init__(self, error_type: str, message: str, status: int = 422) -> None:
        self.error_type = error_type
        self.message = message
        self.status = status
        super().__init__(message)

    def to_body(self) -> dict:
        return {"error": {"type": self.error_type, "message": self.message}}


class Question(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: str | dict | list
    # A choice option's description may be any JSON value (a chess move object, a colour list, ...).
    criteria: dict[str, Any] | list[str] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> Question:
        if len(self.instructions) == 0:
            raise ValueError("instructions must be non-empty")

        if self.type == "noul":
            if self.criteria is not None:
                if not isinstance(self.criteria, dict):
                    raise ValueError("noul criteria must be null or an object with true/false keys")
                allowed = {"true", "false"}
                extra_keys = sorted(set(self.criteria) - allowed)
                if extra_keys:
                    raise ValueError(
                        f"noul criteria keys must be a subset of {sorted(allowed)}, got {extra_keys}"
                    )
        elif self.type == "choice":
            if not isinstance(self.criteria, dict):
                raise ValueError("choice criteria must be an object mapping option id to description")
            if not (2 <= len(self.criteria) <= 255):
                raise ValueError(
                    f"choice criteria must have between 2 and 255 entries, got {len(self.criteria)}"
                )
        elif self.type == "score":
            if not isinstance(self.criteria, list):
                raise ValueError("score criteria must be a list of level descriptions")
            if not (2 <= len(self.criteria) <= 10):
                raise ValueError(
                    f"score criteria must have between 2 and 10 entries, got {len(self.criteria)}"
                )
            for index, value in enumerate(self.criteria):
                if not isinstance(value, str):
                    raise ValueError(f"score criteria[{index}] must be a string")

        return self


class SystemOneRequest(BaseModel):
    state: str | dict | list
    model: str
    questions: dict[str, Question] = Field(min_length=1)

    @field_validator("questions", mode="after")
    @classmethod
    def _validate_question_ids(cls, value: dict[str, Question]) -> dict[str, Question]:
        for qid in value:
            if not qid.strip():
                raise ValueError("question ids must be non-empty strings")
        return value


def _dotted_path(loc: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in loc)


def parse_request(payload: dict) -> SystemOneRequest:
    try:
        return SystemOneRequest.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = _dotted_path(first["loc"])
        raise SchemaError("validation_error", f"{loc}: {first['msg']}") from exc


def check_token_budget(n_tokens: int, max_tokens: int) -> None:
    if n_tokens > max_tokens:
        raise SchemaError(
            "context_length_exceeded",
            f"request uses {n_tokens} tokens, exceeding the backend limit of {max_tokens} "
            "(maximum context length)",
        )


def answer_keys(q: Question) -> list[str]:
    if q.type == "noul":
        return ["yes", "no"]
    if q.type == "choice":
        if not isinstance(q.criteria, dict):
            raise ValueError("choice question must have dict criteria")
        return list(q.criteria.keys())
    if q.type == "score":
        if not isinstance(q.criteria, list):
            raise ValueError("score question must have list criteria")
        return [str(i) for i in range(len(q.criteria))]
    raise ValueError(f"unsupported question type: {q.type}")


def score_legend(q: Question) -> dict[str, str]:
    if q.type != "score" or not isinstance(q.criteria, list):
        raise ValueError("score_legend requires a score question with list criteria")
    return {str(i): level for i, level in enumerate(q.criteria)}
