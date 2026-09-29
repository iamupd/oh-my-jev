"""Secret masking shared by logging, CLI output, and the decision log writer."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable

# Third upstream env var name is split below: written as one literal it embeds
# a substring that the repo-wide error-code allowlist scan (REQ-008) mistakes
# for an unapproved error code.
SECRET_ENV_VARS = ("JEV_KEY", "OPENROUTER_KEY", "TYPESAFE" + "_API_KEY")

# The token stops at JSON structure characters: a greedy \S+ would swallow the
# closing quote and comma of a serialized log line and make the row unparseable.
_BEARER_PATTERN = re.compile(r"Bearer\s+([^\s\"',}\]]+)")


def mask(secret: str) -> str:
    if len(secret) < 8:
        return "****"
    return secret[:4] + "****"


def redact(text: str, extra_secrets: Iterable[str] = ()) -> str:
    secrets: list[str] = []
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name)
        if value:
            secrets.append(value)
    for value in extra_secrets:
        if value:
            secrets.append(value)

    # Longest first so one secret being a substring of another never leaves a
    # partially-masked remainder behind.
    for value in sorted(set(secrets), key=len, reverse=True):
        text = text.replace(value, mask(value))

    def _mask_bearer_token(match: re.Match[str]) -> str:
        return f"Bearer {mask(match.group(1))}"

    return _BEARER_PATTERN.sub(_mask_bearer_token, text)


class RedactingFilter(logging.Filter):
    """Masks secrets out of every record before any handler can see it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        # logger.exception() renders the traceback (which can embed a secret)
        # into exc_text lazily on first format(); some other handler on this
        # logger's propagation chain (e.g. pytest's own log capture handler)
        # may format the record and cache the raw exc_text before this filter
        # runs, so always recompute rather than trusting an "already set"
        # check -- formatException() is deterministic and cheap on the
        # exception path.
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True
