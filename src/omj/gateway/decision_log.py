"""Append-only JSONL decision log: daily files, sha256'd state, opt-in state payload."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from omj.config import omj_home
from omj.redaction import redact


def redact_value(value: Any) -> Any:
    """Redact every string inside a nested payload before it is serialized.

    Masking only the serialized line would have to survive JSON quoting; doing
    it per field keeps the emitted row valid JSON no matter what a secret
    pattern matches.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {redact_value(k): redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item) for item in value]
    return value


def state_sha256(state: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(state, str):
        payload = state
    else:
        payload = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class DecisionLogRow:
    ts: str
    request_id: str
    backend: str
    model: str
    state_sha256: str
    questions: dict[str, Any]
    answers: dict[str, Any]
    latency_ms: float
    calibrated: bool
    upstream_extra: dict[str, Any]
    state: str | dict[str, Any] | list[Any] | None = None


class DecisionLogger:
    def __init__(self, log_dir: Path | None = None, include_state: bool = False) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else omj_home() / "logs"
        self.include_state = include_state
        # Guards open/write/flush so concurrent writers (max_concurrency 16)
        # cannot interleave partial JSONL lines into the same daily file.
        self._lock = threading.Lock()

    def write(self, row: DecisionLogRow) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(UTC).strftime("%Y%m%d")
        path = self.log_dir / f"decisions-{date_str}.jsonl"

        payload: dict[str, Any] = {
            "ts": row.ts,
            "request_id": row.request_id,
            "backend": row.backend,
            "model": row.model,
            "state_sha256": row.state_sha256,
            "questions": redact_value(row.questions),
            "answers": redact_value(row.answers),
            "latency_ms": row.latency_ms,
            "calibrated": row.calibrated,
            "upstream_extra": redact_value(row.upstream_extra),
        }
        if self.include_state:
            payload["state"] = redact_value(row.state)

        # Second pass as a safety net for anything the field walk cannot reach.
        line = redact(json.dumps(payload, ensure_ascii=False))

        with self._lock:
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
                f.flush()

        return path


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
