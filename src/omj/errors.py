"""Error codes and exception type shared by every omj command and route."""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    E_NO_GPU = "E_NO_GPU"
    E_SCHEMA = "E_SCHEMA"
    E_DOWNLOAD = "E_DOWNLOAD"
    E_CONFIG = "E_CONFIG"
    E_BACKEND = "E_BACKEND"
    E_AUTH = "E_AUTH"
    E_NET = "E_NET"


HINTS: dict[ErrorCode, str] = {
    ErrorCode.E_NO_GPU: "Run with --backend mock or --backend kev, or use a machine with a supported GPU.",
    ErrorCode.E_SCHEMA: "Fix the request payload so it matches the SystemOne request schema.",
    ErrorCode.E_DOWNLOAD: "Check your network connection and retry, or pass --no-download to skip fetching model files.",
    ErrorCode.E_CONFIG: "Fix the reported field in config.toml, or delete it to regenerate defaults with 'omj init'.",
    ErrorCode.E_BACKEND: "Install the required extra (e.g. 'uv sync --extra semif') or choose a different --backend.",
    ErrorCode.E_AUTH: "Set a valid API key via --api-key or the configured environment variable.",
    ErrorCode.E_NET: "Check connectivity to the upstream service and retry.",
}


class OmjError(Exception):
    """Raised for any user-facing failure; carries a stable code, message, and hint."""

    def __init__(self, code: ErrorCode, message: str, hint: str | None = None, exit_code: int = 1) -> None:
        self.code = code
        self.message = message
        self.hint = hint if hint is not None else HINTS[code]
        self.exit_code = exit_code
        super().__init__(message)


def format_error(err: OmjError) -> str:
    return f"{err.code.value}: {err.message}. {err.hint}"
