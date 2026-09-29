"""Lazy backend lookup by name, so optional extras are only imported when used."""

from __future__ import annotations

import importlib

from omj.backends.base import Backend
from omj.errors import ErrorCode, OmjError

BACKENDS: dict[str, str] = {
    "mock": "omj.backends.mock:MockBackend",
    "semif": "omj.backends.semif:SemifBackend",
    "kev": "omj.backends.kev:KevBackend",
    "typesafe": "omj.backends.typesafe:TypeSafeBackend",
}

_EXTRA_HINTS: dict[str, str] = {
    "semif": "uv sync --extra semif",
}


def create_backend(name: str) -> Backend:
    target = BACKENDS.get(name)
    if target is None:
        raise OmjError(ErrorCode.E_BACKEND, f"unknown backend {name!r}")

    module_name, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
        backend_cls = getattr(module, class_name)
    except ImportError as exc:
        hint = _EXTRA_HINTS.get(name)
        if hint:
            raise OmjError(
                ErrorCode.E_BACKEND,
                f"backend {name!r} requires the '{name}' extra; run '{hint}'",
            ) from exc
        raise OmjError(ErrorCode.E_BACKEND, f"backend {name!r} is unavailable: {exc}") from exc

    return backend_cls()
