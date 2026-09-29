"""Training package: recipes, data building, loss, loop, calibration (omj train)."""

from __future__ import annotations

import importlib

from omj.errors import ErrorCode, OmjError


def require_peft():
    """Import and return the ``peft`` module, or raise E_BACKEND with an install hint."""
    try:
        return importlib.import_module("peft")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise OmjError(
            ErrorCode.E_BACKEND,
            "omj train needs the 'peft' package; run 'uv sync --extra semif'",
        ) from exc
