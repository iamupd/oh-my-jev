"""Read which base model a LoRA adapter was trained on (no torch import, so CLI code can use it)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ADAPTER_CONFIG_FILE = "adapter_config.json"
_HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
# omj model cache path: .../models/<org>__<name>/<revision>
_OMJ_CACHE_DIR = re.compile(r"([\w.-]+)__([\w.-]+)(?:[\\/]([0-9a-f]{40}|main))?[\\/]?$")


def adapter_base(adapter_dir: str | Path) -> tuple[str, str] | None:
    """(Hugging Face id, revision) of the adapter's base model, or None when it cannot be told.

    Accepts a plain id in adapter_config.json ("Qwen/Qwen3.5-4B", revision from its
    "revision" field) or an omj model cache path (".../models/Qwen__Qwen3.5-4B/<revision>").
    Any other local path returns None. The revision is "" when unknown.
    """
    try:
        raw = json.loads((Path(adapter_dir).expanduser() / ADAPTER_CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    base = str(raw.get("base_model_name_or_path") or "").strip()
    if _HF_ID.match(base):
        return base, str(raw.get("revision") or "")
    match = _OMJ_CACHE_DIR.search(base)
    if not match:
        return None
    revision = match.group(3) or ""
    return f"{match.group(1)}/{match.group(2)}", "" if revision == "main" else revision


def adapter_base_model(adapter_dir: str | Path) -> str | None:
    base = adapter_base(adapter_dir)
    return base[0] if base else None


def follow_adapter_base(backend: dict[str, Any], adapter: str | Path, vram_gb: float | None = None) -> str | None:
    """Point a semif backend section at the base model an explicitly given adapter was trained on.

    Mutates `backend` (a BackendSection dump) and returns a note for the user, or None when the
    configured model already matches or the adapter's base cannot be read.
    """
    if not isinstance(adapter, (str, Path)):
        return None  # left for BackendSection validation to reject
    base = adapter_base(adapter)
    if base is None or backend.get("name") != "semif":
        return None
    model, revision = base
    if model.lower() == str(backend.get("model") or "").lower():
        return None
    previous = backend.get("model") or "(none)"
    backend["model"], backend["revision"] = model, revision
    if vram_gb is not None:
        from omj.models.sizing import auto_quant

        backend["quant"] = auto_quant(model, vram_gb)
    return f"Using {model} (the adapter's base model, {backend.get('quant', 'bf16')}) instead of the configured {previous}."
