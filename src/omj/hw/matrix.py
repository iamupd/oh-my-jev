"""Backend selection matrix based on hardware detection."""

from __future__ import annotations

from dataclasses import dataclass

from omj.errors import ErrorCode, OmjError
from omj.hw.detect import Hardware


@dataclass(frozen=True)
class Overrides:
    """Override values for backend selection matrix."""

    backend: str | None = None
    model: str | None = None
    quant: str | None = None
    revision: str | None = None


_BACKEND_DEFAULTS: dict[str, tuple[str, str]] = {
    "typesafe": ("jev-latest", ""),
    "kev": ("jaredpalmer/kev-4b", ""),
    "semif": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "mock": ("", ""),
}


def _backend_default(backend: str, hw: Hardware) -> tuple[str, str]:
    """Default model/revision for an overridden backend, sized to the GPU tier."""
    if backend == "kev" and hw.device == "cuda" and hw.vram_gb is not None and hw.vram_gb >= 24.0:
        return ("jaredpalmer/kev-9b", "")
    return _BACKEND_DEFAULTS.get(backend, ("", ""))


@dataclass(frozen=True)
class Selection:
    """Selected backend configuration with reasoning."""

    backend: str
    model: str
    revision: str
    quant: str
    reason: str


def select_backend(
    hw: Hardware,
    has_typesafe_key: bool,
    has_openrouter_key: bool,
    overrides: Overrides = Overrides(),
) -> Selection:
    """Select backend and model based on hardware.

    Implements design matrix from §4.4:
    - cuda, vram ≥ 24: kev backend with jaredpalmer/kev-9b
    - cuda, 12 ≤ vram < 24: semif backend with Qwen/Qwen3.5-4B
    - cuda, 6 ≤ vram < 12: semif backend with Qwen/Qwen3.5-2B
    - cuda, vram < 6 OR cpu: typesafe (if keys), else raise E_NO_GPU

    Overrides replace matrix results after evaluation (REQ-003).
    When overrides.backend is given, the GPU rule must not raise (REQ-003).
    """
    # Evaluate matrix first
    backend = ""
    model = ""
    revision = ""
    quant = "bf16"
    reason = ""

    if hw.device == "cuda" and hw.vram_gb is not None:
        if hw.vram_gb >= 24.0:
            backend = "kev"
            model = "jaredpalmer/kev-9b"
            revision = ""
            quant = "bf16"
            reason = "CUDA device with ≥24GB VRAM"
        elif hw.vram_gb >= 12.0:
            backend = "semif"
            model = "Qwen/Qwen3.5-4B"
            revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
            quant = "bf16"
            reason = "CUDA device with 12-24GB VRAM"
        elif hw.vram_gb >= 6.0:
            backend = "semif"
            model = "Qwen/Qwen3.5-2B"
            revision = "15852e8c16360a2fea060d615a32b45270f8a8fc"
            quant = "bf16"
            reason = "CUDA device with 6-12GB VRAM"
        else:
            # vram < 6: an explicit --backend or --model overrides the matrix, so
            # it must not raise here (REQ-003). A bare --model means the local
            # semif backend, which can still run a small model on this GPU.
            if overrides.backend is not None or overrides.model is not None:
                backend = ""
                model = ""
                revision = ""
                quant = "bf16"
                reason = ""
            elif has_typesafe_key or has_openrouter_key:
                backend = "typesafe"
                model = "jev-latest"
                revision = ""
                quant = "bf16"
                provider = "typesafe" if has_typesafe_key else "openrouter"
                reason = f"CUDA device with <6GB VRAM, using {provider} backend"
            else:
                raise OmjError(
                    ErrorCode.E_NO_GPU,
                    f"CUDA device detected but insufficient VRAM ({hw.vram_gb}GB < 6GB) and no API key configured.",
                )
    elif hw.device == "cpu":
        # CPU path: only an explicit --backend avoids the raise. A bare --model
        # still needs a GPU (semif is the only local model backend), so it keeps
        # the E_NO_GPU error rather than silently picking an unusable backend.
        if overrides.backend is not None:
            backend = ""
            model = ""
            revision = ""
            quant = "bf16"
            reason = ""
        elif has_typesafe_key or has_openrouter_key:
            backend = "typesafe"
            model = "jev-latest"
            revision = ""
            quant = "bf16"
            provider = "typesafe" if has_typesafe_key else "openrouter"
            reason = f"CPU device, using {provider} backend"
        else:
            raise OmjError(
                ErrorCode.E_NO_GPU,
                "CPU device detected and no API key configured.",
            )

    # Apply overrides after matrix evaluation. A model/revision pair belongs to
    # one backend, so switching backend or model must not keep the other half of
    # the pair the matrix picked (REQ-003).
    if not backend and overrides.backend is None and overrides.model is not None:
        backend = "semif"
        reason = "model override on a small GPU, using the local semif backend"
    if overrides.backend is not None and overrides.backend != backend:
        backend = overrides.backend
        model, revision = _backend_default(backend, hw)
    if overrides.model is not None and overrides.model != model:
        model = overrides.model
        revision = ""
    if overrides.revision is not None:
        revision = overrides.revision
    if overrides.quant is not None:
        quant = overrides.quant

    return Selection(
        backend=backend,
        model=model,
        revision=revision,
        quant=quant,
        reason=reason,
    )
