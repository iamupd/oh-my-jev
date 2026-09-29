"""Target parsing and per-target gateway construction for `omj ui` (REQ-U01, REQ-U03).

A target is one decision model the UI calls. Each target gets its own, unchanged
gateway app (the same one `omj serve` runs), so the browser speaks the exact
System One wire protocol and sees the gateway's calibration and validation.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from fastapi import FastAPI

from omj.backends.adapter_info import follow_adapter_base
from omj.backends.registry import BACKENDS
from omj.config import BackendSection, Config, load_config
from omj.models.sizing import configured_vram_gb, is_hf_id, resolve_named_model
from omj.errors import ErrorCode, OmjError

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
JEV_KEY_ENV = "JEV_KEY"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    backend: str
    adapter: str = ""
    # A config.toml whose [backend] this target uses (``name=@path``): lets each target load its own model.
    profile: str = ""
    # A Hugging Face model id run on semif (``name=org/model``), no config file needed.
    model: str = ""


def _label(text: str) -> str:
    """A target name made from free text: lowercase, allowed characters only, at most 64 chars."""
    label = re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-._")[:64]
    return label or "target"


def _derived_name(spec_text: str) -> str:
    if is_hf_id(spec_text):
        return _label(spec_text.split("/", 1)[1])            # Qwen/Qwen3.5-0.8B -> qwen3.5-0.8b
    if spec_text.startswith("@"):
        return _label(Path(spec_text[1:]).stem)               # @~/.omj/qwen-2b.toml -> qwen-2b
    backend, _, adapter = spec_text.partition(":")
    if not adapter.strip():
        return _label(backend)                                # semif -> semif
    path = Path(adapter.strip())
    # .../adapters/<recipe>/<run_id>/best -> <recipe>; otherwise the folder name
    return _label(path.parents[1].name if path.name in ("best", "last") and len(path.parents) > 1 else path.name)


def parse_target(text: str) -> TargetSpec:
    """Parse one ``--target``: ``[name=]org/model``, ``[name=]backend[:adapter]`` or ``[name=]@config.toml``.

    ``org/model`` is a Hugging Face id run locally with semif. Without ``name=`` the on-screen
    name is derived from the rest (``Qwen/Qwen3.5-0.8B`` -> ``qwen3.5-0.8b``). The adapter may
    contain ``:`` (Windows paths). ``@path`` takes the whole [backend] section from another config file.
    """
    text = text.strip()
    head, sep, tail = text.partition("=")
    if sep and _NAME_RE.match(head.strip()):
        name, rest = head.strip(), tail.strip()
    elif sep and not head.strip():
        raise OmjError(ErrorCode.E_CONFIG, f"--target has an empty name: {text!r}")
    else:
        name, rest = "", text
    if not rest:
        raise OmjError(ErrorCode.E_CONFIG, f"--target {text!r} names nothing to run")
    backend_part = rest.partition(":")[0].strip()
    if not (is_hf_id(rest) or rest.startswith("@") or backend_part in BACKENDS):
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"--target {text!r}: expected a Hugging Face id (org/model), a backend "
            f"({', '.join(BACKENDS)})[:adapter] or @config.toml, optionally prefixed with name=",
        )
    name = name or _derived_name(rest)
    if is_hf_id(rest):
        return TargetSpec(name=name, backend="semif", model=rest)
    if rest.startswith("@"):
        profile = Path(rest[1:]).expanduser()
        if not profile.is_file():
            raise OmjError(ErrorCode.E_CONFIG, f"target {name!r}: config file not found: {profile}")
        return TargetSpec(name=name, backend=load_config(profile).backend.name, profile=str(profile))
    backend, _, adapter = rest.partition(":")
    adapter = str(Path(adapter.strip()).expanduser()) if adapter.strip() else ""
    return TargetSpec(name=name, backend=backend.strip(), adapter=adapter)


def parse_targets(texts: list[str]) -> list[TargetSpec]:
    """Parse every ``--target``; derived names that collide get ``-2``, ``-3`` ... (explicit ones must be unique)."""
    specs: list[TargetSpec] = []
    for text in texts:
        spec = parse_target(text)
        explicit = "=" in text and _NAME_RE.match(text.partition("=")[0].strip()) is not None
        if not explicit:
            taken = {s.name for s in specs}
            base, n = spec.name, 2
            while spec.name in taken:
                spec = replace(spec, name=f"{base[:60]}-{n}")
                n += 1
        specs.append(spec)
    check_unique(specs)
    return specs


def default_targets(config: Config, env: Mapping[str, str] | None = None) -> list[TargetSpec]:
    """config.toml's backend, plus a ``jev`` target when JEV_KEY is set (REQ-U03)."""
    env = os.environ if env is None else env
    first = config.backend.name
    targets = [TargetSpec("jev" if first == "typesafe" else "local", first, config.backend.adapter)]
    if first != "typesafe" and env.get(JEV_KEY_ENV):
        targets.append(TargetSpec("jev", "typesafe"))
    return targets


def check_unique(targets: list[TargetSpec]) -> None:
    names = [t.name for t in targets]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise OmjError(ErrorCode.E_CONFIG, f"duplicate target name(s): {', '.join(duplicates)}")


def target_config(config: Config, spec: TargetSpec) -> Config:
    """Derive one target's Config: its backend and adapter, no gateway bearer key."""
    serve = config.serve.model_copy(update={"api_key": ""})
    if spec.profile:
        return config.model_copy(update={"backend": load_config(Path(spec.profile)).backend, "serve": serve})
    if spec.model:
        data, _ = resolve_named_model(config.backend.model_dump(mode="json"), spec.model, configured_vram_gb(config))
        return config.model_copy(update={"backend": BackendSection.model_validate(data), "serve": serve})
    backend = config.backend.model_copy(update={"name": spec.backend, "adapter": spec.adapter})
    if spec.adapter:
        # Same rule as bench/serve: an adapter loads onto the base model it was trained on.
        data = backend.model_dump(mode="json")
        if follow_adapter_base(data, spec.adapter, configured_vram_gb(config)):
            backend = BackendSection.model_validate(data)
            return config.model_copy(update={"backend": backend, "serve": serve})
    if spec.backend != config.backend.name:
        # model/revision belong to the configured backend; another backend uses its own default
        backend = backend.model_copy(update={"model": "", "revision": "", "calibration": ""})
        if spec.backend == "semif" and config.backend.name != "semif":
            raise OmjError(
                ErrorCode.E_CONFIG,
                "a semif target needs backend.model in config.toml; run 'omj init' or set backend.name=semif",
            )
    # The UI calls its targets from the same origin; the browser never holds a bearer key.
    return config.model_copy(update={"backend": backend, "serve": serve})


def load_env_file(path: Path, env: dict[str, str] | None = None) -> list[str]:
    """Read KEY=VALUE lines into ``env`` (default os.environ) without overriding set keys.

    Returns the key names loaded; values are never returned or logged (REQ-U04).
    """
    target = os.environ if env is None else env
    if not path.is_file():
        raise OmjError(ErrorCode.E_CONFIG, f"env file not found: {path}")
    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in target:
            target[key] = value
            loaded.append(key)
    return loaded


def build_target_apps(
    config: Config,
    specs: list[TargetSpec],
    *,
    builder: Callable[[Config], FastAPI] | None = None,
) -> list[tuple[TargetSpec, FastAPI]]:
    """Build (and load) one gateway app per target, failing fast on the first bad target."""
    check_unique(specs)
    if builder is None:
        from omj.cli.serve_cmd import build_app as builder
    apps = []
    for spec in specs:
        try:
            apps.append((spec, builder(target_config(config, spec))))
        except OmjError as exc:
            raise OmjError(exc.code, f"target {spec.name!r} ({spec.backend}): {exc.message}", hint=exc.hint) from exc
    return apps
