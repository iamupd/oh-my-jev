"""Target parsing and per-target gateway construction for `omj ui` (REQ-U01, REQ-U03).

A target is one decision model the UI calls. Each target gets its own, unchanged
gateway app (the same one `omj serve` runs), so the browser speaks the exact
System One wire protocol and sees the gateway's calibration and validation.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from omj.config import Config
from omj.errors import ErrorCode, OmjError

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
JEV_KEY_ENV = "JEV_KEY"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    backend: str
    adapter: str = ""


def parse_target(text: str) -> TargetSpec:
    """Parse ``name=backend[:adapter]``; the adapter may itself contain ``:`` (Windows paths)."""
    name, sep, rest = text.partition("=")
    name = name.strip()
    if not sep or not rest.strip():
        raise OmjError(ErrorCode.E_CONFIG, f"--target must look like name=backend[:adapter], got {text!r}")
    if not _NAME_RE.match(name):
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"target name {name!r} must be 1-32 chars of lowercase letters, digits, '-' or '_'",
        )
    backend, _, adapter = rest.strip().partition(":")
    return TargetSpec(name=name, backend=backend.strip(), adapter=adapter.strip())


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
    backend = config.backend.model_copy(update={"name": spec.backend, "adapter": spec.adapter})
    if spec.backend != config.backend.name:
        # model/revision belong to the configured backend; another backend uses its own default
        backend = backend.model_copy(update={"model": "", "revision": "", "calibration": ""})
        if spec.backend == "semif" and config.backend.name != "semif":
            raise OmjError(
                ErrorCode.E_CONFIG,
                "a semif target needs backend.model in config.toml; run 'omj init' or set backend.name=semif",
            )
    # The UI calls its targets from the same origin; the browser never holds a bearer key.
    serve = config.serve.model_copy(update={"api_key": ""})
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
            raise OmjError(exc.code, f"target {spec.name!r} ({spec.backend}): {exc.message}") from exc
    return apps
