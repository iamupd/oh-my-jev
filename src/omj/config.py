"""Config model and TOML persistence for $OMJ_HOME/config.toml."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, ValidationError

from omj.errors import ErrorCode, OmjError


class HardwareSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["cuda", "cpu"] | None = None
    vram_gb: float | None = None
    ram_gb: float | None = None
    platform: str | None = None
    python: str | None = None
    gpu_name: str | None = None


class BackendSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "mock"
    model: str = ""
    revision: str = ""
    # Directory holding a trained LoRA adapter; merged into the base weights at load.
    adapter: str = ""
    quant: Literal["bf16", "4bit-prequant", "nf4"] = "bf16"
    calibration: str = ""
    kev_base_url: str = "http://127.0.0.1:8009"
    provider: Literal["typesafe", "openrouter"] = "typesafe"
    api_key_env: str = ""
    # Set by `omj init --no-download`: the backend must then load from cache
    # only, instead of silently fetching weights on its own.
    offline: bool = False


class ServeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8799
    max_concurrency: int = 16
    api_key: str = ""


class LogSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str = ""
    state: bool = False


class UsageSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_tokens_mode: Literal["zero", "typesafe"] = "zero"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hardware: HardwareSection = HardwareSection()
    backend: BackendSection = BackendSection()
    serve: ServeSection = ServeSection()
    log: LogSection = LogSection()
    usage: UsageSection = UsageSection()


def omj_home() -> Path:
    value = os.environ.get("OMJ_HOME")
    if value:
        return Path(value)
    return Path.home() / ".omj"


def config_path() -> Path:
    """$OMJ_CONFIG when set (e.g. a second model profile), else $OMJ_HOME/config.toml."""
    override = os.environ.get("OMJ_CONFIG")
    if override:
        return Path(override)
    return omj_home() / "config.toml"


def _dotted_path(loc: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in loc)


def load_config(path: Path | None = None) -> Config:
    target = path if path is not None else config_path()
    if not target.exists():
        return Config()

    try:
        with target.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise OmjError(ErrorCode.E_CONFIG, f"{target}: {exc}") from exc

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        field_path = _dotted_path(first["loc"])
        raise OmjError(ErrorCode.E_CONFIG, f"{field_path}: {first['msg']}") from exc


def save_config(cfg: Config, path: Path | None = None) -> None:
    target = path if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        # exclude_none: TOML has no null; unset hardware fields simply become
        # absent keys, which pydantic re-fills with their defaults on load.
        tomli_w.dump(cfg.model_dump(mode="json", exclude_none=True), f)
