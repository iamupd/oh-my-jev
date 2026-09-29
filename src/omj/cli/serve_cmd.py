"""`omj serve`: load config, start the FastAPI gateway under uvicorn."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import typer
import uvicorn
from fastapi import FastAPI

from omj.backends.adapter_info import follow_adapter_base
from omj.backends.registry import create_backend
from omj.config import Config, load_config
from omj.errors import ErrorCode, OmjError
from omj.gateway.app import create_app
from omj.gateway.calibration import load_calibration
from omj.gateway.decision_log import DecisionLogger


@dataclass(frozen=True)
class ServeOverrides:
    """CLI-supplied overrides for `omj serve`; a field left as None keeps the config value."""

    backend: str | None = None
    host: str | None = None
    port: int | None = None
    calibration: str | None = None
    log_dir: str | None = None
    log_state: bool | None = None
    max_concurrency: int | None = None
    api_key: str | None = None
    adapter: str | None = None


def _require_adapter_dir(path: Path) -> str:
    """Validate a `--adapter` directory at command time, before any backend loads (REQ-012)."""
    try:
        # imported here: semif pulls transformers, which `omj serve --backend mock` must not need.
        from omj.backends.semif import validate_adapter_dir
    except ImportError as exc:
        raise OmjError(
            ErrorCode.E_BACKEND, f"--adapter needs the 'semif' extra: {exc}"
        ) from exc
    return str(validate_adapter_dir(path))


def build_server_config(config: Config, overrides: ServeOverrides) -> Config:
    """Return a new Config with `overrides` applied on top of `config`; never mutates either."""
    data = config.model_dump(mode="json")

    if overrides.backend is not None:
        data["backend"]["name"] = overrides.backend
    if overrides.calibration is not None:
        data["backend"]["calibration"] = overrides.calibration
    if overrides.host is not None:
        data["serve"]["host"] = overrides.host
    if overrides.port is not None:
        data["serve"]["port"] = overrides.port
    if overrides.max_concurrency is not None:
        data["serve"]["max_concurrency"] = overrides.max_concurrency
    if overrides.api_key is not None:
        data["serve"]["api_key"] = overrides.api_key
    if overrides.log_dir is not None:
        data["log"]["dir"] = overrides.log_dir
    if overrides.log_state is not None:
        data["log"]["state"] = overrides.log_state
    if overrides.adapter:
        data["backend"]["adapter"] = overrides.adapter
        from omj.models.sizing import configured_vram_gb

        note = follow_adapter_base(data["backend"], overrides.adapter, configured_vram_gb(config))
        if note:
            print(note, file=sys.stderr)

    return Config.model_validate(data)


def build_app(config: Config) -> FastAPI:
    """Create the backend, load it, and wire calibration/decision-log per config."""
    backend = create_backend(config.backend.name)
    backend.load(config.backend)

    calibration = None
    if config.backend.calibration:
        calibration = load_calibration(config.backend.calibration)

    decision_logger = DecisionLogger(
        log_dir=Path(config.log.dir) if config.log.dir else None,
        include_state=config.log.state,
    )

    return create_app(backend, config, calibration=calibration, decision_logger=decision_logger)


def run_serve(config: Config, *, run: Callable[..., None] = uvicorn.run) -> None:
    """Build the gateway app and hand it to `run` (uvicorn.run by default; injectable for tests)."""
    app = build_app(config)
    backend = app.state.backend
    typer.echo(
        f"omj serve: {backend.name}/{backend.model_id} on "
        f"http://{config.serve.host}:{config.serve.port}/v1/systemone"
    )
    run(app, host=config.serve.host, port=config.serve.port)


def register(app: typer.Typer) -> None:
    @app.command()
    def serve(
        backend: str | None = typer.Option(
            None, "--backend", help="Override backend.name from config.toml."
        ),
        port: int | None = typer.Option(
            None, "--port", help="Override serve.port from config.toml (default 8799)."
        ),
        host: str | None = typer.Option(
            None, "--host", help="Override serve.host from config.toml (default 127.0.0.1)."
        ),
        calibration: Path | None = typer.Option(
            None, "--calibration", help="Path to a calibration JSON file."
        ),
        log_dir: Path | None = typer.Option(
            None, "--log-dir", help="Directory for decision log JSONL files."
        ),
        log_state: bool = typer.Option(
            False,
            "--log-state",
            help="Include the raw request state in decision log rows.",
        ),
        max_concurrency: int | None = typer.Option(
            None, "--max-concurrency", help="Override serve.max_concurrency from config.toml."
        ),
        api_key: str | None = typer.Option(
            None, "--api-key", help="Override serve.api_key for this process only."
        ),
        adapter: Path | None = typer.Option(
            None,
            "--adapter",
            help="Path to a PEFT adapter directory to load on top of backend.model (semif only).",
        ),
    ) -> None:
        """Load config.toml, apply CLI overrides, and start the gateway with uvicorn."""
        if api_key is not None:
            typer.echo(
                "warning: an API key passed on the command line is visible in this "
                "machine's process list; prefer setting serve.api_key in config.toml instead."
            )

        adapter_value = _require_adapter_dir(adapter) if adapter is not None else None

        overrides = ServeOverrides(
            backend=backend if backend is not None or adapter_value is None else "semif",
            host=host,
            port=port,
            calibration=str(calibration) if calibration is not None else None,
            log_dir=str(log_dir) if log_dir is not None else None,
            log_state=True if log_state else None,
            max_concurrency=max_concurrency,
            api_key=api_key,
            adapter=adapter_value,
        )

        config = build_server_config(load_config(), overrides)
        run_serve(config, run=uvicorn.run)
