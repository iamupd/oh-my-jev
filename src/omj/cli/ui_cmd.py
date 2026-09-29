"""`omj ui`: serve the decision playground over one or more targets (specs/omj-ui/spec.md)."""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from pathlib import Path

import typer
import uvicorn

from omj.config import load_config
from omj.errors import ErrorCode, OmjError

DEFAULT_PORT = 8800
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
ACCESS_ENV_VAR = "OMJ_UI_TOKEN"


def resolve_access(host: str, allow_remote: bool, code: str | None) -> str | None:
    """Loopback needs no access code; any other bind needs --allow-remote and gets one (REQ-U11)."""
    if host in LOOPBACK_HOSTS:
        return code or None
    if not allow_remote:
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"refusing to bind omj ui to {host}: it has no login and its targets may spend your API keys. "
            "Pass --allow-remote to expose it with a required access token.",
        )
    return code or secrets.token_urlsafe(24)


def build_ui(
    targets: list[str],
    *,
    env_file: Path | None = None,
    builder: Callable | None = None,
    access_code: str | None = None,
):
    """Resolve targets, load their backends, and return the UI app (injectable builder for tests)."""
    from omj.ui.app import create_ui_app
    from omj.ui.targets import build_target_apps, default_targets, load_env_file, parse_target

    if env_file is not None:
        loaded = load_env_file(env_file)
        typer.echo(f"omj ui: loaded {len(loaded)} variable name(s) from {env_file}")  # names only, never values
    config = load_config()
    specs = [parse_target(t) for t in targets] if targets else default_targets(config)
    return create_ui_app(build_target_apps(config, specs, builder=builder), access_code=access_code)


def register(app: typer.Typer) -> None:
    @app.command()
    def ui(
        target: list[str] = typer.Option(
            [],
            "--target",
            help="name=backend[:adapter], repeatable. e.g. --target mine=semif:C:/adapters/best --target jev=typesafe. "
            "Default: config.toml's backend, plus 'jev' when JEV_KEY is set.",
        ),
        host: str = typer.Option("127.0.0.1", "--host", help="Bind address. Non-loopback addresses need --allow-remote."),
        port: int = typer.Option(DEFAULT_PORT, "--port", help="Port for the web UI."),
        allow_remote: bool = typer.Option(
            False, "--allow-remote", help="Allow a non-loopback --host; every request then needs an access token."
        ),
        access: str | None = typer.Option(
            None, "--token", help=f"Access token for remote use (default: ${ACCESS_ENV_VAR}, else generated)."
        ),
        env_file: Path | None = typer.Option(
            None, "--env-file", help="KEY=VALUE file to load API keys from (existing variables win)."
        ),
    ) -> None:
        """Open a local web playground to test and compare decision models (needs no extra)."""
        code = resolve_access(host, allow_remote, access or os.environ.get(ACCESS_ENV_VAR))
        ui_app = build_ui(target, env_file=env_file, access_code=code)
        names = ", ".join(t["name"] for t in ui_app.state.targets)
        url = f"http://{host}:{port}/" + (f"?token={code}" if code else "")
        typer.echo(f"omj ui: targets [{names}]")
        typer.echo(f"omj ui: open {url}")
        if code:
            typer.echo("omj ui: remote access is on; share the URL only with people who may spend your API quota")
        uvicorn.run(ui_app, host=host, port=port)
