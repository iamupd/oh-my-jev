"""omj CLI root.

Registers each subcommand's implementation module lazily so that touching one
subcommand (e.g. rewriting `init_cmd.py` in a later task) never requires
touching this file. Also centralizes OmjError -> exit code / stderr handling
so every subcommand can simply `raise OmjError(...)`.
"""

from __future__ import annotations

import sys

import click
import typer
from typer.core import TyperGroup

from omj import __version__
from omj.errors import OmjError, format_error

# Windows console default codepage (cp949) mangles UTF-8 output; force UTF-8.
def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


_force_utf8_streams()


class OmjGroup(TyperGroup):
    """Click group that turns any OmjError raised by a command into a formatted exit."""

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except OmjError as err:
            typer.echo(format_error(err), err=True)
            raise SystemExit(err.exit_code) from None


app = typer.Typer(cls=OmjGroup, add_completion=False, no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"omj {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the omj version and exit.",
    ),
) -> None:
    """omj: run open System One decision models behind a TypeSafe-Jev-compatible gateway."""


def _register_commands() -> None:
    from omj.cli import bench_cmd, compare_cmd, init_cmd, serve_cmd, train_cmd, ui_cmd

    init_cmd.register(app)
    serve_cmd.register(app)
    bench_cmd.register(app)
    train_cmd.register(app)
    compare_cmd.register(app)
    ui_cmd.register(app)


_register_commands()
