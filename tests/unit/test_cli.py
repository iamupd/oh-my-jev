"""CLI root behavior: --version, --help, and uniform OmjError exit handling. # REQ-047, REQ-008"""

from __future__ import annotations

from typer.testing import CliRunner

from omj import __version__
from omj.cli.main import app

runner = CliRunner()


def _combined_output(result) -> str:
    output = result.stdout
    try:
        output += result.stderr
    except ValueError:
        pass  # stdout/stderr were mixed into one stream
    return output


def test_version_reports_package_version() -> None:
    # REQ-047
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_all_subcommands() -> None:
    # REQ-047
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("init", "serve", "bench"):
        assert name in result.stdout


def test_init_unknown_backend_exits_one_with_e_backend() -> None:
    # REQ-008
    # An explicit unknown --backend fails in create_backend() regardless of the
    # host's detected hardware, so this is deterministic on any machine/CI.
    result = runner.invoke(app, ["init", "--backend", "nonexistent", "--yes"])
    assert result.exit_code == 1
    assert "E_BACKEND" in _combined_output(result)


def test_serve_unknown_backend_exits_one_with_e_backend() -> None:
    # REQ-008
    # T14 replaced the `serve` stub with a real command that starts uvicorn;
    # an unknown --backend still fails before uvicorn.run() is reached, so this
    # exercises the shared OmjError -> exit 1 path without binding a real socket.
    # See tests/integration/test_serve_cmd.py for full serve behavior coverage.
    result = runner.invoke(app, ["serve", "--backend", "no-such-backend"])
    assert result.exit_code == 1
    assert "E_BACKEND" in _combined_output(result)


def test_bench_no_target_exits_one_with_e_config() -> None:
    # REQ-008
    # With neither --backend nor --endpoint, `bench` uses the backend in config.toml;
    # the isolated test home has none, so it fails before touching any backend.
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 1
    assert "E_CONFIG" in _combined_output(result)
