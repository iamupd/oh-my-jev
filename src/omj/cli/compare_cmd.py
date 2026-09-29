"""`omj compare`: diff two or more bench report.json files with deltas vs. the first (REQ-013)."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from omj.bench.compare import compare_reports, write_compare
from omj.bench.terminal import print_compare


def _parse_labels(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",")]


def register(app: typer.Typer) -> None:
    @app.command()
    def compare(
        reports: list[Path] = typer.Argument(
            ..., help="Two or more report.json paths to compare (delta is vs. the first)."
        ),
        out: Path = typer.Option(..., "--out", help="Output directory for compare.json/compare.md."),
        labels: str | None = typer.Option(
            None, "--labels", help="Comma-separated labels, one per report (default: backend+model)."
        ),
        json_output: bool = typer.Option(
            False, "--json", help="Print compare.json to stdout instead of the comparison tables."
        ),
    ) -> None:
        """Compare bench report.json files and write compare.json/compare.md."""
        result = compare_reports(reports, labels=_parse_labels(labels))
        json_path, md_path = write_compare(out, result)

        if json_output:
            typer.echo(json_path.read_text(encoding="utf-8"), nl=False)
        else:
            print_compare(sys.stdout, result, [md_path, json_path])
