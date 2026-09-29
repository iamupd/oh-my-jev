"""Bench report generation: report.json/report.md and environment metadata (design.md 3; REQ-040)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from omj import __version__
from omj.bench.report_md import render_markdown
from omj.bench.types import Metrics

Route = Literal["local", "direct", "openrouter"]


@dataclass
class Environment:
    hardware: dict[str, Any]
    backend: str
    model: str
    revision: str
    quant: str
    route: Route
    git_commit: str
    omj_version: str
    run_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hardware_to_dict(hardware: Any) -> dict[str, Any]:
    if hardware is None:
        return {}
    if isinstance(hardware, dict):
        return dict(hardware)
    if hasattr(hardware, "model_dump"):
        return hardware.model_dump()
    if is_dataclass(hardware) and not isinstance(hardware, type):
        return asdict(hardware)
    return dict(vars(hardware))


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit if commit else "unknown"


def collect_environment(
    config_or_none: Any,
    backend_name: str,
    model: str,
    route: Route,
) -> Environment:
    """Assemble the Environment record for a bench report (REQ-040).

    `config_or_none` is an `omj.config.Config`-like object exposing `.hardware`
    and `.backend` sections, or None. Hardware/revision/quant fall back to
    empty values when no config is supplied.
    """
    hardware: dict[str, Any] = {}
    revision = ""
    quant = ""
    if config_or_none is not None:
        hardware = _hardware_to_dict(getattr(config_or_none, "hardware", None))
        backend_cfg = getattr(config_or_none, "backend", None)
        if backend_cfg is not None:
            revision = getattr(backend_cfg, "revision", "") or ""
            quant = getattr(backend_cfg, "quant", "") or ""

    return Environment(
        hardware=hardware,
        backend=backend_name,
        model=model,
        revision=revision,
        quant=quant,
        route=route,
        git_commit=_git_commit(),
        omj_version=__version__,
        run_at=datetime.now(timezone.utc).isoformat(),
    )


def write_report(
    out_dir: Path,
    *,
    suite_metrics: dict[str, Metrics],
    env: Environment,
    calibration: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    details: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Write report.json and report.md into `out_dir`, returning both paths (REQ-040).

    `run` (times, command, options) and `details` (per-suite breakdowns from
    omj.bench.details) are optional so older callers keep working.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    report_data: dict[str, Any] = {
        "omj": {"version": __version__},
        "environment": env.to_dict(),
        "suites": {name: metrics.to_dict() for name, metrics in suite_metrics.items()},
        "calibration": calibration,
    }
    if run is not None:
        report_data["run"] = run
    if details is not None:
        report_data["details"] = details

    json_path = out_dir / "report.json"
    json_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    md_path = out_dir / "report.md"
    md_path.write_text(render_markdown(report_data), encoding="utf-8", newline="\n")

    return json_path, md_path
