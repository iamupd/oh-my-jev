"""The Jev reference shown next to every bench result (terminal, report.md, Reports page).

For each suite, in order:
  1. a finished local Jev run (backend typesafe) that measured the suite
  2. a fresh Jev run on the same items, when JEV_KEY or OPENROUTER_KEY is set (saved as a normal run,
     so later benches reuse it through rule 1)
  3. the values bundled with omj (references/jev-1.13.json, measured once by the omj authors)
Suites none of these cover get no reference.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from omj.errors import ErrorCode, OmjError

BUNDLED_FILE = Path(__file__).resolve().parent / "references" / "jev-1.13.json"
JEV_KEY_ENV = "JEV_KEY"
OPENROUTER_KEY_ENV = "OPENROUTER_KEY"


@lru_cache(maxsize=1)
def bundled() -> dict[str, Any]:
    return json.loads(BUNDLED_FILE.read_text(encoding="utf-8"))


def jev_provider() -> str | None:
    """The provider a fresh Jev measurement would use, or None without a key."""
    if os.environ.get(JEV_KEY_ENV):
        return "typesafe"
    if os.environ.get(OPENROUTER_KEY_ENV):
        return "openrouter"
    return None


def local_jev_runs(suites: list[str], exclude: Path | None = None) -> dict[str, dict[str, Any]]:
    """For each suite, the newest finished local Jev run that measured it."""
    from omj.bench.report_index import load_runs

    found: dict[str, dict[str, Any]] = {}
    skip = exclude.resolve() if exclude is not None else None
    for run in load_runs():  # newest first
        if run["backend"] != "typesafe" or (skip is not None and Path(run["path"]).resolve() == skip):
            continue
        for suite in suites:
            if suite not in found and suite in run["suites"] and run["suites"][suite].get("accuracy") is not None:
                found[suite] = {
                    "metrics": run["suites"][suite],
                    "source": "local run",
                    "run": run["name"],
                    "measured_at": (run["run_at"] or "")[:10],
                }
    return found


def from_report(path: Path, suites: list[str]) -> dict[str, dict[str, Any]]:
    """A reference taken from one report.json (bench --reference <file>)."""
    try:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OmjError(ErrorCode.E_CONFIG, f"--reference {path}: not a readable report.json ({exc})") from exc
    env = data.get("environment") or {}
    measured = ((data.get("run") or {}).get("finished_at") or env.get("run_at") or "")[:10]
    return {
        s: {"metrics": m, "source": "report", "run": Path(path).expanduser().parent.name, "measured_at": measured}
        for s, m in (data.get("suites") or {}).items()
        if s in suites
    }


def from_bundled(suites: list[str]) -> dict[str, dict[str, Any]]:
    ref = bundled()
    return {
        s: {"metrics": ref["suites"][s], "source": "bundled", "run": "", "measured_at": ref["measured_at"]}
        for s in suites
        if s in ref["suites"]
    }


def assemble(name: str, per_suite: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The `reference` block stored in report.json."""
    if not per_suite:
        return None
    return {"name": name, "note": bundled()["note"] if any(v["source"] == "bundled" for v in per_suite.values()) else "",
            "suites": per_suite}


def verdict(value: float | None, ref: float | None, ci: list[float] | None, direction: str) -> str:
    """"≈" when the reference lies inside the run's 95% CI (accuracy only), else "▲" better / "▼" worse."""
    if value is None or ref is None:
        return ""
    if ci is not None and ci[0] <= ref <= ci[1]:
        return "≈"
    if abs(value - ref) < 5e-4:
        return "="
    better = value > ref if direction == "up" else value < ref
    return "▲" if better else "▼"
