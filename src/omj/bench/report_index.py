"""Index of finished bench runs for the Reports view (`omj ui` /reports and `omj bench --view`).

Runs live in `$OMJ_HOME/runs/<dir>/report.json`. A run written elsewhere with `--out` is
remembered in `$OMJ_HOME/runs/.extra-runs.json` so the Reports view still finds it.
Only `report.json` files are read; nothing is written except that small list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omj.config import omj_home

EXTRA_FILE = ".extra-runs.json"
MAX_EXTRA = 200
REPORT_FILE = "report.json"
SUMMARY_METRICS = (
    "n", "accuracy", "chance_corrected_accuracy", "hard_accuracy", "ece", "ece_jevbench", "brier", "nll",
    "coverage_at_risk", "valid_vector_rate", "order_shift_max", "overconfidence_rate", "ko_en_gap",
    "pair_agreement", "latency_p50_ms", "latency_p95_ms", "cost_per_1k",
)
DETAIL_FIELDS = ("accuracy_ci95", "n_scored", "wall_seconds", "by_kind", "by_tag", "reliability")


def runs_root() -> Path:
    return omj_home() / "runs"


def _extra_path() -> Path:
    return runs_root() / EXTRA_FILE


def register_run(out_dir: Path) -> None:
    """Remember a run directory outside `runs_root()` so the Reports view lists it."""
    out_dir = Path(out_dir).resolve()
    root = runs_root().resolve()
    if out_dir.parent == root:
        return
    try:
        extra = json.loads(_extra_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        extra = []
    extra = [p for p in extra if p != str(out_dir)] + [str(out_dir)]
    _extra_path().parent.mkdir(parents=True, exist_ok=True)
    _extra_path().write_text(json.dumps(extra[-MAX_EXTRA:], indent=1), encoding="utf-8")


def _run_dirs() -> list[Path]:
    root = runs_root()
    dirs = [d for d in root.iterdir() if (d / REPORT_FILE).is_file()] if root.is_dir() else []
    try:
        extra = json.loads(_extra_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        extra = []
    seen = {d.resolve() for d in dirs}
    for raw in extra:
        path = Path(raw)
        if (path / REPORT_FILE).is_file() and path.resolve() not in seen:
            dirs.append(path)
            seen.add(path.resolve())
    return dirs


def _compact(run_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((run_dir / REPORT_FILE).read_text(encoding="utf-8"))
        env = data.get("environment") or {}
        run = data.get("run") or {}
        suites = {
            name: {k: metrics.get(k) for k in SUMMARY_METRICS}
            for name, metrics in (data.get("suites") or {}).items()
            if name != "massive-pair"
        }
    except (OSError, ValueError, AttributeError):
        return None
    details = {
        name: {k: d.get(k) for k in DETAIL_FIELDS}
        for name, d in (data.get("details") or {}).items()
        if isinstance(d, dict)
    }
    options = run.get("options") or {}
    reference = data.get("reference") or {}
    ref_suites = {
        name: {"metrics": {k: (info.get("metrics") or {}).get(k) for k in SUMMARY_METRICS},
               "source": info.get("source", ""), "run": info.get("run", ""), "measured_at": info.get("measured_at", "")}
        for name, info in (reference.get("suites") or {}).items()
        if isinstance(info, dict)
    }
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "model": env.get("model") or "",
        "backend": env.get("backend") or "",
        "quant": env.get("quant") or "",
        "route": env.get("route") or "",
        "adapter": Path(options["adapter"]).name if options.get("adapter") else "",
        "run_at": run.get("finished_at") or env.get("run_at") or "",
        "wall_seconds": run.get("wall_seconds"),
        "command": run.get("command") or "",
        "suites": suites,
        "details": details,
        "reference": ref_suites,
    }


def load_runs(limit: int = 300) -> list[dict[str, Any]]:
    """Every readable run, newest first, each with a unique `id` (its folder name, suffixed on clashes)."""
    runs = [r for r in (_compact(d) for d in _run_dirs()) if r is not None and r["suites"]]
    runs.sort(key=lambda r: r["run_at"], reverse=True)
    taken: dict[str, int] = {}
    for run in runs[:limit]:
        base = run["name"]
        taken[base] = taken.get(base, 0) + 1
        run["id"] = base if taken[base] == 1 else f"{base}-{taken[base]}"
    return runs[:limit]


def run_id_for(out_dir: Path) -> str | None:
    """The Reports-view id of the run written to `out_dir`."""
    target = Path(out_dir).resolve()
    for run in load_runs():
        if Path(run["path"]).resolve() == target:
            return run["id"]
    return None
