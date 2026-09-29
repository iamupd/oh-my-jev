"""Compare bench report.json files side by side with deltas vs. the first (design.md 2/3; REQ-013)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from omj.errors import ErrorCode, OmjError

METRICS: tuple[str, ...] = (
    "accuracy",
    "hard_accuracy",
    "ece",
    "brier",
    "coverage_at_risk",
    "latency_p50_ms",
    "cost_per_1k",
)


def _load_report(path: Path) -> dict[str, Any]:
    """Read one report.json, mapping a missing/unreadable/invalid file to E_CONFIG (REQ-013)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OmjError(ErrorCode.E_CONFIG, f"report not readable: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise OmjError(
            ErrorCode.E_CONFIG, f"report not readable: {path}: top level is not an object"
        )
    return data


def _model_tail(model: str | None) -> str:
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def _default_labels(paths: list[Path], reports_data: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    for data in reports_data:
        env = data.get("environment", {}) or {}
        backend = env.get("backend") or "unknown"
        tail = _model_tail(env.get("model"))
        candidates.append(f"{backend}+{tail}" if tail else backend)

    counts = Counter(candidates)
    resolved = [
        Path(path).parent.name if counts[candidate] > 1 else candidate
        for path, candidate in zip(paths, candidates)
    ]

    # Parent directory names can still collide (e.g. reports named report.json
    # inside identically named run folders); a numeric suffix guarantees the
    # labels stay usable as unique table/column headers.
    resolved_counts = Counter(resolved)
    seen: dict[str, int] = {}
    final: list[str] = []
    for label in resolved:
        if resolved_counts[label] > 1:
            seen[label] = seen.get(label, 0) + 1
            final.append(f"{label}-{seen[label]}")
        else:
            final.append(label)
    return final


def _report_info(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    env = data.get("environment", {}) or {}
    return {
        "path": str(path),
        "backend": env.get("backend"),
        "model": env.get("model"),
        "route": env.get("route"),
        "run_at": env.get("run_at"),
    }


def _common_suites(reports_data: list[dict[str, Any]]) -> list[str]:
    suite_sets = [set((data.get("suites") or {}).keys()) for data in reports_data]
    common = set.intersection(*suite_sets) if suite_sets else set()
    if not common:
        raise OmjError(ErrorCode.E_CONFIG, "no suites are common to every report")
    # Preserve the first report's suite order for a stable table order.
    first_order = list((reports_data[0].get("suites") or {}).keys())
    return [name for name in first_order if name in common]


def compare_reports(paths: list[Path], labels: list[str] | None = None) -> dict[str, Any]:
    """Load `paths`' report.json files and compute per-suite metric deltas vs. the first (REQ-013)."""
    if len(paths) < 2:
        raise OmjError(ErrorCode.E_CONFIG, "compare requires at least 2 report.json paths")
    if labels is not None and len(labels) != len(paths):
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"--labels count ({len(labels)}) does not match report count ({len(paths)})",
        )

    reports_data = [_load_report(path) for path in paths]
    resolved_labels = list(labels) if labels is not None else _default_labels(paths, reports_data)
    report_infos = [_report_info(path, data) for path, data in zip(paths, reports_data)]
    suite_names = _common_suites(reports_data)

    suites_out: dict[str, Any] = {}
    for suite_name in suite_names:
        suite_out: dict[str, Any] = {}
        delta_out: dict[str, list[float | None]] = {}
        for metric in METRICS:
            values: list[float | None] = [
                (data.get("suites", {}).get(suite_name, {}) or {}).get(metric) for data in reports_data
            ]
            baseline = values[0]
            deltas: list[float | None] = [None]
            for value in values[1:]:
                if value is None or baseline is None:
                    deltas.append(None)
                else:
                    deltas.append(value - baseline)
            suite_out[metric] = values
            delta_out[metric] = deltas
        suite_out["delta"] = delta_out
        suites_out[suite_name] = suite_out

    return {
        "labels": resolved_labels,
        "reports": report_infos,
        "suites": suites_out,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _format_delta(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:+.4f}"


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _render_markdown(result: dict[str, Any]) -> str:
    labels: list[str] = result["labels"]
    reports: list[dict[str, Any]] = result["reports"]
    suites: dict[str, Any] = result["suites"]

    lines: list[str] = []
    lines.append("# Benchmark comparison")
    lines.append("")
    lines.append(
        "Metrics of the suites shared by the given report.json files; "
        "Δ is the difference from the first report."
    )
    lines.append("")

    lines.append("## Reports")
    lines.append("")
    header_rows = [
        [label, _cell(info.get("backend")), _cell(info.get("model")), _cell(info.get("route")), _cell(info.get("run_at"))]
        for label, info in zip(labels, reports)
    ]
    lines.extend(_markdown_table(["Label", "Backend", "Model", "Route", "Run at"], header_rows))
    lines.append("")

    for suite_name, suite_data in suites.items():
        lines.append(f"## Suite: {suite_name}")
        lines.append("")
        deltas: dict[str, list[float | None]] = suite_data.get("delta", {})
        metric_rows: list[list[str]] = []
        for metric in METRICS:
            values: list[Any] = suite_data.get(metric, [])
            if all(value is None for value in values):
                continue  # not measured by any report: no row, just noise
            metric_deltas = deltas.get(metric, [])
            cells = [metric]
            for i, value in enumerate(values):
                if value is None:
                    cells.append("N/A")
                    continue
                delta = metric_deltas[i] if i < len(metric_deltas) else None
                delta_text = _format_delta(delta)
                if i == 0 or delta_text is None:
                    cells.append(_format_value(value))
                else:
                    cells.append(f"{_format_value(value)} (Δ{delta_text})")
            metric_rows.append(cells)
        lines.extend(_markdown_table(["Metric", *labels], metric_rows))
        lines.append("")
        lines.append("N/A: not measured by that report. Metrics no report measured are omitted.")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_compare(out_dir: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    """Write compare.json and compare.md into `out_dir`, returning both paths (REQ-013)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "compare.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    md_path = out_dir / "compare.md"
    md_path.write_text(_render_markdown(result), encoding="utf-8", newline="\n")

    return json_path, md_path
