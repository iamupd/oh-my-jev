"""Render report.json as a human-readable report.md.

Layout follows common evaluation reports (run header with time, command and
configuration; a cross-suite summary with confidence intervals; per-suite
detail with breakdowns and a reliability table): the reader should be able to
tell what was run, on what, and how far to trust each number.
Metrics a suite cannot produce are listed once as "not measured", with the
option that would measure them, instead of filling tables with N/A.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# name -> (label, direction, meaning). Direction: "up" higher is better, "down" lower is better,
# "zero" closer to 0 is better.
METRIC_INFO: dict[str, tuple[str, str, str]] = {
    "accuracy": ("Accuracy", "up", "Top-1 answer equals the expected key"),
    "chance_corrected_accuracy": ("Chance-corrected accuracy", "up", "Accuracy rescaled so random guessing scores 0"),
    "hard_accuracy": ("Hard accuracy", "up", "Accuracy on items tagged `hard`"),
    "ece": ("ECE (15 bins)", "down", "Expected calibration error of the top-1 probability"),
    "ece_jevbench": ("ECE (10 bins)", "down", "ECE with the 10-bin JevBench convention"),
    "brier": ("Brier", "down", "Mean squared error of the full probability vector"),
    "nll": ("NLL", "down", "Negative log-likelihood of the expected key"),
    "coverage_at_risk": ("Coverage at 5% risk", "up", "Largest share of items answerable with at most 5% error, most confident first"),
    "valid_vector_rate": ("Valid vector rate", "up", "Answers whose probabilities sum to 1 (tolerance 1e-3)"),
    "valid_vector_rate_lenient": ("Valid vector rate (lenient)", "up", "Same with tolerance 2e-2"),
    "order_shift_max": ("Order shift (max)", "down", "Largest probability change when options are reordered"),
    "overconfidence_rate": ("Overconfidence rate", "down", "Underdetermined items answered with top probability >= 0.9"),
    "mean_max_prob": ("Mean top probability", "down", "Mean top probability on underdetermined items"),
    "uniform_deviation": ("Uniform deviation", "down", "Distance from uniform where no option is favoured"),
    "ko_en_gap": ("KO-EN gap", "zero", "English minus Korean accuracy on paired MASSIVE items"),
    "pair_agreement": ("KO-EN agreement", "up", "Same prediction for the Korean and English utterance"),
    "latency_p50_ms": ("Latency p50 (ms)", "down", "Median time per item"),
    "latency_p95_ms": ("Latency p95 (ms)", "down", "95th percentile time per item"),
    "cost_per_1k": ("Cost per 1k items (USD)", "down", "API price or --gpu-cost-per-hour times wall time"),
}
ARROW = {"up": "↑", "down": "↓", "zero": "→0"}

# How to obtain a metric a suite did not produce.
HOW_TO_MEASURE: dict[str, str] = {
    "hard_accuracy": "a suite with `hard`-tagged items, e.g. jevbench-public",
    "order_shift_max": "`--suite order`",
    "overconfidence_rate": "`--suite underdetermined`",
    "mean_max_prob": "`--suite underdetermined`",
    "uniform_deviation": "`--suite underdetermined`",
    "ko_en_gap": "`--suite massive-ko --suite massive-en`",
    "pair_agreement": "`--suite massive-ko --suite massive-en`",
    "cost_per_1k": "`--gpu-cost-per-hour` for a local model",
}


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _table(headers: list[str], rows: list[list[str]], align: str | None = None) -> list[str]:
    align = align or "l" + "r" * (len(headers) - 1)
    marks = {"l": "---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(marks[a] for a in align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _when(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    local = stamp.astimezone()
    return f"{stamp.strftime('%Y-%m-%d %H:%M:%S')} UTC ({local.strftime('%Y-%m-%d %H:%M:%S UTC%z')} local)"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def _accuracy_cell(metrics: dict[str, Any], details: dict[str, Any]) -> str:
    acc = metrics.get("accuracy")
    if acc is None:
        return "-"
    ci = details.get("accuracy_ci95")
    return f"{acc:.4f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{acc:.4f}"


def _run_section(data: dict[str, Any]) -> list[str]:
    env = data.get("environment", {})
    run = data.get("run") or {}
    options = run.get("options") or {}
    rows = [
        ["Started", _when(run.get("started_at"))],
        ["Finished", _when(run.get("finished_at") or env.get("run_at"))],
        ["Duration", _duration(run.get("wall_seconds"))],
        ["Command", f"`{run['command']}`" if run.get("command") else "-"],
        ["omj version", _num(env.get("omj_version"))],
        ["Git commit", _num(env.get("git_commit"))],
    ]
    lines = ["## 1. Run", ""] + _table(["Field", "Value"], rows, "ll") + [""]
    set_options = {k: v for k, v in options.items() if v not in (None, "", [], False)}
    if set_options:
        opt_rows = [[f"`{k}`", ", ".join(map(str, v)) if isinstance(v, list) else ("yes" if v is True else str(v))]
                    for k, v in set_options.items()]
        lines += ["Options set for this run:", ""] + _table(["Option", "Value"], opt_rows, "ll") + [""]
    return lines


def _model_section(data: dict[str, Any]) -> list[str]:
    env = data.get("environment", {})
    options = (data.get("run") or {}).get("options") or {}
    rows = [
        ["Backend", _num(env.get("backend"))],
        ["Model", f"`{env.get('model')}`" if env.get("model") else "-"],
        ["Revision", _num(env.get("revision") or None)],
        ["Adapter", f"`{options['adapter']}`" if options.get("adapter") else "-"],
        ["Quantization", _num(env.get("quant") or None)],
        ["Route", _num(env.get("route"))],
        ["Calibration", f"`{options['calibration']}`" if options.get("calibration") else ("fitted in this run" if data.get("calibration") else "none")],
    ]
    lines = ["## 2. Model", ""] + _table(["Field", "Value"], rows, "ll") + [""]
    hardware = env.get("hardware") or {}
    if hardware:
        hw_rows = [[k, "-" if v is None else f"{v:g}" if isinstance(v, float) else str(v)] for k, v in hardware.items()]
        lines += ["Hardware:", ""] + _table(["Field", "Value"], hw_rows, "ll") + [""]
    return lines


def _summary_section(data: dict[str, Any]) -> list[str]:
    details = data.get("details") or {}
    headers = ["Suite", "Items", "Scored", "Accuracy [95% CI]", "Hard acc ↑", "ECE ↓", "Brier ↓", "NLL ↓", "Cov@5% ↑", "p50 / p95 ms"]
    rows = []
    for name, m in data["suites"].items():
        d = details.get(name, {})
        p50, p95 = m.get("latency_p50_ms"), m.get("latency_p95_ms")
        rows.append(
            [
                name,
                _num(m.get("n")),
                _num(d.get("n_scored")),
                _accuracy_cell(m, d),
                _num(m.get("hard_accuracy")),
                _num(m.get("ece")),
                _num(m.get("brier")),
                _num(m.get("nll")),
                _num(m.get("coverage_at_risk")),
                "-" if p50 is None else f"{p50:.0f} / {p95:.0f}",
            ]
        )
    lines = ["## 3. Summary", ""] + _table(headers, rows) + [""]
    lines.append("↑ higher is better, ↓ lower is better. The interval is a 95% Wilson interval over scored answers.")
    lines.append("")
    return lines


def _reference_section(data: dict[str, Any]) -> list[str]:
    from omj.bench.reference import verdict

    reference = data.get("reference")
    if not reference or not reference.get("suites"):
        return []
    details = data.get("details") or {}
    rows = []
    for suite, info in reference["suites"].items():
        mine, jev = data["suites"].get(suite, {}), info["metrics"]
        ci = (details.get(suite) or {}).get("accuracy_ci95")

        def diff(key: str, direction: str, use_ci: bool = False) -> str:
            value, ref = mine.get(key), jev.get(key)
            if value is None or ref is None:
                return "-"
            return f"{value - ref:+.3f} {verdict(value, ref, ci if use_ci else None, direction)}".rstrip()

        if info["source"] == "bundled":
            source = f"bundled Jev 1.13, {info['measured_at']}"
        else:
            source = f"{info['source']} `{info['run']}`, {info['measured_at']}"
        rows.append([suite, _num(jev.get("accuracy"), 3), diff("accuracy", "up", True), _num(jev.get("ece"), 3),
                     diff("ece", "down"), diff("brier", "down"), diff("hard_accuracy", "up"), source])
    lines = ["### Compared with Jev", ""]
    lines += _table(["Suite", "Jev accuracy", "Accuracy Δ", "Jev ECE", "ECE Δ", "Brier Δ", "Hard Δ", "Jev numbers from"], rows, "lrrrrrrl")
    lines += ["", "≈ Jev lies inside this run's 95% interval (not a significant difference) · ▲ better · ▼ worse."]
    if reference.get("note"):
        lines += ["", f"Bundled values: {reference['note']}"]
    return lines + [""]


def _suite_section(index: int, name: str, metrics: dict[str, Any], details: dict[str, Any]) -> list[str]:
    lines = [f"### 4.{index} {name}", ""]
    facts = [
        f"{_num(metrics.get('n'))} answers",
        f"{_num(metrics.get('n_valid'))} valid",
        f"{_num(metrics.get('n_unattempted'))} unattempted",
    ]
    if details:
        facts += [
            f"{_num(details.get('n_errors'))} errors",
            f"wall time {_duration(details.get('wall_seconds'))}",
        ]
        if details.get("items_per_second"):
            facts.append(f"{details['items_per_second']:.2f} items/s")
    lines += ["- " + ", ".join(facts)]
    if details.get("stopped_early"):
        lines.append(f"- Stopped early: {details.get('stop_reason') or 'unknown reason'}")
    lines.append("")

    measured, missing = [], []
    for key, (label, direction, meaning) in METRIC_INFO.items():
        value = metrics.get(key)
        if value is None:
            missing.append(key)
            continue
        cell = _num(value, 1 if key.startswith("latency") else 4)
        if key == "accuracy" and details.get("accuracy_ci95"):
            lo, hi = details["accuracy_ci95"]
            cell += f" [{lo:.3f}, {hi:.3f}]"
        measured.append([label, cell, ARROW[direction], meaning])
    if measured:
        lines += _table(["Metric", "Value", "Better", "Meaning"], measured, "lrcl") + [""]
    if missing:
        parts = [f"`{k}`" + (f" ({HOW_TO_MEASURE[k]})" if k in HOW_TO_MEASURE else "") for k in missing]
        lines += ["Not measured for this suite: " + ", ".join(parts) + ".", ""]

    by_kind = details.get("by_kind") or {}
    if len(by_kind) > 1:
        rows = [[k, str(s["n"]), _num(s["accuracy"]), _num(s["mean_confidence"])] for k, s in by_kind.items()]
        lines += ["By question type:", ""] + _table(["Type", "N", "Accuracy", "Mean confidence"], rows) + [""]

    by_tag = details.get("by_tag") or {}
    if by_tag:
        rows = [
            [t, str(s["n"]), _num(s["accuracy"]), _num(s["mean_confidence"]), f"{s['mean_confidence'] - s['accuracy']:+.3f}"]
            for t, s in by_tag.items()
        ]
        lines += ["By tag (tags on at least 5 scored answers; Conf - acc > 0 means overconfident):", ""]
        lines += _table(["Tag", "N", "Accuracy", "Mean confidence", "Conf - acc"], rows) + [""]

    reliability = details.get("reliability") or []
    if reliability:
        rows = [
            [f"{b['range'][0]:.1f} to {b['range'][1]:.1f}", str(b["n"]), _num(b["mean_confidence"]), _num(b["accuracy"]), f"{b['mean_confidence'] - b['accuracy']:+.3f}"]
            for b in reliability
        ]
        lines += ["Reliability (top-1 confidence, 10 bins; a well calibrated model has gap near 0):", ""]
        lines += _table(["Confidence", "N", "Mean confidence", "Accuracy", "Gap"], rows) + [""]
    return lines


def _calibration_section(calibration: dict[str, Any] | None) -> list[str]:
    if not calibration:
        return []
    rows = [[k, f"`{v}`" if isinstance(v, (dict, list)) else _num(v)] for k, v in calibration.items()]
    return ["## 5. Calibration fitted in this run", ""] + _table(["Field", "Value"], rows, "ll") + [""]


def render_markdown(data: dict[str, Any]) -> str:
    env = data.get("environment", {})
    suites = list(data["suites"])
    lines = [
        "# omj benchmark report",
        "",
        f"`{env.get('model') or env.get('backend')}` on {', '.join(f'`{s}`' for s in suites)}, "
        f"finished {_when((data.get('run') or {}).get('finished_at') or env.get('run_at'))}.",
        "",
    ]
    lines += _run_section(data)
    lines += _model_section(data)
    lines += _summary_section(data)
    lines += _reference_section(data)
    lines += ["## 4. Suites", ""]
    details = data.get("details") or {}
    for index, (name, metrics) in enumerate(data["suites"].items(), 1):
        lines += _suite_section(index, name, metrics, details.get(name) or {})
    lines += _calibration_section(data.get("calibration"))
    lines += [
        "## Files",
        "",
        "- `report.json`: every metric, including the ones not measured (null)",
        "- `reliability.svg`: reliability diagram over all scored answers",
        "",
    ]
    return "\n".join(lines)
