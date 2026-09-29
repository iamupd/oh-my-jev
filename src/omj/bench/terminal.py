"""Terminal rendering of a bench report (the data written to report.json).

Default output, top to bottom: a one-block run header, one section per suite
(metrics grouped by what they measure, breakdown by question type and tag,
reliability table), then the cross-suite summary and the report location, so
the answer to "how did it do?" is the last thing on screen.
`brief=True` prints only the summary and the report location.
Colour is used only on a real terminal; logs and pipes get plain text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TextIO

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from omj.bench.report_md import HOW_TO_MEASURE, METRIC_INFO

LOG_WIDTH = 100  # width used when stdout is not a terminal, so logs do not wrap at 80
MAX_WIDTH = 100  # never draw wider than this: frames drawn at full width break when the window is narrowed later
WEAK_MARGIN = 0.10  # a tag this far below the suite accuracy is highlighted
OVERCONFIDENT = 0.05  # confidence minus accuracy above this is highlighted

GROUPS: list[tuple[str, list[str]]] = [
    ("Accuracy", ["accuracy", "chance_corrected_accuracy", "hard_accuracy"]),
    ("Calibration", ["ece", "ece_jevbench", "brier", "nll", "coverage_at_risk"]),
    ("Robustness", ["valid_vector_rate", "valid_vector_rate_lenient", "order_shift_max",
                    "overconfidence_rate", "mean_max_prob", "uniform_deviation", "ko_en_gap", "pair_agreement"]),
    ("Speed and cost", ["latency_p50_ms", "latency_p95_ms", "cost_per_1k"]),
]
SHORT_LABEL = {
    "accuracy": "accuracy",
    "chance_corrected_accuracy": "chance-corrected",
    "hard_accuracy": "hard items",
    "ece": "ECE 15 bins",
    "ece_jevbench": "ECE 10 bins",
    "brier": "Brier",
    "nll": "NLL",
    "coverage_at_risk": "coverage @5% risk",
    "valid_vector_rate": "valid vectors",
    "valid_vector_rate_lenient": "valid (lenient)",
    "order_shift_max": "order shift max",
    "overconfidence_rate": "overconfidence",
    "mean_max_prob": "mean top prob",
    "uniform_deviation": "uniform deviation",
    "ko_en_gap": "KO-EN gap",
    "pair_agreement": "KO-EN agreement",
    "latency_p50_ms": "latency p50",
    "latency_p95_ms": "latency p95",
    "cost_per_1k": "cost / 1k items",
}
ARROW = {"up": "↑", "down": "↓", "zero": "→0"}


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def _value(key: str, value: float, details: dict[str, Any]) -> str:
    if key.startswith("latency"):
        return f"{value:,.1f} ms"
    if key == "cost_per_1k":
        return f"${value:.4f}"
    if key.startswith("valid_vector"):
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _header(data: dict[str, Any]) -> Group:
    env = data.get("environment", {})
    run = data.get("run") or {}
    options = run.get("options") or {}
    title = Text()
    title.append("omj bench  ", style="bold")
    title.append(str(env.get("model") or env.get("backend")), style="bold cyan")
    facts = [env.get("backend"), env.get("quant") or None, env.get("route")]
    line2 = Text("  " + " · ".join(str(f) for f in facts if f), style="dim")
    extra = []
    if options.get("adapter"):
        extra.append(f"adapter {options['adapter']}")
    if options.get("calibration"):
        extra.append(f"calibration {options['calibration']}")
    extra.append(f"duration {_duration(run.get('wall_seconds'))}")
    line3 = Text("  " + "   ".join(extra), style="dim")
    return Group(title, line2, line3)


def _metric_grid(metrics: dict[str, Any], details: dict[str, Any]) -> Table | None:
    columns = []
    for title, keys in GROUPS:
        rows = [(k, metrics[k]) for k in keys if metrics.get(k) is not None]
        if not rows:
            continue
        col = Table.grid(padding=(0, 1))
        col.add_column(style="dim", no_wrap=True)
        col.add_column(justify="right", no_wrap=True)
        col.add_column(style="dim", no_wrap=True)
        col.add_row(Text(title, style="bold"), "", "")
        for key, value in rows:
            col.add_row(SHORT_LABEL[key], _value(key, value, details), ARROW[METRIC_INFO[key][1]])
            if key == "accuracy" and details.get("accuracy_ci95"):
                lo, hi = details["accuracy_ci95"]
                col.add_row("  95% CI", f"{lo:.3f}-{hi:.3f}", "")
        columns.append(col)
    if not columns:
        return None
    # Two groups per row keeps every value intact on an 80-column terminal.
    grid = Table.grid(padding=(0, 6))
    grid.add_column(vertical="top")
    grid.add_column(vertical="top")
    for i in range(0, len(columns), 2):
        pair = columns[i:i + 2]
        if i:
            grid.add_row("", "")
        grid.add_row(*pair, *([""] * (2 - len(pair))))
    return grid


def _heading(title: str) -> Text:
    return Text.assemble(("▍", "cyan"), (f" {title}", "bold cyan"))


def _by_type(details: dict[str, Any]) -> Text | None:
    by_kind = details.get("by_kind") or {}
    if len(by_kind) < 2:
        return None
    return Text("   ".join(f"{k} {s['accuracy']:.3f} (n={s['n']})" for k, s in by_kind.items()))


def _by_tag(details: dict[str, Any], suite_accuracy: float | None) -> Table | None:
    by_tag = details.get("by_tag") or {}
    if not by_tag:
        return None
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold", title_justify="left")
    table.add_column("Tag", no_wrap=True)
    for col in ("N", "Accuracy", "Confidence", "Conf - acc"):
        table.add_column(col, justify="right", no_wrap=True)
    for tag, s in by_tag.items():
        gap = s["mean_confidence"] - s["accuracy"]
        weak = suite_accuracy is not None and s["accuracy"] < suite_accuracy - WEAK_MARGIN
        table.add_row(
            Text(tag, style="red" if weak else ""),
            str(s["n"]),
            Text(f"{s['accuracy']:.3f}", style="red" if weak else ""),
            f"{s['mean_confidence']:.3f}",
            Text(f"{gap:+.3f}", style="yellow" if gap > OVERCONFIDENT else "dim"),
        )
    return table


def _reliability(details: dict[str, Any]) -> Table | None:
    bins = details.get("reliability") or []
    if not bins:
        return None
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Confidence bin", no_wrap=True)
    for col in ("N", "Confidence", "Accuracy", "Gap"):
        table.add_column(col, justify="right", no_wrap=True)
    table.add_column("", no_wrap=True)
    total = max(b["n"] for b in bins)
    for b in bins:
        gap = b["mean_confidence"] - b["accuracy"]
        bar = "█" * max(1, round(12 * b["n"] / total))
        table.add_row(
            f"{b['range'][0]:.1f}-{b['range'][1]:.1f}",
            str(b["n"]),
            f"{b['mean_confidence']:.3f}",
            f"{b['accuracy']:.3f}",
            Text(f"{gap:+.3f}", style="yellow" if gap > OVERCONFIDENT else "dim"),
            Text(bar, style="cyan"),
        )
    return table


def _suite(name: str, metrics: dict[str, Any], details: dict[str, Any]) -> Panel:
    """One suite in its own frame; each block inside starts with a coloured heading."""
    facts = [f"{metrics.get('n')} answers"]
    if details:
        facts += [f"{details.get('n_errors', 0)} errors", _duration(details.get("wall_seconds"))]
        if details.get("items_per_second"):
            facts.append(f"{details['items_per_second']:.2f} items/s")
    parts: list[RenderableType] = []
    if details.get("stopped_early"):
        parts += [Text(f"Stopped early: {details.get('stop_reason')}", style="bold red"), Text()]
    grid = _metric_grid(metrics, details)
    if grid is not None:
        parts += [_heading("Metrics"), grid]
        missing = [SHORT_LABEL[k] for k in METRIC_INFO if metrics.get(k) is None and k in HOW_TO_MEASURE]
        if missing:
            parts += [Text(), Text("Not measured: " + ", ".join(missing) + " (see report.md)", style="dim")]
    for title, block in (("By question type", _by_type(details)),
                         ("By tag", _by_tag(details, metrics.get("accuracy"))),
                         ("Reliability", _reliability(details))):
        if block is not None:
            parts += [Text(), _heading(title), block]
    return Panel(
        Group(*parts),
        title=Text(f" {name} ", style="bold"),
        title_align="left",
        subtitle=Text(" " + " · ".join(facts) + " ", style="dim"),
        subtitle_align="right",
        box=box.ROUNDED,
        border_style="blue",
        padding=(1, 2),
        expand=False,
    )


def _summary(data: dict[str, Any]) -> Table:
    details = data.get("details") or {}
    # Short headers and two-space gaps keep the table inside the frame on an 80-column terminal.
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold",
                  padding=(0, 1))
    table.add_column("Suite", no_wrap=True)
    for title in ("N", "Acc", "95% CI", "Hard", "ECE↓", "Brier↓", "p50ms"):
        table.add_column(title, justify="right", no_wrap=True)

    def cell(value: float | None, fmt: str = ".3f") -> str:
        return "-" if value is None else format(value, fmt)

    for name, m in data["suites"].items():
        ci = (details.get(name) or {}).get("accuracy_ci95")
        table.add_row(
            Text(name, style="bold"),
            str(m.get("n")),
            Text(cell(m.get("accuracy")), style="bold"),
            f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "-",
            cell(m.get("hard_accuracy")),
            cell(m.get("ece")),
            cell(m.get("brier")),
            cell(m.get("latency_p50_ms"), ".1f"),
        )
    return table


def _console(stream: TextIO) -> Console:
    is_tty = hasattr(stream, "isatty") and stream.isatty()
    if not is_tty:
        return Console(file=stream, highlight=False, soft_wrap=False, width=LOG_WIDTH, force_terminal=False)
    width = min(Console(file=stream).width, MAX_WIDTH)
    return Console(file=stream, highlight=False, soft_wrap=False, width=width)


def _source_line(reference: dict[str, Any]) -> str:
    """Where each suite's Jev numbers come from, grouped: 'bundled Jev 1.13 (2026-09-22): a, b'."""
    groups: dict[str, list[str]] = {}
    for suite, info in reference.get("suites", {}).items():
        if info["source"] == "bundled":
            label = f"bundled Jev 1.13, measured {info['measured_at']}"
        elif info["source"] == "measured now":
            label = f"measured now ({info['run']})"
        else:
            label = f"{info['source']} {info['run']} ({info['measured_at']})"
        groups.setdefault(label, []).append(suite)
    return "; ".join(f"{label}: {', '.join(suites)}" for label, suites in groups.items())


def _reference_table(data: dict[str, Any]) -> Table | None:
    from omj.bench.reference import verdict

    reference = data.get("reference")
    if not reference or not reference.get("suites"):
        return None
    details = data.get("details") or {}
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold", padding=(0, 1))
    table.add_column("vs Jev", no_wrap=True)
    for title in ("Jev acc", "Acc Δ", "Jev ECE", "ECE Δ", "Brier Δ"):
        table.add_column(title, justify="right", no_wrap=True)

    def diff(key: str, direction: str, ci=None) -> Text:
        value, ref = mine.get(key), jev.get(key)
        if value is None or ref is None:
            return Text("-", style="dim")
        mark = verdict(value, ref, ci, direction)
        style = {"▲": "green", "▼": "red"}.get(mark, "dim")
        return Text(f"{value - ref:+.3f} {mark}".rstrip(), style=style)

    for suite, info in reference["suites"].items():
        mine, jev = data["suites"].get(suite, {}), info["metrics"]
        ci = (details.get(suite) or {}).get("accuracy_ci95")
        jev_acc = jev.get("accuracy")
        jev_ece = jev.get("ece")
        table.add_row(
            Text(suite, style="bold"),
            "-" if jev_acc is None else f"{jev_acc:.3f}",
            diff("accuracy", "up", ci),
            "-" if jev_ece is None else f"{jev_ece:.3f}",
            diff("ece", "down"),
            diff("brier", "down"),
        )
    return table


def print_results(stream: TextIO, data: dict[str, Any], out_dir: Path, *, brief: bool = False) -> None:
    console = _console(stream)
    if not brief:
        console.print(_header(data))
        console.print()
        details = data.get("details") or {}
        for name, metrics in data["suites"].items():
            console.print(_suite(name, metrics, details.get(name) or {}))
            console.print()
    parts: list[RenderableType] = [_summary(data)]
    ref_table = _reference_table(data)
    if ref_table is not None:
        parts += [Text(), ref_table, Text("≈ Jev inside this run's 95% CI · ▲ better · ▼ worse", style="dim"),
                  Text("Jev: " + _source_line(data["reference"]), style="dim")]
    report = Group(
        *parts,
        Text(),
        Text.assemble(("Report  ", "bold"), str(out_dir)),
        Text("        report.md · report.json · reliability.svg", style="dim"),
    )
    console.print(Panel(report, title=Text(" Summary ", style="bold"), title_align="left",
                        box=box.DOUBLE, border_style="green", padding=(1, 1), expand=False))


def _cmp_value(metric: str, value: float) -> str:
    if metric.startswith("latency"):
        return f"{value:,.1f}"
    if metric == "cost_per_1k":
        return f"{value:.4f}"
    return f"{value:.3f}"


def _cmp_delta(metric: str, delta: float) -> Text:
    direction = METRIC_INFO[metric][1]
    text = f"{delta:+,.1f}" if metric.startswith("latency") else f"{delta:+.4f}" if metric == "cost_per_1k" else f"{delta:+.3f}"
    if float(text.replace(",", "")) == 0:  # rounds to zero: no sign, no colour
        return Text(f"({text.lstrip('+-')})", style="dim")
    better = delta > 0 if direction == "up" else delta < 0
    return Text(f"({text})", style="green" if better else "red")


def print_compare(stream: TextIO, result: dict[str, Any], written: list[Path]) -> None:
    """Terminal view of `omj compare`: the reports, one framed table per shared suite, then the files."""
    console = _console(stream)
    labels: list[str] = result["labels"]

    reports = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    for col in ("", "Label", "Backend"):
        reports.add_column(col, no_wrap=True)
    reports.add_column("Model", overflow="fold")  # long ids wrap instead of being cut on narrow terminals
    reports.add_column("Run at (UTC)", no_wrap=True)
    for i, (label, info) in enumerate(zip(labels, result["reports"])):
        run_at = str(info.get("run_at") or "-").replace("T", " ")[:16]
        reports.add_row("base" if i == 0 else str(i), Text(label, style="bold"), str(info.get("backend") or "-"),
                        str(info.get("model") or "-"), run_at)
    console.print(Panel(reports, title=Text(" Reports ", style="bold"), title_align="left",
                        box=box.ROUNDED, border_style="blue", padding=(1, 2), expand=False))
    console.print()

    for suite, data in result["suites"].items():
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
        table.add_column("Metric", no_wrap=True)
        for label in labels:
            table.add_column(label, justify="right", no_wrap=True)
        deltas = data.get("delta", {})
        rows = 0
        for metric, values in data.items():
            if metric == "delta" or metric not in METRIC_INFO or all(v is None for v in values):
                continue
            direction = METRIC_INFO[metric][1]
            present = [v for v in values if v is not None]
            best = (max(present) if direction == "up" else min(present)) if len(present) > 1 and direction != "zero" else None
            cells: list[Text] = []
            for i, value in enumerate(values):
                if value is None:
                    cells.append(Text("n/a", style="dim"))
                    continue
                cell = Text(_cmp_value(metric, value), style="bold" if best is not None and value == best else "")
                delta = (deltas.get(metric) or [None] * len(values))[i]
                if i and delta is not None:
                    cell.append(" ")
                    cell.append_text(_cmp_delta(metric, delta))
                cells.append(cell)
            table.add_row(Text(f"{SHORT_LABEL.get(metric, metric)} {ARROW[direction]}"), *cells)
            rows += 1
        if rows:
            console.print(Panel(table, title=Text(f" {suite} ", style="bold"), title_align="left",
                                box=box.ROUNDED, border_style="blue", padding=(1, 2), expand=False))
            console.print()

    console.print(Text("Δ is against the base report: green is better, red is worse; bold is the best value. "
                       "n/a: not measured by that report.", style="dim"))
    for path in written:
        console.print(Text.assemble(("Wrote  ", "bold"), str(path)))
