"""`omj bench`: run decision suites against a backend or endpoint and report metrics (REQ-032).

Wires the pieces built by the earlier tasks into one command: suite loaders ->
runner -> metrics -> optional temperature fit -> report.json/report.md/
reliability.svg (+ an optional JevBench submission bundle).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

import typer

from omj.backends.registry import create_backend
from omj.bench.metrics import compute_metrics
from omj.bench.report import collect_environment, write_report
from omj.backends.adapter_info import follow_adapter_base
from omj.bench.details import suite_details
from omj.bench.progress import BenchProgress, progress_enabled
from omj.bench.terminal import print_results
from omj.bench.runner import BackendTarget, EndpointTarget, Target, order_groups_from_rows, run_suite
from omj.bench.submission import write_submission
from omj.bench.suites import SUITE_NAMES, load_suite
from omj.bench.suites.base import DecisionItem
from omj.bench.suites.fromlog import load_from_log
from omj.bench.suites.order import make_order_suite
from omj.bench.svg import bins_from_rows, reliability_svg
from omj.bench.temperature import fit_and_report
from omj.bench.types import Metrics, ResultRow
from omj.config import BackendSection, Config, config_path, load_config, omj_home
from omj.errors import ErrorCode, OmjError
from omj.gateway.calibration import load_calibration

ORDER_SUITE = "order"
FROM_LOG_SUITE = "from-log"
ALL_SUITES = "all"
PAIR_SUITE = "massive-pair"
DEFAULT_SUITE = "omj-smoke"
SELECTABLE_SUITES: tuple[str, ...] = (*SUITE_NAMES, ORDER_SUITE, ALL_SUITES)
#: TypeSafe Jev input price in USD per million prompt tokens (design.md 4.3).
TYPESAFE_PRICE_PER_MTOK = 0.042
RUN_DIR_FORMAT = "%Y%m%dT%H%M%SZ"

SuiteLoader = Callable[[str], list[DecisionItem]]


@dataclass
class BenchOptions:
    suites: list[str] = field(default_factory=lambda: [DEFAULT_SUITE])
    backend: str | None = None
    endpoint: str | None = None
    adapter: str | None = None
    calibration: str | None = None
    fit_temperature: bool = False
    from_log: Path | None = None
    gpu_cost_per_hour: float | None = None
    submission: bool = False
    out: Path | None = None
    json_output: bool = False
    brief: bool = False


def _require_adapter_dir(path: Path) -> str:
    """Validate a `--adapter` directory at command time, before any backend loads (REQ-012)."""
    try:
        # imported here: semif pulls transformers, which `omj bench --backend mock` must not need.
        from omj.backends.semif import validate_adapter_dir
    except ImportError as exc:
        raise OmjError(
            ErrorCode.E_BACKEND, f"--adapter needs the 'semif' extra: {exc}"
        ) from exc
    return str(validate_adapter_dir(path))


@dataclass
class ResolvedTarget:
    target: Target
    backend_name: str
    model: str
    route: str


def _resolve_suite_names(opts: BenchOptions) -> list[str]:
    if opts.from_log is not None:
        return [FROM_LOG_SUITE]

    requested: list[str] = []
    for name in opts.suites or [DEFAULT_SUITE]:
        if name == ALL_SUITES:
            requested.extend([*SUITE_NAMES, ORDER_SUITE])
            continue
        if name not in SELECTABLE_SUITES:
            raise OmjError(
                ErrorCode.E_CONFIG,
                f"unknown suite {name!r} (expected one of {', '.join(SELECTABLE_SUITES)})",
            )
        requested.append(name)

    ordered: list[str] = []
    for name in requested:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _prepare_out_dir(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now(timezone.utc).strftime(RUN_DIR_FORMAT)
        target = omj_home() / "runs" / stamp
    else:
        target = Path(path)

    if target.exists():
        if not target.is_dir():
            raise OmjError(ErrorCode.E_CONFIG, f"--out {target} exists and is not a directory")
        if any(target.iterdir()):
            raise OmjError(
                ErrorCode.E_CONFIG,
                f"--out {target} already exists and is not empty",
                hint="Pass a new --out directory, or omit --out to write a timestamped folder under ~/.omj/runs.",
            )
    target.mkdir(parents=True, exist_ok=True)
    return target


def _default_target_factory(opts: BenchOptions, config: Config) -> ResolvedTarget:
    if opts.endpoint is not None:
        return ResolvedTarget(
            target=EndpointTarget(opts.endpoint),
            backend_name="endpoint",
            model="jev-latest",
            route="direct",
        )

    name = opts.backend or ""
    backend_data = config.backend.model_dump(mode="json")
    backend_data["name"] = name
    if opts.adapter:
        backend_data["adapter"] = opts.adapter
        note = follow_adapter_base(backend_data, opts.adapter)
        if note:
            print(note, file=sys.stderr)
    backend_cfg = BackendSection.model_validate(backend_data)
    backend = create_backend(name)
    backend.load(backend_cfg)
    route = config.backend.provider if name == "typesafe" else "local"
    calibration = load_calibration(opts.calibration) if opts.calibration else None
    return ResolvedTarget(
        target=BackendTarget(backend, calibration=calibration),
        backend_name=name,
        model=backend.model_id,
        route=route,
    )


def _load_items(
    name: str,
    opts: BenchOptions,
    suite_loader: SuiteLoader,
    loaded: dict[str, list[DecisionItem]],
) -> list[DecisionItem]:
    if name == FROM_LOG_SUITE:
        assert opts.from_log is not None
        return load_from_log(opts.from_log)
    if name == ORDER_SUITE:
        base = next(
            (items for suite, items in loaded.items() if suite != ORDER_SUITE and items),
            None,
        )
        if base is None:
            base = suite_loader(DEFAULT_SUITE)
        items, _groups = make_order_suite(base)
        return items
    return suite_loader(name)


def _price_per_mtok(backend_name: str) -> float | None:
    return TYPESAFE_PRICE_PER_MTOK if backend_name == "typesafe" else None


def _fit_calibration(
    suite_rows: dict[str, list[ResultRow]],
    *,
    backend_name: str,
    model: str,
) -> dict[str, Any] | None:
    for suite_name, rows in suite_rows.items():
        if not any(row.logits is not None and row.expected is not None for row in rows):
            continue
        model_safe = model.replace("/", "__")
        out_path = omj_home() / "calibration" / f"{backend_name}-{model_safe}.json"
        cal, summary = fit_and_report(
            rows,
            backend=backend_name,
            model=model,
            suite=suite_name,
            out_path=out_path,
        )
        return {
            "suite": suite_name,
            "path": str(out_path),
            "temperatures": cal.temperatures,
            **summary,
        }
    return None


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _command_line(opts: BenchOptions) -> str:
    """The `omj bench` invocation that reproduces this run (report header)."""
    parts = ["omj", "bench"]
    parts += ["--backend", opts.backend] if opts.backend else ["--endpoint", str(opts.endpoint)]
    if opts.from_log is not None:
        parts += ["--from-log", str(opts.from_log)]
    else:
        for suite in opts.suites or [DEFAULT_SUITE]:
            parts += ["--suite", suite]
    for flag, value in (("--adapter", opts.adapter), ("--calibration", opts.calibration),
                        ("--gpu-cost-per-hour", opts.gpu_cost_per_hour)):
        if value is not None:
            parts += [flag, str(value)]
    if opts.fit_temperature:
        parts.append("--fit-temperature")
    if opts.submission:
        parts.append("--submission")
    return " ".join(f'"{p}"' if " " in p else p for p in parts)


def _run_options(opts: BenchOptions, suite_names: list[str], out_dir: Path) -> dict[str, Any]:
    return {
        "suites": suite_names,
        "backend": opts.backend,
        "endpoint": opts.endpoint,
        "adapter": opts.adapter,
        "calibration": opts.calibration,
        "fit_temperature": opts.fit_temperature,
        "from_log": str(opts.from_log) if opts.from_log is not None else None,
        "gpu_cost_per_hour": opts.gpu_cost_per_hour,
        "submission": opts.submission,
        "out": str(out_dir),
    }


def _configured_backend() -> str:
    """The backend `omj init` wrote to config.toml, used when neither --backend nor --endpoint is given."""
    path = config_path()
    if not path.exists():
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"no --backend given and no config at {path}",
            hint="Run 'uv run omj init' first, or pass --backend NAME (e.g. --backend mock).",
        )
    name = load_config(path).backend.name
    if not name:
        raise OmjError(ErrorCode.E_CONFIG, f"{path} has no [backend] name", hint="Rerun 'uv run omj init' or pass --backend NAME.")
    return name


def run_bench(
    opts: BenchOptions,
    *,
    target_factory: Callable[[BenchOptions, Config], ResolvedTarget] | None = None,
    suite_loader: SuiteLoader = load_suite,
    out: TextIO | None = None,
    progress: BenchProgress | None = None,
) -> dict[str, Any]:
    """Run every requested suite and write the report bundle; returns the report data."""
    stream = out if out is not None else sys.stdout
    if progress is None:
        progress = BenchProgress(enabled=out is None and progress_enabled())

    if opts.backend is not None and opts.endpoint is not None:
        raise OmjError(ErrorCode.E_CONFIG, "pass at most one of --backend NAME or --endpoint URL")
    if opts.backend is None and opts.endpoint is None:
        opts = replace(opts, backend=_configured_backend())

    if opts.adapter is not None:
        if opts.endpoint is not None:
            raise OmjError(
                ErrorCode.E_CONFIG,
                "--adapter can only be used together with --backend, not --endpoint",
            )
        # Validated here (before any backend loads) so a bad path fails fast
        # regardless of which target_factory ends up consuming opts.adapter.
        _require_adapter_dir(Path(opts.adapter))

    if opts.calibration is not None:
        if opts.endpoint is not None:
            raise OmjError(
                ErrorCode.E_CONFIG,
                "--calibration applies to --backend only; an endpoint applies its own calibration",
            )
        if opts.fit_temperature:
            raise OmjError(ErrorCode.E_CONFIG, "--calibration and --fit-temperature cannot be combined")
        load_calibration(opts.calibration)  # fail fast on a missing or malformed file

    suite_names = _resolve_suite_names(opts)
    started_at = datetime.now(timezone.utc)
    config = load_config()
    out_dir = _prepare_out_dir(opts.out)
    what = f"backend {opts.backend}" if opts.backend is not None else f"endpoint {opts.endpoint}"
    progress.note(f"Loading {what} (a local model can take a minute)...")
    resolved = (target_factory or _default_target_factory)(opts, config)

    loaded: dict[str, list[DecisionItem]] = {}
    suite_rows: dict[str, list[ResultRow]] = {}
    suite_metrics: dict[str, Metrics] = {}
    details: dict[str, dict[str, Any]] = {}
    all_rows: list[ResultRow] = []
    price_per_mtok = _price_per_mtok(resolved.backend_name)

    try:
        with progress:
            for name in suite_names:
                items = _load_items(name, opts, suite_loader, loaded)
                loaded[name] = items
                result = run_suite(
                    items,
                    resolved.target,
                    suite_name=name,
                    backend_name=resolved.backend_name,
                    model=resolved.model,
                    on_item=progress.suite(name, len(items)),
                )
                suite_rows[name] = result.rows
                all_rows.extend(result.rows)
                details[name] = suite_details(
                    result.rows,
                    n_items=len(items),
                    wall_seconds=result.wall_seconds,
                    n_errors=result.n_errors,
                    stopped_early=result.stopped_early,
                    stop_reason=result.stop_reason,
                )
                suite_metrics[name] = compute_metrics(
                    result.rows,
                    gpu_cost_per_hour=opts.gpu_cost_per_hour,
                    total_wall_seconds=result.wall_seconds,
                    price_input_per_mtok=price_per_mtok,
                    input_tokens_total=result.input_tokens_total,
                    order_groups=order_groups_from_rows(result.rows) if name == ORDER_SUITE else None,
                )
    finally:
        closer = getattr(resolved.target, "close", None)
        if callable(closer):
            closer()

    # ko_en_gap and pair_agreement only exist across the two MASSIVE halves, so
    # they need one extra pass over both suites' rows together.
    if "massive-ko" in suite_rows and "massive-en" in suite_rows:
        suite_metrics[PAIR_SUITE] = compute_metrics(suite_rows["massive-ko"] + suite_rows["massive-en"])

    calibration = None
    if opts.fit_temperature:
        calibration = _fit_calibration(
            suite_rows,
            backend_name=resolved.backend_name,
            model=resolved.model,
        )

    finished_at = datetime.now(timezone.utc)
    env = collect_environment(config, resolved.backend_name, resolved.model, resolved.route)
    json_path, _md_path = write_report(
        out_dir,
        suite_metrics=suite_metrics,
        env=env,
        calibration=calibration,
        run={
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "wall_seconds": (finished_at - started_at).total_seconds(),
            "command": _command_line(opts),
            "options": _run_options(opts, suite_names, out_dir),
        },
        details=details,
    )

    report_text = json_path.read_text(encoding="utf-8")
    report_data = json.loads(report_text)

    svg_rows = [row for row in all_rows if row.expected is not None]
    (out_dir / "reliability.svg").write_text(
        reliability_svg(bins_from_rows(svg_rows)),
        encoding="utf-8",
        newline="\n",
    )

    if opts.submission:
        write_submission(
            out_dir,
            all_rows,
            model=resolved.model,
            backend=resolved.backend_name,
            price_input_per_m=price_per_mtok,
        )

    if opts.json_output:
        stream.write(report_text)
    else:
        print_results(stream, report_data, out_dir, brief=opts.brief)

    return report_data


def register(app: typer.Typer) -> None:
    @app.command()
    def bench(
        suite: list[str] = typer.Option(
            [DEFAULT_SUITE],
            "--suite",
            help=f"Suite to run; repeatable. One of: {', '.join(SELECTABLE_SUITES)}.",
        ),
        backend: str | None = typer.Option(
            None, "--backend", help="Run the suite in-process against this backend (default: the backend in config.toml)."
        ),
        endpoint: str | None = typer.Option(
            None, "--endpoint", help="Run the suite against a running gateway at this base URL."
        ),
        adapter: Path | None = typer.Option(
            None,
            "--adapter",
            help="Path to a PEFT adapter directory to load on top of --backend (semif only).",
        ),
        calibration: Path | None = typer.Option(
            None,
            "--calibration",
            help="Apply a calibration file (per-type temperatures) to --backend logits.",
        ),
        fit_temperature: bool = typer.Option(
            False, "--fit-temperature", help="Fit per-type temperatures and save a calibration file."
        ),
        from_log: Path | None = typer.Option(
            None,
            "--from-log",
            help="Replay decisions from a decision-log directory instead of a suite.",
        ),
        gpu_cost_per_hour: float | None = typer.Option(
            None, "--gpu-cost-per-hour", help="GPU price in USD per hour, used for cost_per_1k."
        ),
        submission: bool = typer.Option(
            False, "--submission", help="Also write a JevBench submission bundle."
        ),
        out: Path | None = typer.Option(
            None, "--out", help="Output directory (default: $OMJ_HOME/runs/<UTC timestamp>)."
        ),
        json_output: bool = typer.Option(
            False, "--json", help="Print report.json to stdout instead of the results."
        ),
        brief: bool = typer.Option(
            False, "--brief", help="Print only the summary table (default: every metric, breakdown and reliability table)."
        ),
    ) -> None:
        """Run a benchmark suite and report calibration/accuracy metrics."""
        run_bench(
            BenchOptions(
                suites=list(suite),
                backend=backend,
                endpoint=endpoint,
                adapter=str(adapter) if adapter is not None else None,
                calibration=str(calibration) if calibration is not None else None,
                fit_temperature=fit_temperature,
                from_log=from_log,
                gpu_cost_per_hour=gpu_cost_per_hour,
                submission=submission,
                out=out,
                json_output=json_output,
                brief=brief,
            )
        )
