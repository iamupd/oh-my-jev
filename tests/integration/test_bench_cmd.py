"""End-to-end `omj bench` runs over the mock backend and an injected HTTP endpoint. # REQ-032"""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from omj.bench.runner import BackendTarget, EndpointTarget, RunResult, run_suite
from omj.bench.suites import load_suite
from omj.bench.suites.base import DecisionItem
from omj.cli.bench_cmd import BenchOptions, ResolvedTarget, run_bench
from omj.cli.main import app
from omj.errors import ErrorCode, OmjError

runner = CliRunner()


def _combined_output(result) -> str:
    output = result.stdout
    try:
        output += result.stderr
    except ValueError:
        pass  # stdout and stderr were captured as one stream
    return output


def _invoke(args: list[str]):
    return runner.invoke(app, ["bench", *args])


def _endpoint_handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content.decode("utf-8"))
    answers: dict[str, dict] = {}
    for qid, question in payload["questions"].items():
        kind = question["type"]
        if kind == "noul":
            answers[qid] = {"type": "noul", "noul": 0.75}
        elif kind == "choice":
            keys = list(question["criteria"])
            head = 0.6
            rest = (1.0 - head) / (len(keys) - 1)
            probs = [head] + [rest] * (len(keys) - 1)
            answers[qid] = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": dict(zip(keys, probs)),
                "confidence": 0.5,
            }
        else:
            keys = [str(i) for i in range(len(question["criteria"]))]
            probs = [1.0 / len(keys)] * len(keys)
            answers[qid] = {
                "type": "score",
                "score": 1.0,
                "legend": dict(zip(keys, question["criteria"])),
                "probabilities": dict(zip(keys, probs)),
                "confidence": 0.1,
            }
    return httpx.Response(
        200,
        json={
            "model": "jev-latest",
            "answers": answers,
            "usage": {"input_tokens": 11, "output_tokens": 0},
        },
    )


def _endpoint_target(handler=_endpoint_handler) -> EndpointTarget:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8799")
    return EndpointTarget("http://127.0.0.1:8799", client=client)


class _AuthFailingTarget:
    """Fails the second item with E_AUTH; every other item answers uniformly."""

    def __init__(self, fail_on: int = 2, code: ErrorCode = ErrorCode.E_AUTH) -> None:
        self.fail_on = fail_on
        self.code = code
        self.calls = 0

    def decide_item(self, item: DecisionItem) -> dict[str, tuple]:
        self.calls += 1
        if self.calls >= self.fail_on:
            raise OmjError(self.code, "upstream rejected the API key")
        answers: dict[str, tuple] = {}
        for qid, question in item.questions.items():
            kind = question["type"]
            if kind == "noul":
                keys = ["yes", "no"]
            elif kind == "choice":
                keys = list(question["criteria"])
            else:
                keys = [str(i) for i in range(len(question["criteria"]))]
            probs = [1.0 / len(keys)] * len(keys)
            answers[qid] = (kind, keys, probs, None, {})
        return answers


def test_bench_mock_backend_writes_all_artifacts(tmp_path: Path) -> None:
    # REQ-032
    out_dir = tmp_path / "run"
    result = _invoke(
        [
            "--backend",
            "mock",
            "--suite",
            "omj-smoke",
            "--suite",
            "underdetermined",
            "--suite",
            "order",
            "--out",
            str(out_dir),
        ]
    )

    assert result.exit_code == 0, _combined_output(result)
    for name in ("report.json", "report.md", "reliability.svg"):
        assert (out_dir / name).is_file(), f"{name} was not written"

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert set(report["suites"]) == {"omj-smoke", "underdetermined", "order"}
    assert report["suites"]["omj-smoke"]["n"] == 30
    assert "badge" not in report and "badge_basis" not in report
    assert report["environment"]["backend"] == "mock"
    assert report["environment"]["route"] == "local"
    assert report["suites"]["order"]["order_shift_max"] is not None
    # every item must be attempted and answered: a backend that cannot read a
    # question shape the gateway normalizes away would show up as unattempted.
    for name in ("omj-smoke", "underdetermined", "order"):
        assert report["suites"][name]["n_unattempted"] == 0, name


def test_bench_prints_english_summary_table(tmp_path: Path) -> None:
    # REQ-032
    result = _invoke(["--backend", "mock", "--out", str(tmp_path / "run")])

    assert result.exit_code == 0, _combined_output(result)
    assert "Suite" in result.stdout and "Accuracy" in result.stdout
    assert "omj-smoke" in result.stdout
    assert "Badge" not in result.stdout
    assert "Report" in result.stdout
    assert "| ---" not in result.stdout  # a terminal table, not markdown


def test_bench_json_flag_prints_report_json_only(tmp_path: Path) -> None:
    # REQ-032
    out_dir = tmp_path / "run"
    result = _invoke(["--backend", "mock", "--out", str(out_dir), "--json"])

    assert result.exit_code == 0, _combined_output(result)
    printed = json.loads(result.stdout)
    on_disk = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert printed == on_disk
    assert "Suite" not in result.stdout


def test_bench_fit_temperature_saves_calibration(tmp_path: Path, omj_home: Path) -> None:
    # REQ-032
    out_dir = tmp_path / "run"
    result = _invoke(["--backend", "mock", "--out", str(out_dir), "--fit-temperature"])

    assert result.exit_code == 0, _combined_output(result)
    saved = list((omj_home / "calibration").glob("*.json"))
    assert len(saved) == 1, f"expected one calibration file, got {saved}"

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    calibration = report["calibration"]
    assert calibration is not None
    assert calibration["suite"] == "omj-smoke"
    assert "ece_before" in calibration
    assert "ece_after" in calibration
    assert set(calibration["temperatures"]) == {"noul", "choice", "score"}


def test_bench_submission_bundle_has_one_row_per_decision(tmp_path: Path) -> None:
    # REQ-032
    out_dir = tmp_path / "run"
    result = _invoke(["--backend", "mock", "--suite", "omj-smoke", "--out", str(out_dir), "--submission"])

    assert result.exit_code == 0, _combined_output(result)
    results_path = out_dir / "submission" / "results.jsonl"
    assert results_path.is_file()
    lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 30
    assert json.loads(lines[0])["adapter"] == "omj"


def test_bench_from_log_replays_logged_decisions(tmp_path: Path) -> None:
    # REQ-032
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    row = {
        "ts": "2026-09-22T00:00:00+00:00",
        "request_id": "req-1",
        "backend": "mock",
        "model": "mock-keyword-v1",
        "state_sha256": "0" * 64,
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Route this ticket.",
                "criteria": {"billing": "money questions", "bug": "broken software"},
            }
        },
        "answers": {},
        "latency_ms": 1.0,
        "calibrated": False,
        "upstream_extra": {},
        "state": "The invoice charged me twice.",
    }
    (log_dir / "decisions-20260922.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    out_dir = tmp_path / "run"
    result = _invoke(["--backend", "mock", "--from-log", str(log_dir), "--out", str(out_dir)])

    assert result.exit_code == 0, _combined_output(result)
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert set(report["suites"]) == {"from-log"}
    assert report["suites"]["from-log"]["n"] == 1


def test_bench_endpoint_target_runs_over_mock_transport(tmp_path: Path) -> None:
    # REQ-032
    out_dir = tmp_path / "run"

    def factory(opts: BenchOptions, config) -> ResolvedTarget:
        return ResolvedTarget(
            target=_endpoint_target(),
            backend_name="endpoint",
            model="jev-latest",
            route="direct",
        )

    report = run_bench(
        BenchOptions(
            suites=["omj-smoke"],
            endpoint="http://127.0.0.1:8799",
            out=out_dir,
        ),
        target_factory=factory,
    )

    assert report["suites"]["omj-smoke"]["n"] == 30
    assert report["suites"]["omj-smoke"]["n_unattempted"] == 0
    assert report["suites"]["omj-smoke"]["valid_vector_rate"] == pytest.approx(1.0)
    assert report["environment"]["route"] == "direct"
    assert (out_dir / "report.json").is_file()


def test_endpoint_target_parses_every_question_kind() -> None:
    # REQ-032
    target = _endpoint_target()
    item = DecisionItem(
        id="x-1",
        suite="unit",
        state="hello",
        questions={
            "a": {"type": "noul", "instructions": "?", "criteria": {"true": "t", "false": "f"}},
            "b": {"type": "choice", "instructions": "?", "criteria": {"p": "P", "q": "Q"}},
            "c": {"type": "score", "instructions": "?", "criteria": ["low", "mid", "high"]},
        },
        expected=None,
    )

    answers = target.decide_item(item)

    assert answers["a"][1] == ["yes", "no"]
    assert answers["a"][2] == pytest.approx([0.75, 0.25])
    assert answers["b"][1] == ["p", "q"]
    assert answers["b"][2] == pytest.approx([0.6, 0.4])
    assert answers["c"][1] == ["0", "1", "2"]
    assert sum(answers["c"][2]) == pytest.approx(1.0)
    assert target.input_tokens(item) == 11


def test_endpoint_target_maps_401_to_e_auth() -> None:
    # REQ-032
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"type": "authentication_error", "message": "no"}})

    target = _endpoint_target(unauthorized)
    item = DecisionItem(
        id="x-1",
        suite="unit",
        state="hello",
        questions={"a": {"type": "noul", "instructions": "?", "criteria": None}},
        expected=None,
    )

    with pytest.raises(OmjError) as excinfo:
        target.decide_item(item)
    assert excinfo.value.code is ErrorCode.E_AUTH


def test_endpoint_target_refuses_non_loopback_when_network_blocked() -> None:
    # REQ-032
    with pytest.raises(OmjError) as excinfo:
        EndpointTarget("https://jev.example.com")
    assert excinfo.value.code is ErrorCode.E_NET


def test_bench_rejects_both_backend_and_endpoint(tmp_path: Path) -> None:
    # REQ-032
    result = _invoke(
        ["--backend", "mock", "--endpoint", "http://127.0.0.1:8799", "--out", str(tmp_path / "run")]
    )

    assert result.exit_code == 1
    assert "E_CONFIG" in _combined_output(result)


def test_bench_without_backend_or_config_asks_for_init(tmp_path: Path) -> None:
    # No --backend/--endpoint and no config.toml: nothing to default to.
    result = _invoke(["--out", str(tmp_path / "run")])

    assert result.exit_code == 1
    output = _combined_output(result)
    assert "E_CONFIG" in output and "omj init" in output


def test_bench_defaults_to_the_backend_in_config(tmp_path: Path, omj_home: Path) -> None:
    from omj.config import Config, save_config

    cfg = Config()
    cfg.backend.name = "mock"
    save_config(cfg)
    out_dir = tmp_path / "run"
    result = _invoke(["--suite", "omj-smoke", "--out", str(out_dir)])

    assert result.exit_code == 0, _combined_output(result)
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["environment"]["backend"] == "mock"
    assert report["run"]["command"].startswith("omj bench --backend mock")



def test_bench_refuses_existing_non_empty_out_dir(tmp_path: Path) -> None:
    # REQ-032
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "keep.txt").write_text("previous run", encoding="utf-8")

    result = _invoke(["--backend", "mock", "--out", str(out_dir)])

    assert result.exit_code == 1
    assert "E_CONFIG" in _combined_output(result)
    assert (out_dir / "keep.txt").read_text(encoding="utf-8") == "previous run"


def test_run_suite_stops_at_first_auth_error() -> None:
    # REQ-032
    items = load_suite("omj-smoke")
    target = _AuthFailingTarget(fail_on=2)

    result = run_suite(items, target, suite_name="omj-smoke", backend_name="fake", model="fake-1")

    assert isinstance(result, RunResult)
    assert len(result.rows) == 2
    assert result.rows[0].error is None
    assert result.rows[1].error is not None
    assert result.n_errors == 1
    assert target.calls == 2


def test_run_suite_stops_after_three_consecutive_net_errors() -> None:
    # REQ-032
    items = load_suite("omj-smoke")
    target = _AuthFailingTarget(fail_on=1, code=ErrorCode.E_NET)

    result = run_suite(items, target, suite_name="omj-smoke", backend_name="fake", model="fake-1")

    assert len(result.rows) == 3
    assert all(row.error is not None for row in result.rows)
    assert target.calls == 3


def test_run_suite_scores_rows_against_expected() -> None:
    # REQ-032
    items = load_suite("omj-smoke")[:3]
    target = BackendTarget(_mock_backend())

    result = run_suite(items, target, suite_name="omj-smoke", backend_name="mock", model="mock-keyword-v1")

    assert len(result.rows) == 3
    assert result.n_errors == 0
    assert result.input_tokens_total > 0
    assert result.wall_seconds >= 0.0
    for row in result.rows:
        assert row.suite == "omj-smoke"
        assert row.probs is not None
        assert row.logits is not None
        assert row.correct in (True, False)
        assert row.expected is not None


def _mock_backend():
    from omj.backends.registry import create_backend
    from omj.config import BackendSection

    backend = create_backend("mock")
    backend.load(BackendSection(name="mock"))
    return backend


def _write_calibration(path: Path, temps: dict[str, float]) -> Path:
    path.write_text(json.dumps({
        "version": 1, "backend": "mock", "model": "mock-keyword-v1", "temperatures": temps,
        "fitted_on": {"suite": "dev", "n": 10}, "ece_before": 0.1, "ece_after": 0.05,
        "created_at": "2026-09-24T00:00:00Z",
    }), encoding="utf-8")
    return path


def test_bench_calibration_is_applied_to_backend_logits(tmp_path: Path) -> None:
    cal = _write_calibration(tmp_path / "cal.json", {"noul": 5.0, "choice": 5.0, "score": 5.0})
    raw = run_bench(BenchOptions(suites=["omj-smoke"], backend="mock", out=tmp_path / "raw"))
    hot = run_bench(BenchOptions(suites=["omj-smoke"], backend="mock", calibration=str(cal), out=tmp_path / "cal"))
    r, h = raw["suites"]["omj-smoke"], hot["suites"]["omj-smoke"]
    assert h["accuracy"] == r["accuracy"]  # temperature never changes argmax
    assert h["brier"] != r["brier"]  # but it does change the probabilities


@pytest.mark.parametrize("extra", [
    {"endpoint": "http://127.0.0.1:8799", "backend": None},
    {"fit_temperature": True},
])
def test_bench_calibration_rejects_bad_combinations(tmp_path: Path, extra: dict) -> None:
    cal = _write_calibration(tmp_path / "cal.json", {"noul": 1.0, "choice": 1.0, "score": 1.0})
    opts = {"suites": ["omj-smoke"], "backend": "mock", "calibration": str(cal), "out": tmp_path / "o", **extra}
    with pytest.raises(OmjError) as exc:
        run_bench(BenchOptions(**opts))
    assert exc.value.code is ErrorCode.E_CONFIG


def test_bench_calibration_missing_file_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(OmjError):
        run_bench(BenchOptions(suites=["omj-smoke"], backend="mock", calibration=str(tmp_path / "nope.json"),
                               out=tmp_path / "o"))


def test_run_suite_reports_every_attempted_item_to_on_item() -> None:
    items = load_suite("omj-smoke")[:5]
    seen: list[bool] = []
    run_suite(
        items,
        _AuthFailingTarget(fail_on=4, code=ErrorCode.E_NET),
        suite_name="omj-smoke",
        backend_name="t",
        model="m",
        on_item=seen.append,
    )
    assert seen == [True, True, True, False, False]


def test_bench_progress_renders_one_bar_per_suite_when_enabled() -> None:
    import io

    from rich.console import Console

    from omj.bench.progress import BenchProgress

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=100)
    with BenchProgress(enabled=True, console=console) as progress:
        progress.note("Loading backend mock...")
        advance = progress.suite("omj-smoke", 2)
        assert advance is not None
        advance(True)
        advance(False)
    text = buf.getvalue()
    assert "Loading backend mock" in text
    assert "omj-smoke" in text and "2/2" in text and "errors 1" in text


def test_bench_progress_is_silent_when_disabled() -> None:
    import io

    from rich.console import Console

    from omj.bench.progress import BenchProgress

    buf = io.StringIO()
    with BenchProgress(enabled=False, console=Console(file=buf)) as progress:
        progress.note("hidden")
        assert progress.suite("omj-smoke", 3) is None
    assert buf.getvalue() == ""


def test_bench_default_output_shows_every_section_and_ends_with_the_summary(tmp_path: Path) -> None:
    result = _invoke(["--backend", "mock", "--suite", "omj-smoke", "--out", str(tmp_path / "run")])

    assert result.exit_code == 0, _combined_output(result)
    out = result.stdout
    for section in ("Metrics", "Accuracy", "Calibration", "95% CI", "By question type", "By tag", "Reliability", "Summary", "Report"):
        assert section in out, section
    assert out.index("Reliability") < out.index("Summary") < out.index("Report")


def test_bench_brief_prints_only_the_summary(tmp_path: Path) -> None:
    result = _invoke(["--backend", "mock", "--suite", "omj-smoke", "--brief", "--out", str(tmp_path / "run")])

    assert result.exit_code == 0, _combined_output(result)
    assert "Summary" in result.stdout and "omj-smoke" in result.stdout
    assert "Reliability" not in result.stdout and "By tag" not in result.stdout


def test_bench_model_option_runs_a_named_model_on_semif(tmp_path: Path) -> None:
    seen: list[BenchOptions] = []

    def factory(opts: BenchOptions, config):
        seen.append(opts)
        return ResolvedTarget(target=_AuthFailingTarget(fail_on=10**6), backend_name="semif", model=opts.model or "", route="local")

    report = run_bench(
        BenchOptions(model="Qwen/Qwen3.5-0.8B", suites=["omj-smoke"], out=tmp_path / "run"),
        target_factory=factory,
        out=io.StringIO(),
    )

    assert seen[0].backend == "semif" and seen[0].model == "Qwen/Qwen3.5-0.8B"
    assert report["run"]["command"].startswith("omj bench --model Qwen/Qwen3.5-0.8B")


@pytest.mark.parametrize("extra", [["--endpoint", "http://127.0.0.1:8799"], ["--backend", "mock"]])
def test_bench_model_option_rejects_endpoint_or_other_backend(tmp_path: Path, extra: list[str]) -> None:
    result = _invoke(["--model", "Qwen/Qwen3.5-0.8B", *extra, "--out", str(tmp_path / "run")])
    assert result.exit_code == 1 and "E_CONFIG" in _combined_output(result)


def test_bench_adapter_without_backend_runs_on_semif(tmp_path: Path, omj_home: Path) -> None:
    from omj.config import Config, save_config

    cfg = Config()
    cfg.backend.name = "mock"  # the configured backend must not swallow the adapter
    save_config(cfg)
    adapter = tmp_path / "best"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    seen: list[BenchOptions] = []

    def factory(opts: BenchOptions, config):
        seen.append(opts)
        return ResolvedTarget(target=_AuthFailingTarget(fail_on=10**6), backend_name="semif", model="m", route="local")

    import omj.cli.bench_cmd as bench_cmd

    original = bench_cmd._require_adapter_dir
    bench_cmd._require_adapter_dir = lambda path: str(path)  # no semif extra needed for this check
    try:
        run_bench(BenchOptions(adapter=str(adapter), suites=["omj-smoke"], out=tmp_path / "run"), target_factory=factory, out=io.StringIO())
    finally:
        bench_cmd._require_adapter_dir = original
    assert seen[0].backend == "semif"
