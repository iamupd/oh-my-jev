"""Bench report (report.json/report.md, environment metadata) and reliability.svg. # REQ-040"""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from omj import __version__
from omj.bench.report import Environment, collect_environment, write_report
from omj.bench.svg import bins_from_rows, reliability_svg
from omj.bench.types import Metrics, ResultRow
from omj.config import Config


def make_metrics(**overrides: object) -> Metrics:
    base: dict[str, object] = dict(
        n=10,
        n_valid=10,
        n_unattempted=0,
        accuracy=0.8,
        chance_corrected_accuracy=0.6,
        hard_accuracy=0.6,
        ece=0.03,
        ece_jevbench=0.04,
        brier=0.1,
        nll=0.2,
        coverage_at_risk=0.9,
        valid_vector_rate=1.0,
        valid_vector_rate_lenient=1.0,
        order_shift_max=0.02,
        overconfidence_rate=None,
        mean_max_prob=None,
        uniform_deviation=None,
        ko_en_gap=None,
        pair_agreement=None,
        latency_p50_ms=120.0,
        latency_p95_ms=200.0,
        cost_per_1k=1.5,
    )
    base.update(overrides)
    return Metrics(**base)  # type: ignore[arg-type]


def make_row(
    *,
    item_id: str = "item-1",
    suite: str = "synthetic",
    qid: str = "q1",
    kind: str = "choice",
    keys: list[str],
    probs: list[float] | None,
    expected: str | None = None,
) -> ResultRow:
    return ResultRow(
        item_id=item_id,
        suite=suite,
        qid=qid,
        kind=kind,
        keys=keys,
        probs=probs,
        logits=None,
        expected=expected,
        correct=None,
        latency_ms=10.0,
        cost_usd=None,
        backend="mock",
        model="mock-model",
    )


class TestCollectEnvironment:
    def test_fields_from_config(self) -> None:
        # REQ-040
        cfg = Config()
        cfg.hardware.device = "cuda"
        cfg.hardware.vram_gb = 24.0
        cfg.hardware.ram_gb = 64.0
        cfg.hardware.platform = "windows"
        cfg.hardware.python = "3.11.9"
        cfg.hardware.gpu_name = "RTX 4090"
        cfg.backend.revision = "abc123"
        cfg.backend.quant = "bf16"

        env = collect_environment(cfg, "semif", "Qwen/Qwen3.5-2B", "local")

        assert env.backend == "semif"
        assert env.model == "Qwen/Qwen3.5-2B"
        assert env.route == "local"
        assert env.revision == "abc123"
        assert env.quant == "bf16"
        assert env.hardware["device"] == "cuda"
        assert env.hardware["gpu_name"] == "RTX 4090"
        assert env.omj_version == __version__
        assert env.run_at
        assert env.git_commit

    def test_fields_without_config(self) -> None:
        # REQ-040
        env = collect_environment(None, "mock", "mock-model", "direct")
        assert env.hardware == {}
        assert env.revision == ""
        assert env.quant == ""
        assert env.route == "direct"

    def test_git_commit_unknown_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # REQ-040
        def boom(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", boom)
        env = collect_environment(None, "mock", "mock-model", "openrouter")
        assert env.git_commit == "unknown"


def make_env() -> Environment:
    return Environment(
        hardware={
            "device": "cpu",
            "vram_gb": None,
            "ram_gb": 32.0,
            "platform": "windows",
            "python": "3.11.9",
            "gpu_name": None,
        },
        backend="mock",
        model="mock-model",
        revision="",
        quant="bf16",
        route="local",
        git_commit="abc1234",
        omj_version="0.1.0",
        run_at="2026-09-22T00:00:00+00:00",
    )


class TestWriteReport:
    def test_report_json_keys_and_environment(self, tmp_path: Path) -> None:
        # REQ-040
        suite_metrics = {"synthetic": make_metrics(), "massive-ko": make_metrics(accuracy=0.7)}
        json_path, md_path = write_report(
            tmp_path,
            suite_metrics=suite_metrics,
            env=make_env(),
            calibration={"temperatures": {"noul": 1.0, "choice": 1.0, "score": 1.0}},
        )

        assert json_path == tmp_path / "report.json"
        assert md_path == tmp_path / "report.md"
        assert json_path.exists()
        assert md_path.exists()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert set(data.keys()) == {
            "omj",
            "environment",
            "suites",
            "calibration",
        }
        assert data["omj"]["version"] == __version__
        assert data["suites"]["synthetic"]["accuracy"] == 0.8
        assert data["suites"]["massive-ko"]["accuracy"] == 0.7
        assert data["calibration"]["temperatures"]["noul"] == 1.0

        env_data = data["environment"]
        for key in (
            "hardware",
            "backend",
            "model",
            "revision",
            "quant",
            "route",
            "git_commit",
            "omj_version",
            "run_at",
        ):
            assert key in env_data
        assert env_data["backend"] == "mock"
        assert env_data["route"] == "local"
        assert env_data["hardware"]["device"] == "cpu"

    def test_report_json_calibration_defaults_to_null(self, tmp_path: Path) -> None:
        # REQ-040
        json_path, _ = write_report(
            tmp_path,
            suite_metrics={"synthetic": make_metrics()},
            env=make_env(),
        )
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["calibration"] is None

    def test_report_md_contains_suite_tables(self, tmp_path: Path) -> None:
        # REQ-040
        suite_metrics = {"synthetic": make_metrics(), "massive-en": make_metrics(accuracy=0.9)}
        _, md_path = write_report(
            tmp_path,
            suite_metrics=suite_metrics,
            env=make_env(),
        )
        text = md_path.read_text(encoding="utf-8")

        assert "synthetic" in text
        assert "massive-en" in text
        assert "badge" not in text.lower()
        assert "accuracy" in text
        assert "0.8" in text
        assert "## 1. Run" in text and "## 2. Model" in text and "## 3. Summary" in text

    def test_report_md_never_uses_em_dash(self, tmp_path: Path) -> None:
        # REQ-040
        _, md_path = write_report(
            tmp_path,
            suite_metrics={"synthetic": make_metrics()},
            env=make_env(),
        )
        text = md_path.read_text(encoding="utf-8")
        assert "—" not in text

    def test_report_files_use_lf_line_endings(self, tmp_path: Path) -> None:
        # REQ-040
        json_path, md_path = write_report(
            tmp_path,
            suite_metrics={"synthetic": make_metrics()},
            env=make_env(),
        )
        assert b"\r\n" not in json_path.read_bytes()
        assert b"\r\n" not in md_path.read_bytes()


class TestReliabilitySvg:
    def test_svg_is_well_formed_xml(self) -> None:
        # REQ-040
        bins = [(0.1, 0.05, 3), (0.5, 0.48, 10), (0.9, 0.85, 20)]
        svg_text = reliability_svg(bins)
        root = ET.fromstring(svg_text)
        assert root.tag.endswith("svg")

    def test_svg_has_one_rect_per_bin(self) -> None:
        # REQ-040
        bins = [(0.1, 0.05, 3), (0.5, 0.48, 10), (0.9, 0.85, 20), (0.99, 0.9, 1)]
        svg_text = reliability_svg(bins)
        root = ET.fromstring(svg_text)
        rects = [el for el in root.iter() if el.tag.endswith("rect")]
        assert len(rects) == len(bins)

    def test_svg_dimensions(self) -> None:
        # REQ-040
        svg_text = reliability_svg([(0.5, 0.5, 5)])
        root = ET.fromstring(svg_text)
        assert root.attrib["width"] == "480"
        assert root.attrib["height"] == "360"

    def test_svg_empty_bins_still_well_formed(self) -> None:
        # REQ-040
        svg_text = reliability_svg([])
        root = ET.fromstring(svg_text)
        rects = [el for el in root.iter() if el.tag.endswith("rect")]
        assert rects == []

    def test_svg_contains_count_labels(self) -> None:
        # REQ-040
        svg_text = reliability_svg([(0.5, 0.5, 42)])
        assert "42" in svg_text

    def test_svg_custom_title(self) -> None:
        # REQ-040
        svg_text = reliability_svg([(0.5, 0.5, 1)], title="My Suite")
        assert "My Suite" in svg_text

    def test_bins_from_rows_groups_by_confidence(self) -> None:
        # REQ-040
        rows = [
            make_row(keys=["A", "B"], probs=[0.9, 0.1], expected="A"),
            make_row(keys=["A", "B"], probs=[0.85, 0.15], expected="B"),
            make_row(keys=["A", "B"], probs=[0.4, 0.6], expected="B"),
        ]
        bins = bins_from_rows(rows, n_bins=10)
        total_count = sum(count for _, _, count in bins)
        assert total_count == 3
        for mean_confidence, accuracy, count in bins:
            assert 0.0 <= mean_confidence <= 1.0
            assert 0.0 <= accuracy <= 1.0
            assert count > 0

    def test_bins_from_rows_excludes_unscored_rows(self) -> None:
        # REQ-040
        rows = [
            make_row(keys=["A", "B"], probs=[0.9, 0.1], expected="A"),
            make_row(keys=["A", "B"], probs=[0.9, 0.1], expected=None),
            make_row(keys=["A", "B"], probs=None, expected="A"),
        ]
        bins = bins_from_rows(rows, n_bins=10)
        total_count = sum(count for _, _, count in bins)
        assert total_count == 1

    def test_bins_from_rows_feeds_reliability_svg(self) -> None:
        # REQ-040
        rows = [
            make_row(keys=["A", "B"], probs=[0.9, 0.1], expected="A"),
            make_row(keys=["A", "B"], probs=[0.6, 0.4], expected="A"),
        ]
        bins = bins_from_rows(rows, n_bins=15)
        svg_text = reliability_svg(bins)
        root = ET.fromstring(svg_text)
        rects = [el for el in root.iter() if el.tag.endswith("rect")]
        assert len(rects) == len(bins)
