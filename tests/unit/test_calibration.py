"""Calibration file loading and temperature application. REQ-021"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omj.errors import ErrorCode, OmjError
from omj.gateway.calibration import (
    Calibration,
    apply_temperature,
    load_calibration,
    save_calibration,
    temperature_for,
)


class TestLoadCalibration:
    def test_load_valid_calibration(self, tmp_path: Path) -> None:
        cal_data = {
            "version": 1,
            "backend": "semif",
            "model": "Qwen/Qwen3.5-2B",
            "temperatures": {"noul": 0.8, "choice": 1.2, "score": 0.9},
            "fitted_on": {"suite": "jevbench", "n": 100},
            "ece_before": 0.15,
            "ece_after": 0.05,
            "created_at": "2026-09-22T12:00:00Z",
        }
        cal_file = tmp_path / "test_cal.json"
        cal_file.write_text(json.dumps(cal_data), encoding="utf-8")

        cal = load_calibration(str(cal_file))

        assert cal.version == 1
        assert cal.backend == "semif"
        assert cal.model == "Qwen/Qwen3.5-2B"
        assert cal.temperatures == {"noul": 0.8, "choice": 1.2, "score": 0.9}
        assert cal.fitted_on == {"suite": "jevbench", "n": 100}
        assert cal.ece_before == 0.15
        assert cal.ece_after == 0.05
        assert cal.created_at == "2026-09-22T12:00:00Z"

    def test_missing_temperature_key_defaults_to_one(self, tmp_path: Path) -> None:
        cal_data = {
            "version": 1,
            "backend": "semif",
            "model": "Qwen/Qwen3.5-2B",
            "temperatures": {"noul": 0.8},
            "fitted_on": {"suite": "jevbench", "n": 100},
            "ece_before": None,
            "ece_after": None,
            "created_at": "2026-09-22T12:00:00Z",
        }
        cal_file = tmp_path / "test_cal.json"
        cal_file.write_text(json.dumps(cal_data), encoding="utf-8")

        cal = load_calibration(str(cal_file))

        assert cal.temperatures["noul"] == 0.8
        assert cal.temperatures["choice"] == 1.0
        assert cal.temperatures["score"] == 1.0

    def test_invalid_temperature_raises_config_error(self, tmp_path: Path) -> None:
        cal_data = {
            "version": 1,
            "backend": "semif",
            "model": "Qwen/Qwen3.5-2B",
            "temperatures": {"noul": 0.0, "choice": 1.0, "score": 1.0},
            "fitted_on": {"suite": "jevbench", "n": 100},
            "ece_before": None,
            "ece_after": None,
            "created_at": "2026-09-22T12:00:00Z",
        }
        cal_file = tmp_path / "test_cal.json"
        cal_file.write_text(json.dumps(cal_data), encoding="utf-8")

        with pytest.raises(OmjError) as exc_info:
            load_calibration(str(cal_file))

        assert exc_info.value.code == ErrorCode.E_CONFIG
        assert "calibration:" in exc_info.value.message

    def test_malformed_json_raises_config_error(self, tmp_path: Path) -> None:
        cal_file = tmp_path / "test_cal.json"
        cal_file.write_text("{invalid json", encoding="utf-8")

        with pytest.raises(OmjError) as exc_info:
            load_calibration(str(cal_file))

        assert exc_info.value.code == ErrorCode.E_CONFIG
        assert "calibration:" in exc_info.value.message

    def test_nonexistent_file_raises_config_error(self, tmp_path: Path) -> None:
        cal_file = tmp_path / "nonexistent.json"

        with pytest.raises(OmjError) as exc_info:
            load_calibration(str(cal_file))

        assert exc_info.value.code == ErrorCode.E_CONFIG
        assert "calibration:" in exc_info.value.message

    def test_non_dict_temperatures_raises_config_error_with_prefix(self, tmp_path: Path) -> None:
        # REQ-021: a non-dict `temperatures` value must still surface the
        # "calibration:" prefix like every other E_CONFIG message here.
        cal_data = {
            "version": 1,
            "backend": "semif",
            "model": "Qwen/Qwen3.5-2B",
            "temperatures": ["not", "a", "dict"],
            "fitted_on": {"suite": "jevbench", "n": 100},
            "ece_before": None,
            "ece_after": None,
            "created_at": "2026-09-22T12:00:00Z",
        }
        cal_file = tmp_path / "test_cal.json"
        cal_file.write_text(json.dumps(cal_data), encoding="utf-8")

        with pytest.raises(OmjError) as exc_info:
            load_calibration(str(cal_file))

        assert exc_info.value.code == ErrorCode.E_CONFIG
        assert exc_info.value.message.startswith("calibration:")


class TestTemperatureFor:
    def test_temperature_for_with_calibration(self) -> None:
        cal = Calibration(
            version=1,
            backend="semif",
            model="Qwen/Qwen3.5-2B",
            temperatures={"noul": 0.8, "choice": 1.2, "score": 0.9},
            fitted_on={"suite": "jevbench", "n": 100},
            ece_before=0.15,
            ece_after=0.05,
            created_at="2026-09-22T12:00:00Z",
        )

        assert temperature_for(cal, "noul") == 0.8
        assert temperature_for(cal, "choice") == 1.2
        assert temperature_for(cal, "score") == 0.9

    def test_temperature_for_with_none_calibration(self) -> None:
        assert temperature_for(None, "noul") == 1.0
        assert temperature_for(None, "choice") == 1.0
        assert temperature_for(None, "score") == 1.0


class TestApplyTemperature:
    def test_apply_temperature_divides_logits(self) -> None:
        logits = [2.0, 4.0, 1.0]
        result = apply_temperature(logits, 2.0)

        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(0.5)

    def test_apply_temperature_preserves_argmax(self) -> None:
        import random

        random.seed(42)

        for _ in range(100):
            logits = [random.uniform(-10, 10) for _ in range(random.randint(2, 10))]
            original_argmax = logits.index(max(logits))

            for temp in [0.3, 1.0, 2.5]:
                result = apply_temperature(logits, temp)
                result_argmax = result.index(max(result))
                assert (
                    original_argmax == result_argmax
                ), f"argmax changed for logits={logits}, temp={temp}"

    def test_apply_temperature_returns_list(self) -> None:
        logits = [1.0, 2.0, 3.0]
        result = apply_temperature(logits, 1.5)

        assert isinstance(result, list)
        assert len(result) == 3


class TestSaveAndLoad:
    def test_round_trip_save_and_load(self, tmp_path: Path) -> None:
        cal = Calibration(
            version=1,
            backend="semif",
            model="Qwen/Qwen3.5-2B",
            temperatures={"noul": 0.8, "choice": 1.2, "score": 0.9},
            fitted_on={"suite": "jevbench", "n": 100},
            ece_before=0.15,
            ece_after=0.05,
            created_at="2026-09-22T12:00:00Z",
        )
        cal_file = tmp_path / "test_cal.json"

        save_calibration(cal, str(cal_file))
        loaded_cal = load_calibration(str(cal_file))

        assert loaded_cal == cal

    def test_save_uses_indent_2(self, tmp_path: Path) -> None:
        cal = Calibration(
            version=1,
            backend="semif",
            model="Qwen/Qwen3.5-2B",
            temperatures={"noul": 0.8, "choice": 1.2, "score": 0.9},
            fitted_on={"suite": "jevbench", "n": 100},
            ece_before=0.15,
            ece_after=0.05,
            created_at="2026-09-22T12:00:00Z",
        )
        cal_file = tmp_path / "test_cal.json"

        save_calibration(cal, str(cal_file))
        content = cal_file.read_text(encoding="utf-8")

        parsed = json.loads(content)
        assert parsed["version"] == 1
        assert "  " in content

    def test_save_writes_lf_line_endings_not_crlf(self, tmp_path: Path) -> None:
        # REQ-021: write_text() without newline="\n" translates "\n" to the
        # platform default, which is CRLF on Windows.
        cal = Calibration(
            version=1,
            backend="semif",
            model="Qwen/Qwen3.5-2B",
            temperatures={"noul": 0.8, "choice": 1.2, "score": 0.9},
            fitted_on={"suite": "jevbench", "n": 100},
            ece_before=0.15,
            ece_after=0.05,
            created_at="2026-09-22T12:00:00Z",
        )
        cal_file = tmp_path / "test_cal.json"

        save_calibration(cal, str(cal_file))
        raw_bytes = cal_file.read_bytes()

        assert b"\r" not in raw_bytes
