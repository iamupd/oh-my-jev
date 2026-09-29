"""An adapter trained on one base model must not be loaded onto another."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from omj.backends.semif import adapter_base_model, check_adapter_base  # noqa: E402
from omj.errors import ErrorCode, OmjError  # noqa: E402


def _adapter(tmp_path: Path, base: str) -> Path:
    d = tmp_path / "best"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}), encoding="utf-8")
    return d


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
        ("/home/u/.omj/models/Qwen__Qwen3.5-4B/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "Qwen/Qwen3.5-4B"),
        (r"C:\Users\u\.omj\models\Qwen__Qwen3.5-2B\main", "Qwen/Qwen3.5-2B"),
        ("/data/my-local-model", None),
        ("", None),
    ],
)
def test_adapter_base_model_reads_hub_ids_and_omj_cache_paths(tmp_path: Path, base: str, expected) -> None:
    assert adapter_base_model(_adapter(tmp_path, base)) == expected


def test_check_adapter_base_rejects_a_different_base_with_a_config_hint(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, "/x/.omj/models/Qwen__Qwen3.5-4B/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    with pytest.raises(OmjError) as err:
        check_adapter_base(adapter, "Qwen/Qwen3.5-2B")
    assert err.value.code == ErrorCode.E_CONFIG
    assert "Qwen/Qwen3.5-4B" in err.value.hint and "OMJ_CONFIG" in err.value.hint


def test_check_adapter_base_accepts_the_matching_or_an_unknown_base(tmp_path: Path) -> None:
    check_adapter_base(_adapter(tmp_path, "Qwen/Qwen3.5-4B"), "qwen/qwen3.5-4b")
    (tmp_path / "other").mkdir()
    check_adapter_base(_adapter(tmp_path / "other", "/data/custom"), "Qwen/Qwen3.5-4B")
