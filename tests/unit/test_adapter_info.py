"""Following an explicitly given adapter to the base model it was trained on (audit A4)."""

from __future__ import annotations

import json
from pathlib import Path

from omj.backends.adapter_info import adapter_base, follow_adapter_base

REV = "2fc06364715b967f1860aea9cf38778875588b17"


def _adapter(tmp_path: Path, base: str, **extra: str) -> Path:
    d = tmp_path / "best"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base, **extra}), encoding="utf-8")
    return d


def test_adapter_base_reads_the_revision_from_an_omj_cache_path(tmp_path: Path) -> None:
    assert adapter_base(_adapter(tmp_path, f"/home/u/.omj/models/Qwen__Qwen3.5-0.8B/{REV}")) == ("Qwen/Qwen3.5-0.8B", REV)


def test_adapter_base_reads_a_hub_id_and_its_revision_field(tmp_path: Path) -> None:
    assert adapter_base(_adapter(tmp_path, "Qwen/Qwen3.5-4B", revision="abc")) == ("Qwen/Qwen3.5-4B", "abc")


def test_follow_adapter_base_switches_a_semif_section_to_the_adapter_base(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, "C:\\Users\\u\\.omj\\models\\Qwen__Qwen3.5-0.8B\\" + REV)
    backend = {"name": "semif", "model": "Qwen/Qwen3.5-2B", "revision": "15852e8c", "quant": "bf16"}

    note = follow_adapter_base(backend, adapter)

    assert (backend["model"], backend["revision"], backend["quant"]) == ("Qwen/Qwen3.5-0.8B", REV, "bf16")
    assert note is not None and "Qwen/Qwen3.5-0.8B" in note and "Qwen/Qwen3.5-2B" in note


def test_follow_adapter_base_leaves_matching_unknown_or_non_semif_sections_alone(tmp_path: Path) -> None:
    same = {"name": "semif", "model": "qwen/qwen3.5-4b", "revision": "r"}
    assert follow_adapter_base(same, _adapter(tmp_path, "Qwen/Qwen3.5-4B")) is None and same["revision"] == "r"
    (tmp_path / "x").mkdir()
    unknown = {"name": "semif", "model": "Qwen/Qwen3.5-4B"}
    assert follow_adapter_base(unknown, _adapter(tmp_path / "x", "/data/custom")) is None
    (tmp_path / "y").mkdir()
    mock = {"name": "mock", "model": ""}
    assert follow_adapter_base(mock, _adapter(tmp_path / "y", "Qwen/Qwen3.5-4B")) is None and mock["model"] == ""


def test_adapter_paths_with_a_tilde_are_expanded(tmp_path: Path, monkeypatch) -> None:
    from omj.ui.targets import parse_target

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    _adapter(tmp_path, "Qwen/Qwen3.5-4B")
    assert adapter_base("~/best") == ("Qwen/Qwen3.5-4B", "")
    assert parse_target("tuned=semif:~/best").adapter == str(tmp_path / "best")


def test_follow_adapter_base_picks_nf4_when_the_base_does_not_fit(tmp_path: Path) -> None:
    backend = {"name": "semif", "model": "Qwen/Qwen3.5-2B", "revision": "x", "quant": "bf16"}
    note = follow_adapter_base(backend, _adapter(tmp_path, "Qwen/Qwen3.5-4B"), vram_gb=8.0)
    assert backend["quant"] == "nf4" and "nf4" in note
