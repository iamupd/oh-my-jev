"""Choosing bf16 or nf4 for a model given by name, and the semif section built for it."""

from __future__ import annotations

from pathlib import Path

import pytest

from omj.models.download import cached_revision
from omj.models.sizing import auto_quant, is_hf_id, named_model_backend, param_billions


@pytest.mark.parametrize(
    ("model", "billions"),
    [
        ("Qwen/Qwen3.5-4B", 4.0),
        ("Qwen/Qwen3.5-0.8B", 0.8),
        ("meta-llama/Llama-3.2-3B-Instruct", 3.0),
        ("google/gemma-3-1b-it", 1.0),
        ("example-org/Decision-4B-Adapter", 4.0),
        ("mistralai/Mixtral-8x7B-v0.1", None),
        ("org/no-size-tag", None),
    ],
)
def test_param_billions_reads_the_size_tag_not_the_version(model: str, billions: float | None) -> None:
    assert param_billions(model) == billions


def test_auto_quant_picks_nf4_only_when_bf16_would_not_fit() -> None:
    assert auto_quant("Qwen/Qwen3.5-4B", 8.0) == "nf4"
    assert auto_quant("Qwen/Qwen3.5-4B", 24.0) == "bf16"
    assert auto_quant("Qwen/Qwen3.5-2B", 8.0) == "bf16"
    assert auto_quant("Qwen/Qwen3.5-4B", None) == "bf16"
    assert auto_quant("org/no-size-tag", 8.0) == "bf16"


def test_is_hf_id_rejects_paths_and_backend_specs() -> None:
    assert is_hf_id("Qwen/Qwen3.5-0.8B")
    for text in ("semif", "semif:C:/adapters/best", "@~/.omj/x.toml", "/abs/path", "a/b/c"):
        assert not is_hf_id(text)


def test_named_model_backend_reuses_a_cached_revision_and_drops_the_old_adapter(omj_home: Path) -> None:
    rev = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    (omj_home / "models" / "Qwen__Qwen3.5-4B" / rev).mkdir(parents=True)
    (omj_home / "models" / "Qwen__Qwen3.5-4B" / rev / ".omj-complete").write_text("")
    base = {"name": "mock", "model": "", "revision": "x", "quant": "bf16", "adapter": "/a", "calibration": "/c"}

    out = named_model_backend(base, "Qwen/Qwen3.5-4B", 8.0)

    assert out["name"] == "semif" and out["model"] == "Qwen/Qwen3.5-4B"
    assert (out["revision"], out["quant"], out["adapter"], out["calibration"]) == (rev, "nf4", "", "")
    assert cached_revision("Qwen/Qwen3.5-0.8B") is None
    assert named_model_backend(base, "Qwen/Qwen3.5-0.8B", 8.0)["revision"] == ""


def test_resolve_named_model_loads_an_adapter_repo_on_its_base(omj_home: Path) -> None:
    import json

    from omj.models.sizing import resolve_named_model

    repo = omj_home / "models" / "example-org__Decision-4B-Adapter" / "main"
    repo.mkdir(parents=True)
    (repo / ".omj-complete").write_text("")
    (repo / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B", "revision": "851bf6e8"}), encoding="utf-8"
    )
    base = {"name": "mock", "model": "", "revision": "", "quant": "bf16", "adapter": "", "calibration": ""}

    out, note = resolve_named_model(base, "example-org/Decision-4B-Adapter", 8.0)

    assert (out["name"], out["model"], out["revision"], out["quant"]) == ("semif", "Qwen/Qwen3.5-4B", "851bf6e8", "nf4")
    assert out["adapter"] == str(repo) and "LoRA adapter" in note


def test_resolve_named_model_offline_treats_an_unknown_id_as_a_full_model(omj_home: Path) -> None:
    from omj.models.sizing import resolve_named_model

    out, note = resolve_named_model({"name": "mock"}, "Qwen/Qwen3.5-0.8B", 8.0)  # OMJ_NO_NETWORK=1 in tests
    assert (out["model"], out["adapter"], note) == ("Qwen/Qwen3.5-0.8B", "", None)
