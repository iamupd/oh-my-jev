"""peft dependency guard and packaging checks for omj train.

# REQ-019
"""

from __future__ import annotations

import builtins
import sys
import tomllib
from pathlib import Path

import pytest

from omj.errors import ErrorCode, OmjError
from omj.train import require_peft

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_require_peft_raises_e_backend_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("no peft")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "peft", raising=False)
    with pytest.raises(OmjError) as info:
        require_peft()
    assert info.value.code is ErrorCode.E_BACKEND
    assert "uv sync --extra semif" in info.value.message


def test_semif_extra_declares_peft() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    semif = data["project"]["optional-dependencies"]["semif"]
    assert any(dep.startswith("peft") for dep in semif), semif
    force_include = data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include.get("recipes") == "recipes"
