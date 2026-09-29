"""Shared fixtures: isolate $OMJ_HOME per test and block network access by default."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def omj_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "omj"
    monkeypatch.setenv("OMJ_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # Later tasks' downloaders/HTTP clients check this and refuse real network calls.
    monkeypatch.setenv("OMJ_NO_NETWORK", "1")
    # A real key in the developer's shell must not make benches measure Jev during tests.
    monkeypatch.delenv("JEV_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
