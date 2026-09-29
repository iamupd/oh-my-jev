"""GPU integration test: the real example-intent-en recipe for a few optimizer steps.

# REQ-015
# REQ-016
# REQ-018
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover - GPU only
    pytest.skip("CUDA GPU required", allow_module_level=True)


@pytest.fixture()
def real_omj_home(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use the developer's real OMJ_HOME so the cached base model and MASSIVE tarball are reused."""
    home = Path(os.environ.get("OMJ_REAL_HOME", str(Path.home() / ".omj")))
    monkeypatch.setenv("OMJ_HOME", str(home))
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    return home


def test_massive_ko_recipe_runs_20_steps_within_budget(real_omj_home: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from omj.cli.main import app

    out_dir = tmp_path / "run"
    result = CliRunner().invoke(
        app,
        ["train", "--recipe", "example-intent-en", "--max-steps", "20", "--out", str(out_dir), "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    train_json = json.loads(result.stdout)
    assert train_json["steps"] == 20
    assert (out_dir / "best" / "adapter_config.json").is_file()
    assert (out_dir / "last" / "trainer_state.pt").is_file()
    assert (out_dir / "calibration.json").is_file()
    assert train_json["peak_vram_gb"] is not None and train_json["peak_vram_gb"] <= 7.5
    assert train_json["data"]["test_overlap"] == 0
    assert train_json["data"]["n_examples"] > 20_000
    # A 20-step probe carries fixed costs (model load, kernel compile, dev evals); allow generous overhead.
    assert train_json["wall_seconds"] < 20 * 9.0 + 420.0


def test_trained_adapter_loads_into_semif_and_changes_model_id(real_omj_home: Path, tmp_path: Path) -> None:
    from omj.backends.semif import SemifBackend
    from omj.config import BackendSection
    from omj.train.recipe import load_recipe

    recipe = load_recipe("example-intent-en")
    adapter = None
    for candidate in sorted((real_omj_home / "adapters" / "example-intent-en").glob("*/best")):
        if (candidate / "adapter_config.json").is_file():
            adapter = candidate
    if adapter is None:
        pytest.skip("no trained example-intent-en adapter under OMJ_HOME/adapters")
    backend = SemifBackend()
    backend.load(
        BackendSection(
            name="semif",
            model=recipe.base_model,
            revision=recipe.revision,
            quant="bf16",
            adapter=str(adapter),
        )
    )
    assert backend.model_id.endswith("+best")
    answers = backend.decide(
        "이번 주 오전 다섯 시에 깨워줘",
        {"q": {"type": "choice", "instructions": "이 발화가 속한 분야는?", "criteria": {"alarm": "알람", "weather": "날씨", "music": "음악"}}},
    )
    assert answers["q"].logits is not None and len(answers["q"].logits) == 3
