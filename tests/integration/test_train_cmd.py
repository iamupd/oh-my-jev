"""`omj train`: option wiring, --dry-run slicing, --json contract, and error paths.

# REQ-009
# REQ-019
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from omj.cli import train_cmd
from omj.cli.main import app
from omj.errors import ErrorCode, OmjError
from omj.train.recipe import DataSection, LoraSection, Recipe, TrainSection

runner = CliRunner()


def _combined_output(result) -> str:
    return result.stdout + result.stderr


def _fake_recipe(**overrides: Any) -> Recipe:
    data = dict(
        name="fake-recipe",
        base_model="fake/base",
        revision="deadbeef",
        data=DataSection(locale="ko-KR", dev_sample=500, criteria_file=""),
        lora=LoraSection(r=4, alpha=8, dropout=0.05, target_modules=["q_proj"]),
        train=TrainSection(
            lr=1e-4,
            epochs=1,
            batch_size=2,
            grad_accum=1,
            max_seq_len=64,
            label_smoothing=0.0,
            seed=7,
            eval_every_steps=1,
        ),
    )
    data.update(overrides)
    return Recipe(**data)


@dataclass
class _FakeTrainResult:
    out_dir: Path
    best_step: int
    best_dev_accuracy: float
    steps: int
    peak_vram_gb: float
    wall_seconds: float
    train_json: Path
    calibration_json: Path | None
    oom_retry: bool


class _FakeTrainer:
    """Records the kwargs it is constructed with and writes a fake train.json."""

    calls: list[dict[str, Any]] = []

    def __init__(self, recipe, model, tokenizer, train_examples, dev_examples, out_dir, **kwargs) -> None:
        self.out_dir = Path(out_dir)
        self.train_examples = train_examples
        self.dev_examples = dev_examples
        self.kwargs = kwargs
        _FakeTrainer.calls.append(
            {
                "recipe": recipe,
                "train_examples": train_examples,
                "dev_examples": dev_examples,
                "out_dir": self.out_dir,
                **kwargs,
            }
        )

    def run(self) -> _FakeTrainResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        train_json_path = self.out_dir / "train.json"
        payload = {
            "run_id": "fake-run",
            "steps": self.kwargs.get("max_steps") or 10,
            "best_step": 1,
            "best_dev_accuracy": 0.9,
            "peak_vram_gb": 1.23,
            "wall_seconds": 4.5,
        }
        train_json_path.write_text(json.dumps(payload), encoding="utf-8")
        return _FakeTrainResult(
            out_dir=self.out_dir,
            best_step=payload["best_step"],
            best_dev_accuracy=payload["best_dev_accuracy"],
            steps=payload["steps"],
            peak_vram_gb=payload["peak_vram_gb"],
            wall_seconds=payload["wall_seconds"],
            train_json=train_json_path,
            calibration_json=None,
            oom_retry=False,
        )


def _make_deps(
    *,
    recipe: Recipe | None = None,
    load_recipe_raises: OmjError | None = None,
    require_peft_raises: OmjError | None = None,
    cuda: bool = True,
    n_train: int = 20,
    n_dev: int = 12,
) -> train_cmd.TrainDeps:
    recipe_obj = recipe if recipe is not None else _fake_recipe()

    def fake_load_recipe(name_or_path: str) -> Recipe:
        if load_recipe_raises is not None:
            raise load_recipe_raises
        return recipe_obj

    def fake_require_peft() -> Any:
        if require_peft_raises is not None:
            raise require_peft_raises
        return object()

    def fake_load_rows(locale: str, split: str, **kwargs: Any) -> list[dict]:
        n = n_train if split == "train" else n_dev
        return [{"id": i, "locale": locale, "split": split} for i in range(n)]

    def fake_load_criteria(locale: str, criteria_file: str = "") -> dict:
        return {}

    def fake_sample_dev(rows: list[dict], n: int, seed: int) -> list[dict]:
        return rows[: min(n, len(rows))]

    def fake_build_examples(rows, recipe, tokenizer, criteria, *, split):
        examples = [{"id": row["id"], "split": split} for row in rows]
        stats = {"n_rows": len(rows), "n_examples": len(examples), "split": split}
        return examples, stats

    def fake_download_model(model_id: str, revision: str) -> Path:
        return Path("/fake/model")

    def fake_load_tokenizer(model_path: Path) -> Any:
        return object()

    def fake_load_model(model_path: Path, device: str) -> Any:
        return object()

    def fake_device_probe() -> bool:
        return cuda

    return train_cmd.TrainDeps(
        load_recipe=fake_load_recipe,
        require_peft=fake_require_peft,
        load_rows=fake_load_rows,
        load_criteria=fake_load_criteria,
        sample_dev=fake_sample_dev,
        build_examples=fake_build_examples,
        download_model=fake_download_model,
        load_tokenizer=fake_load_tokenizer,
        load_model=fake_load_model,
        trainer_factory=_FakeTrainer,
        device_probe=fake_device_probe,
    )


@pytest.fixture(autouse=True)
def _reset_fake_trainer_calls() -> None:
    _FakeTrainer.calls = []


def _patch_deps(monkeypatch: pytest.MonkeyPatch, deps: train_cmd.TrainDeps) -> None:
    monkeypatch.setattr(train_cmd, "_default_deps", lambda: deps)


# --- option parsing -----------------------------------------------------------


def test_options_pass_through_to_trainer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # REQ-009
    _patch_deps(monkeypatch, _make_deps())
    out_dir = tmp_path / "out"
    resume_dir = tmp_path / "resume" / "last"

    result = runner.invoke(
        app,
        [
            "train",
            "--recipe",
            "fake-recipe",
            "--out",
            str(out_dir),
            "--epochs",
            "3",
            "--max-steps",
            "7",
            "--batch-size",
            "2",
            "--resume",
            str(resume_dir),
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    assert len(_FakeTrainer.calls) == 1
    call = _FakeTrainer.calls[0]
    assert call["out_dir"] == out_dir
    assert call["epochs"] == 3
    assert call["max_steps"] == 7
    assert call["batch_size"] == 2
    assert call["resume"] == str(resume_dir)


def test_default_out_dir_uses_omj_home_recipe_and_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REQ-009
    monkeypatch.setenv("OMJ_HOME", str(tmp_path / "omj-home"))
    _patch_deps(monkeypatch, _make_deps(recipe=_fake_recipe(name="massive-ko")))

    result = runner.invoke(app, ["train", "--recipe", "massive-ko"])

    assert result.exit_code == 0, _combined_output(result)
    call = _FakeTrainer.calls[0]
    out_dir = call["out_dir"]
    assert out_dir.parent.parent == tmp_path / "omj-home" / "adapters"
    assert out_dir.parent.name == "massive-ko"


# --- --dry-run ------------------------------------------------------------


def test_dry_run_slices_examples_and_forces_max_steps_two_on_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REQ-009
    _patch_deps(monkeypatch, _make_deps(cuda=False, n_train=100, n_dev=50))

    result = runner.invoke(
        app,
        ["train", "--recipe", "fake-recipe", "--out", str(tmp_path / "out"), "--dry-run"],
    )

    assert result.exit_code == 0, _combined_output(result)
    call = _FakeTrainer.calls[0]
    assert len(call["train_examples"]) == train_cmd.DRY_RUN_TRAIN_EXAMPLES
    assert len(call["dev_examples"]) == train_cmd.DRY_RUN_DEV_EXAMPLES
    assert call["max_steps"] == train_cmd.DRY_RUN_MAX_STEPS
    assert call["device"] == "cpu"


# --- --json ------------------------------------------------------------------


def test_json_stdout_is_exactly_the_train_json_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REQ-009
    _patch_deps(monkeypatch, _make_deps())
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["train", "--recipe", "fake-recipe", "--out", str(out_dir), "--json"],
    )

    assert result.exit_code == 0, _combined_output(result)
    written = json.loads((out_dir / "train.json").read_text(encoding="utf-8"))
    assert json.loads(result.stdout) == written
    # Progress/summary lines must not leak into stdout when --json is set.
    assert "Recipe:" not in result.stdout
    assert "Recipe:" in result.stderr


# --- errors --------------------------------------------------------------


def test_missing_recipe_exits_one_with_e_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # REQ-009
    _patch_deps(
        monkeypatch,
        _make_deps(load_recipe_raises=OmjError(ErrorCode.E_CONFIG, "recipe not found: no-such-recipe")),
    )

    result = runner.invoke(app, ["train", "--recipe", "no-such-recipe", "--out", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "E_CONFIG" in _combined_output(result)


def test_peft_missing_exits_one_with_install_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # REQ-019
    _patch_deps(
        monkeypatch,
        _make_deps(
            require_peft_raises=OmjError(
                ErrorCode.E_BACKEND, "omj train needs the 'peft' package; run 'uv sync --extra semif'"
            )
        ),
    )

    result = runner.invoke(app, ["train", "--recipe", "fake-recipe", "--out", str(tmp_path / "out")])

    assert result.exit_code == 1
    output = _combined_output(result)
    assert "E_BACKEND" in output
    assert "uv sync --extra semif" in output


def test_no_cuda_without_dry_run_exits_one_with_e_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REQ-009
    _patch_deps(monkeypatch, _make_deps(cuda=False))

    result = runner.invoke(app, ["train", "--recipe", "fake-recipe", "--out", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "E_BACKEND" in _combined_output(result)
    assert len(_FakeTrainer.calls) == 0


# --- mixed recipes (recipe #2) --------------------------------------------


def test_mix_recipe_uses_record_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from omj.train.recipe import MixSource

    recipe = _fake_recipe(data=DataSection(source="mix", mix=[MixSource(name="nsmc", train_n=3, dev_n=2)]))
    deps = _make_deps(recipe=recipe)
    seen: list[str] = []

    def fake_load_mix(r: Recipe) -> tuple[list[dict], list[dict]]:
        return [{"id": i} for i in range(3)], [{"id": i} for i in range(2)]

    def fake_build(records, r, tokenizer, *, split):
        seen.append(split)
        return list(records), {"n_examples": len(records)}

    deps.load_mix = fake_load_mix
    deps.build_examples_from_records = fake_build
    deps.load_rows = None  # the MASSIVE-only path must not run
    _patch_deps(monkeypatch, deps)

    result = runner.invoke(app, ["train", "--recipe", "fake", "--out", str(tmp_path / "o")])

    assert result.exit_code == 0, _combined_output(result)
    assert seen == ["train", "dev"]
    assert len(_FakeTrainer.calls[0]["train_examples"]) == 3
    assert len(_FakeTrainer.calls[0]["dev_examples"]) == 2


# --- main.py registration ------------------------------------------------


def test_help_lists_train_and_compare() -> None:
    # REQ-009
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for name in ("init", "serve", "bench", "train", "compare"):
        assert name in result.stdout


def test_nf4_recipe_passes_quant_to_model_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recipe = _fake_recipe()
    recipe.train.quant = "nf4"
    deps = _make_deps(recipe=recipe)
    seen: dict = {}

    def loader(model_path, device, **kwargs):
        seen.update(kwargs)
        return object()

    deps.load_model = loader
    _patch_deps(monkeypatch, deps)
    result = runner.invoke(app, ["train", "--recipe", "fake", "--out", str(tmp_path / "o")])
    assert result.exit_code == 0, _combined_output(result)
    assert seen == {"quant": "nf4"}


def test_bf16_recipe_keeps_two_argument_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    deps = _make_deps()  # its fake load_model accepts exactly (model_path, device)
    _patch_deps(monkeypatch, deps)
    result = runner.invoke(app, ["train", "--recipe", "fake", "--out", str(tmp_path / "o")])
    assert result.exit_code == 0, _combined_output(result)
