"""Recipe schema and loader tests.

# REQ-001
# REQ-002
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omj.errors import ErrorCode, OmjError
from omj.train.recipe import RECIPES_DIR, Recipe, load_recipe, recipe_path

VALID = """
name = "t"
base_model = "yujiepan/qwen2.5-tiny-random"
[data]
questions = ["scenario"]
[lora]
r = 2
alpha = 4
dropout = 0.0
target_modules = ["q_proj"]
[train]
lr = 1e-3
epochs = 1
batch_size = 2
grad_accum = 1
max_seq_len = 128
label_smoothing = 0.0
seed = 1
eval_every_steps = 2
"""


def _write(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "r.toml"
    target.write_text(body, encoding="utf-8")
    return target


def test_valid_recipe_loads_with_defaults(tmp_path: Path) -> None:
    recipe = load_recipe(_write(tmp_path, VALID))
    assert isinstance(recipe, Recipe)
    assert recipe.data.shuffle_options is True
    assert recipe.data.dev_sample == 500
    assert recipe.train.warmup_ratio == pytest.approx(0.03)
    assert recipe.effective_batch_size == 2


@pytest.mark.parametrize(
    ("replacement", "field"),
    [
        ("r = 2", "r = 0"),
        ("label_smoothing = 0.0", "label_smoothing = 1.0"),
        ("max_seq_len = 128", "max_seq_len = 8"),
        ("questions = [\"scenario\"]", "questions = [\"bogus\"]"),
        ("lr = 1e-3", "lr = 0"),
        ("target_modules = [\"q_proj\"]", "target_modules = []"),
    ],
)
def test_invalid_field_raises_e_config_with_path(tmp_path: Path, replacement: str, field: str) -> None:
    body = VALID.replace(replacement, field)
    with pytest.raises(OmjError) as info:
        load_recipe(_write(tmp_path, body))
    assert info.value.code is ErrorCode.E_CONFIG
    assert info.value.message.startswith("recipe ")


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OmjError) as info:
        load_recipe(_write(tmp_path, VALID + "\n[extra]\nx = 1\n"))
    assert info.value.code is ErrorCode.E_CONFIG
    assert "extra" in info.value.message


def test_missing_recipe_name_raises_e_config() -> None:
    with pytest.raises(OmjError) as info:
        load_recipe("no-such-recipe-xyz")
    assert info.value.code is ErrorCode.E_CONFIG
    assert "not found" in info.value.message


def test_recipe_name_resolves_to_recipes_dir() -> None:
    assert recipe_path("example-intent-en") == RECIPES_DIR / "example-intent-en.toml"


def test_example_recipe_values() -> None:
    recipe = load_recipe("example-intent-en")
    assert recipe.name == "example-intent-en"
    assert recipe.base_model == "Qwen/Qwen3.5-0.8B"
    assert recipe.revision == "2fc06364715b967f1860aea9cf38778875588b17"
    assert recipe.data.source == "mix" and recipe.data.locale == "en-US"
    assert [m.name for m in recipe.data.mix] == ["massive", "banking77"]
    assert recipe.train.brier_weight == pytest.approx(0.5)
    assert (recipe.lora.r, recipe.lora.alpha) == (8, 16)
    assert recipe.lora.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]
    assert recipe.train.lr == pytest.approx(2e-4)
    assert recipe.train.epochs == 1
    assert (recipe.train.batch_size, recipe.train.grad_accum) == (4, 4)
    assert recipe.train.max_seq_len == 1024
    assert recipe.train.label_smoothing == pytest.approx(0.05)
    assert recipe.train.seed == 20260924
    assert recipe.train.eval_every_steps == 100
