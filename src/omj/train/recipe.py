"""Recipe schema and loader for ``omj train`` (recipes/<name>.toml)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from omj.errors import ErrorCode, OmjError

QuestionName = Literal["scenario", "intent"]


def _resolve_recipes_dir() -> Path:
    """Find the ``recipes/`` directory in a source checkout or an installed wheel."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "recipes",  # checkout: <root>/src/omj/train/recipe.py
        here.parents[2] / "recipes",  # wheel: site-packages/omj/train/recipe.py -> site-packages/recipes
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


RECIPES_DIR = _resolve_recipes_dir()


MixSourceName = Literal["massive", "massive-en", "banking77", "klue-ynat", "klue-nli", "nsmc", "openjev-business", "jsonl"]


class MixSource(BaseModel):
    """One source of a mixed recipe (data.source = "mix"); see omj.train.sources."""

    model_config = ConfigDict(extra="forbid")

    name: MixSourceName
    # Records to sample; for a jsonl source it may be left out to use every training record.
    train_n: int | None = Field(default=None, ge=1)
    dev_n: int = Field(default=50, ge=0)
    # jsonl only: your own decisions in the suite format (resolved against the recipe's folder).
    path: str = ""

    @model_validator(mode="after")
    def _check_path(self) -> "MixSource":
        if self.name == "jsonl" and not self.path:
            raise ValueError("a jsonl source needs path = \"your-decisions.jsonl\"")
        if self.name != "jsonl" and self.path:
            raise ValueError(f"path only applies to name = \"jsonl\", not {self.name!r}")
        if self.name != "jsonl" and self.train_n is None:
            raise ValueError(f"{self.name}: train_n is required")
        return self


class DataSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["massive", "mix"] = "massive"
    mix: list[MixSource] = Field(default_factory=list)
    locale: str = "ko-KR"
    questions: list[QuestionName] = Field(default_factory=lambda: ["scenario", "intent"], min_length=1)
    shuffle_options: bool = True
    dev_sample: int = Field(default=500, ge=1)
    criteria_file: str = ""
    # Fraction of mixed train records also added as surface-perturbed copies (renamed keys, reordered or
    # re-rendered state, neutral lead-ins) so decisions depend on meaning, not formatting.
    surface_aug: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_mix(self) -> "DataSection":
        if self.source == "mix" and not self.mix:
            raise ValueError("data.source = 'mix' needs at least one [[data.mix]] entry")
        if self.source == "massive" and self.mix:
            raise ValueError("[[data.mix]] entries need data.source = 'mix'")
        names = [(m.name, m.path) for m in self.mix]
        if len(names) != len(set(names)):
            raise ValueError("each data.mix source (and each jsonl path) may appear only once")
        return self


class LoraSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = Field(ge=1)
    alpha: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: list[str] = Field(min_length=1)


class TrainSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = Field(gt=0.0)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    grad_accum: int = Field(ge=1)
    max_seq_len: int = Field(ge=64)
    label_smoothing: float = Field(ge=0.0, lt=1.0)
    # Weight of a multiclass Brier term added to the cross-entropy (0 = off): a calibration-aware objective.
    brier_weight: float = Field(default=0.0, ge=0.0, le=10.0)
    seed: int
    eval_every_steps: int = Field(ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    gradient_checkpointing: bool = True
    # "nf4" loads the base in 4-bit NF4 via bitsandbytes (QLoRA) so 4B-class bases fit an 8 GB GPU.
    quant: Literal["bf16", "nf4"] = "bf16"


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    revision: str = ""
    data: DataSection = Field(default_factory=DataSection)
    lora: LoraSection
    train: TrainSection

    @property
    def effective_batch_size(self) -> int:
        return self.train.batch_size * self.train.grad_accum


def _dotted_path(loc: tuple) -> str:
    return ".".join(str(part) for part in loc) or "<root>"


def recipe_path(name_or_path: str | Path) -> Path:
    """Resolve a recipe name (``massive-ko``) or an explicit ``.toml`` path."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".toml" and candidate.is_file():
        return candidate
    return RECIPES_DIR / f"{name_or_path}.toml"


def load_recipe(name_or_path: str | Path) -> Recipe:
    """Load and validate a recipe; violations raise E_CONFIG with the dotted field path."""
    target = recipe_path(name_or_path)
    if not target.is_file():
        raise OmjError(ErrorCode.E_CONFIG, f"recipe not found: {target}")
    try:
        with target.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise OmjError(ErrorCode.E_CONFIG, f"{target}: {exc}") from exc
    if "name" not in raw:
        raw["name"] = target.stem
    # A jsonl path is relative to the recipe file when it exists there, else to the working directory.
    for entry in (raw.get("data") or {}).get("mix") or []:
        path = entry.get("path") if isinstance(entry, dict) else None
        if path and not Path(path).expanduser().is_absolute() and (target.parent / path).is_file():
            entry["path"] = str((target.parent / path).resolve())
    try:
        return Recipe.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise OmjError(ErrorCode.E_CONFIG, f"recipe {_dotted_path(first['loc'])}: {first['msg']}") from exc
