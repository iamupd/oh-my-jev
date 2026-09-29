"""`omj train`: fine-tune a LoRA adapter on a recipe (REQ-009, REQ-019).

Every heavy dependency (``omj.train.data``/``loop``, ``transformers``, ``torch``,
``peft``) is imported lazily inside ``_default_deps()`` so this module can be
imported (and ``omj --help`` can list the command) even before those pieces
exist or their extras are installed. Tests inject a fake ``TrainDeps`` instead
of calling ``_default_deps()``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import typer

from omj.config import omj_home
from omj.errors import ErrorCode, OmjError

if TYPE_CHECKING:
    from omj.train.loop import TrainResult

RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
DRY_RUN_MAX_STEPS = 2
DRY_RUN_TRAIN_EXAMPLES = 16
DRY_RUN_DEV_EXAMPLES = 8


@dataclass
class TrainOptions:
    recipe: str
    out: Path | None = None
    epochs: int | None = None
    max_steps: int | None = None
    batch_size: int | None = None
    resume: Path | None = None
    dry_run: bool = False
    json_output: bool = False


@dataclass
class TrainDeps:
    """Injectable callables for every non-trivial step of `omj train`."""

    load_recipe: Callable[[str], Any] | None = None
    require_peft: Callable[[], Any] | None = None
    load_rows: Callable[..., list[dict]] | None = None
    load_criteria: Callable[..., dict] | None = None
    sample_dev: Callable[..., list[dict]] | None = None
    build_examples: Callable[..., tuple[list[Any], Any]] | None = None
    download_model: Callable[[str, str], Path] | None = None
    load_tokenizer: Callable[[Path], Any] | None = None
    load_model: Callable[[Path, str], Any] | None = None
    trainer_factory: Callable[..., Any] | None = None
    device_probe: Callable[[], bool] | None = None
    load_mix: Callable[..., tuple[list[Any], list[Any]]] | None = None
    build_examples_from_records: Callable[..., tuple[list[Any], Any]] | None = None


def load_mix_records(recipe: Any) -> tuple[list[Any], list[Any]]:
    """Train/dev ``DecisionRecord``s for a mixed recipe, sampled per [[data.mix]] entry."""
    from omj.train.sources import balanced_sample, load_source

    seed = recipe.train.seed
    train: list[Any] = []
    dev: list[Any] = []
    for entry in recipe.data.mix:
        kwargs = {"locale": recipe.data.locale, "questions": list(recipe.data.questions)}
        pool = load_source(entry.name, "train", **kwargs)
        if len(pool) < entry.train_n:
            raise OmjError(
                ErrorCode.E_CONFIG,
                f"data.mix {entry.name}: train_n={entry.train_n} exceeds the {len(pool)} available records",
            )
        train.extend(balanced_sample(pool, entry.train_n, seed))
        if entry.dev_n:
            dev_pool = load_source(entry.name, "dev", **kwargs)
            dev.extend(balanced_sample(dev_pool, min(entry.dev_n, len(dev_pool)), seed + 1))
    fraction = getattr(recipe.data, "surface_aug", 0.0)
    if fraction:
        from omj.train.sources import add_surface_augmentations

        train = add_surface_augmentations(train, fraction, seed)
    return train, dev


def _default_deps() -> TrainDeps:
    def load_recipe(name_or_path: str) -> Any:
        from omj.train.recipe import load_recipe as _load_recipe

        return _load_recipe(name_or_path)

    def require_peft() -> Any:
        from omj.train import require_peft as _require_peft

        return _require_peft()

    def load_rows(locale: str, split: str, **kwargs: Any) -> list[dict]:
        from omj.train.data import load_massive_rows

        return load_massive_rows(locale, split, **kwargs)

    def load_criteria(locale: str, criteria_file: str = "") -> dict:
        from omj.train.data import load_criteria as _load_criteria

        return _load_criteria(locale, criteria_file)

    def sample_dev(rows: list[dict], n: int, seed: int) -> list[dict]:
        from omj.train.data import sample_dev as _sample_dev

        return _sample_dev(rows, n, seed)

    def build_examples(rows: list[dict], recipe: Any, tokenizer: Any, criteria: dict, *, split: str):
        from omj.train.data import build_examples as _build_examples

        return _build_examples(rows, recipe, tokenizer, criteria, split=split)

    def download_model(model_id: str, revision: str) -> Path:
        from omj.models.download import download_model as _download_model

        return _download_model(model_id, revision)

    def load_tokenizer(model_path: Path) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)

    def load_model(model_path: Path, device: str, quant: str = "bf16") -> Any:
        import torch
        from transformers import AutoModelForCausalLM

        if quant == "nf4":
            if device != "cuda":
                raise OmjError(ErrorCode.E_BACKEND, "train.quant = 'nf4' needs a CUDA GPU")
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise OmjError(ErrorCode.E_BACKEND, "train.quant = 'nf4' needs bitsandbytes; run 'uv sync --extra semif'") from exc
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            return AutoModelForCausalLM.from_pretrained(
                str(model_path), quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0},
                trust_remote_code=False,
            )
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        return AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=dtype, trust_remote_code=False
        )

    def trainer_factory(recipe, model, tokenizer, train_examples, dev_examples, out_dir, **kwargs) -> Any:
        from omj.train.loop import Trainer

        return Trainer(recipe, model, tokenizer, train_examples, dev_examples, out_dir, **kwargs)

    def device_probe() -> bool:
        import torch

        return bool(torch.cuda.is_available())

    def build_examples_from_records(records: list[Any], recipe: Any, tokenizer: Any, *, split: str):
        from omj.train.data import build_examples_from_records as _build

        return _build(records, recipe, tokenizer, split=split)

    return TrainDeps(
        load_mix=load_mix_records,
        build_examples_from_records=build_examples_from_records,
        load_recipe=load_recipe,
        require_peft=require_peft,
        load_rows=load_rows,
        load_criteria=load_criteria,
        sample_dev=sample_dev,
        build_examples=build_examples,
        download_model=download_model,
        load_tokenizer=load_tokenizer,
        load_model=load_model,
        trainer_factory=trainer_factory,
        device_probe=device_probe,
    )


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime(RUN_ID_FORMAT)


def _print_summary(emit: Callable[[str], None], result: "TrainResult") -> None:
    emit(f"Output dir: {result.out_dir}")
    emit(f"Steps: {result.steps}")
    emit(f"best dev accuracy: {result.best_dev_accuracy}")
    emit(f"Peak VRAM (GB): {result.peak_vram_gb}")
    emit(f"Wall time (s): {result.wall_seconds}")


def run_train(opts: TrainOptions, *, deps: TrainDeps | None = None) -> "TrainResult":
    """Run `omj train` end to end and return the Trainer's TrainResult.

    Progress lines always go through `emit`, which is stdout in the default
    (human-readable) mode and stderr when `--json` is set, since `--json`'s
    stdout contract is "train.json content only".
    """
    deps = deps if deps is not None else _default_deps()

    def emit(line: str = "") -> None:
        print(line, file=sys.stderr if opts.json_output else sys.stdout)

    deps.require_peft()
    recipe = deps.load_recipe(opts.recipe)

    cuda_available = bool(deps.device_probe())
    if cuda_available:
        device = "cuda"
    elif opts.dry_run:
        device = "cpu"
    else:
        raise OmjError(
            ErrorCode.E_BACKEND,
            "omj train needs a CUDA GPU; use --dry-run on CPU",
        )

    out_dir = Path(opts.out) if opts.out is not None else omj_home() / "adapters" / recipe.name / _run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    emit(f"Recipe: {recipe.name} (base={recipe.base_model}@{recipe.revision or 'main'})")
    emit(f"Device: {device}")

    model_path = deps.download_model(recipe.base_model, recipe.revision)
    tokenizer = deps.load_tokenizer(model_path)
    quant = getattr(recipe.train, "quant", "bf16")
    model = deps.load_model(model_path, device, quant=quant) if quant != "bf16" else deps.load_model(model_path, device)
    if quant != "bf16":
        emit(f"Quantization: {quant} (QLoRA)")

    if getattr(recipe.data, "source", "massive") == "mix":
        train_records, dev_records = deps.load_mix(recipe)
        train_examples, data_stats = deps.build_examples_from_records(
            train_records, recipe, tokenizer, split="train"
        )
        dev_examples, _dev_stats = deps.build_examples_from_records(dev_records, recipe, tokenizer, split="dev")
    else:
        train_rows = deps.load_rows(recipe.data.locale, "train")
        dev_rows_full = deps.load_rows(recipe.data.locale, "dev")
        dev_rows = deps.sample_dev(dev_rows_full, recipe.data.dev_sample, recipe.train.seed)

        criteria = deps.load_criteria(recipe.data.locale, recipe.data.criteria_file)

        train_examples, data_stats = deps.build_examples(train_rows, recipe, tokenizer, criteria, split="train")
        dev_examples, _dev_stats = deps.build_examples(dev_rows, recipe, tokenizer, criteria, split="dev")

    max_steps = opts.max_steps
    if opts.dry_run:
        train_examples = train_examples[:DRY_RUN_TRAIN_EXAMPLES]
        dev_examples = dev_examples[:DRY_RUN_DEV_EXAMPLES]
        max_steps = DRY_RUN_MAX_STEPS

    emit(f"Train examples: {len(train_examples)}, dev examples: {len(dev_examples)}")

    trainer = deps.trainer_factory(
        recipe,
        model,
        tokenizer,
        train_examples,
        dev_examples,
        out_dir,
        device=device,
        max_steps=max_steps,
        epochs=opts.epochs,
        batch_size=opts.batch_size,
        resume=str(opts.resume) if opts.resume is not None else None,
        log=emit,
        data_stats=data_stats.to_dict() if hasattr(data_stats, "to_dict") else data_stats,
    )
    result = trainer.run()

    _print_summary(emit, result)

    if opts.json_output:
        typer.echo(Path(result.train_json).read_text(encoding="utf-8"), nl=False)

    return result


def register(app: typer.Typer) -> None:
    @app.command()
    def train(
        recipe: str = typer.Option(
            ..., "--recipe", help="Recipe name (recipes/<name>.toml) or a path to a .toml file."
        ),
        out: Path | None = typer.Option(
            None,
            "--out",
            help="Adapter output directory (default: $OMJ_HOME/adapters/<recipe>/<UTC run id>).",
        ),
        epochs: int | None = typer.Option(None, "--epochs", help="Override the recipe's epoch count."),
        max_steps: int | None = typer.Option(
            None, "--max-steps", help="Stop after this many optimizer steps."
        ),
        batch_size: int | None = typer.Option(
            None, "--batch-size", help="Override the recipe's per-device batch size."
        ),
        resume: Path | None = typer.Option(
            None,
            "--resume",
            help="Resume from a last/ checkpoint directory (optimizer state included).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Build data, then run 2 steps on 16 train / 8 dev examples (CPU-friendly smoke test).",
        ),
        json_output: bool = typer.Option(
            False,
            "--json",
            help="Print train.json to stdout only; every other line goes to stderr.",
        ),
    ) -> None:
        """Fine-tune a LoRA adapter on a recipe (needs peft; requires a CUDA GPU unless --dry-run)."""
        run_train(
            TrainOptions(
                recipe=recipe,
                out=out,
                epochs=epochs,
                max_steps=max_steps,
                batch_size=batch_size,
                resume=resume,
                dry_run=dry_run,
                json_output=json_output,
            )
        )
