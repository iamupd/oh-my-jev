"""Training loop on a tiny CPU model: checkpoints, train.json, calibration, resume, OOM.

# REQ-007
# REQ-008
# REQ-010
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from omj.backends.semif import render_prompt_text_with  # noqa: E402
from omj.errors import ErrorCode, OmjError  # noqa: E402
from omj.gateway.calibration import load_calibration  # noqa: E402
from omj.train import loop as loop_module  # noqa: E402
from omj.train.loop import Trainer, collate  # noqa: E402
from omj.train.recipe import Recipe  # noqa: E402

pytestmark = pytest.mark.slow

TINY_MODEL = "yujiepan/qwen2.5-tiny-random"
MAX_SEQ_LEN = 128

QUESTION: dict = {
    "type": "choice",
    "instructions": "Route this ticket.",
    "criteria": {
        "billing": "Invoices, charges, refunds",
        "technical": "Crashes, outages, bugs",
        "account": "Login and profile problems",
    },
}

STATES = [
    "The customer was charged twice for one subscription.",
    "The desktop app crashes on every startup since the update.",
    "The customer cannot sign in and the reset mail never arrives.",
    "An invoice shows a plan the customer never bought.",
    "The dashboard returns a 500 error for every report.",
    "The customer wants to change the e-mail on the profile.",
    "A refund promised last week has not been paid out.",
    "Uploads fail with a timeout on files over ten megabytes.",
    "Two-factor codes are rejected after a phone change.",
    "The annual plan renewed although it was cancelled.",
    "Search results are empty for every query today.",
    "The profile picture cannot be removed from the account.",
]


@dataclass
class Example:
    """Local stand-in for omj.train.data.TrainExample (built in parallel by T03)."""

    id: str
    row_id: str
    qid: str
    input_ids: list[int]
    label_token_ids: list[int]
    target_index: int
    keys: list[str]
    n_tokens: int


@functools.lru_cache(maxsize=1)
def _tiny_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=TINY_MODEL)


def _load_tiny() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = _tiny_path()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    return tokenizer, model


def _examples(
    tokenizer: Any, prefix: str, offset: int, count: int, *, fixed_target: int | None = None
) -> list[Example]:
    built: list[Example] = []
    for i in range(count):
        state = STATES[(offset + i) % len(STATES)]
        rendered = render_prompt_text_with(tokenizer, state, QUESTION)
        ids = tokenizer.encode(rendered.text, add_special_tokens=False)[-MAX_SEQ_LEN:]
        target = fixed_target if fixed_target is not None else i % len(rendered.keys)
        built.append(
            Example(
                id=f"{prefix}-{i}",
                row_id=f"row-{offset + i}",
                qid="route",
                input_ids=list(ids),
                label_token_ids=list(rendered.label_token_ids),
                target_index=target,
                keys=list(rendered.keys),
                n_tokens=len(ids),
            )
        )
    return built


def _recipe(**train_overrides: Any) -> Recipe:
    train = {
        "lr": 1e-3,
        "epochs": 1,
        "batch_size": 2,
        "grad_accum": 1,
        "max_seq_len": MAX_SEQ_LEN,
        "label_smoothing": 0.0,
        "seed": 1,
        "eval_every_steps": 2,
        "gradient_checkpointing": False,
    }
    train.update(train_overrides)
    return Recipe.model_validate(
        {
            "name": "tiny",
            "base_model": TINY_MODEL,
            "lora": {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
            "train": train,
        }
    )


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, list[Example], list[Example]]:
    # conftest blocks omj's downloader; this test pulls the checkpoint from the HF cache.
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    try:
        tokenizer, model = _load_tiny()
    except Exception as exc:  # network-less environments cannot fetch the checkpoint
        pytest.skip(f"tiny model {TINY_MODEL} unavailable: {exc}")
    train_examples = _examples(tokenizer, "train", 0, 8)
    dev_examples = _examples(tokenizer, "dev", 8, 4)
    return tokenizer, model, train_examples, dev_examples


def _trainer(tiny, out_dir: Path, **kwargs: Any) -> Trainer:
    tokenizer, model, train_examples, dev_examples = tiny
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("log", lambda message: None)
    return Trainer(
        _recipe(), model, tokenizer, train_examples, dev_examples, out_dir, **kwargs
    )


def test_collate_left_pads_and_masks_ragged_labels() -> None:
    # REQ-007
    short = Example("a", "r0", "q", [5, 6], [11, 12], 0, ["x", "y"], 2)
    wide = Example("b", "r1", "q", [7, 8, 9], [11, 12, 13], 2, ["x", "y", "z"], 3)
    batch = collate([short, wide], pad_token_id=0)

    assert batch["input_ids"].tolist() == [[0, 5, 6], [7, 8, 9]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batch["label_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["label_token_ids"].tolist() == [[11, 12, 0], [11, 12, 13]]
    assert batch["target_index"].tolist() == [0, 2]


def test_run_writes_best_last_train_json_and_calibration(tiny, tmp_path: Path) -> None:
    # REQ-007
    # REQ-008
    # REQ-010
    out_dir = tmp_path / "run"
    trainer = _trainer(tiny, out_dir, max_steps=4)
    result = trainer.run()

    assert result.steps == 4
    assert result.oom_retry is False
    assert result.peak_vram_gb is None  # CPU run records no CUDA peak
    assert result.wall_seconds > 0.0

    for name in ("adapter_config.json", "adapter_model.safetensors"):
        assert (out_dir / "best" / name).is_file(), name
        assert (out_dir / "last" / name).is_file(), name
    assert (out_dir / "last" / "trainer_state.pt").is_file()

    payload = json.loads(result.train_json.read_text(encoding="utf-8"))
    expected_keys = {
        "recipe",
        "run_id",
        "started_at",
        "finished_at",
        "wall_seconds",
        "git_commit",
        "omj_version",
        "device",
        "torch_version",
        "peak_vram_gb",
        "steps",
        "loss_curve",
        "dev_history",
        "best_step",
        "best_dev_accuracy",
        "data",
        "batch_size_effective",
        "oom_retry",
        "max_steps",
    }
    assert expected_keys <= set(payload)
    assert payload["recipe"]["name"] == "tiny"
    assert payload["steps"] == 4
    assert payload["batch_size_effective"] == 2
    assert payload["oom_retry"] is False
    assert payload["max_steps"] == 4
    assert payload["device"] == "cpu"
    assert len(payload["dev_history"]) >= 2
    assert [entry[0] for entry in payload["dev_history"]] == [2, 4]
    assert payload["best_step"] in {2, 4}
    assert 0.0 <= payload["best_dev_accuracy"] <= 1.0
    assert payload["loss_curve"]

    assert result.calibration_json is not None
    calibration = load_calibration(result.calibration_json)
    assert calibration.backend == "semif"
    assert calibration.model == TINY_MODEL
    assert set(calibration.temperatures) >= {"choice"}
    assert calibration.temperatures["choice"] > 0.0


def test_loss_stays_finite_and_falls_over_four_steps(tiny, tmp_path: Path) -> None:
    # REQ-007
    # A single fixed target makes four steps on an untrained tiny model a learnable
    # signal; cycling targets would only measure batch-to-batch label noise.
    tokenizer, model, _train, _dev = tiny
    train_examples = _examples(tokenizer, "train", 0, 8, fixed_target=0)
    dev_examples = _examples(tokenizer, "dev", 8, 4, fixed_target=0)
    trainer = Trainer(
        _recipe(),
        model,
        tokenizer,
        train_examples,
        dev_examples,
        tmp_path / "run",
        device="cpu",
        max_steps=4,
        log=lambda message: None,
    )
    trainer.run()

    losses = [loss for _step, loss in trainer.loss_history]
    assert len(losses) == 4
    assert all(math.isfinite(loss) and loss >= 0.0 for loss in losses)
    assert losses[-1] < losses[0] + 1.0


def test_resume_from_last_continues_the_step_counter(tiny, tmp_path: Path) -> None:
    # REQ-008
    first_dir = tmp_path / "first"
    first = _trainer(tiny, first_dir, max_steps=4)
    first.run()
    assert first.step == 4

    tokenizer, _model, train_examples, dev_examples = tiny
    from transformers import AutoModelForCausalLM

    fresh = AutoModelForCausalLM.from_pretrained(_tiny_path(), dtype=torch.float32)
    second = Trainer(
        _recipe(),
        fresh,
        tokenizer,
        train_examples,
        dev_examples,
        tmp_path / "second",
        device="cpu",
        max_steps=6,
        epochs=2,
        resume=first_dir / "last",
        log=lambda message: None,
    )
    result = second.run()

    assert result.steps == 6
    assert [step for step, _loss in second.loss_history][:4] == [1, 2, 3, 4]
    assert second.loss_history[-1][0] == 6
    assert (tmp_path / "second" / "last" / "trainer_state.pt").is_file()


def test_gradient_checkpointing_recipe_still_trains(tiny, tmp_path: Path) -> None:
    # REQ-007
    tokenizer, model, train_examples, dev_examples = tiny
    trainer = Trainer(
        _recipe(gradient_checkpointing=True),
        model,
        tokenizer,
        train_examples,
        dev_examples,
        tmp_path / "run",
        device="cpu",
        max_steps=2,
        log=lambda message: None,
    )
    result = trainer.run()

    assert result.steps == 2
    assert all(math.isfinite(loss) for _step, loss in trainer.loss_history)
    assert (tmp_path / "run" / "best" / "adapter_config.json").is_file()


def _patch_forward_to_oom(model: Any, failures: int, exc: BaseException) -> None:
    original = model.forward
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc
        return original(*args, **kwargs)

    model.forward = flaky


def _patch_forward_to_oom_at(model: Any, call_index: int, exc: BaseException) -> None:
    """Fail the ``call_index``-th forward only, so an OOM can be aimed at one micro-step."""
    original = model.forward
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any):
        calls["n"] += 1
        if calls["n"] == call_index:
            raise exc
        return original(*args, **kwargs)

    model.forward = flaky


def _record_committed_examples(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Ids of the training examples whose gradients actually reached an optimizer step.

    Micro-batches that were rolled back by an OOM are dropped from the record, so the
    returned list is exactly what the epoch trained on.
    """
    collated: list[list[str]] = []
    pending: list[str] = []
    committed: list[str] = []

    original_collate = loop_module.collate

    def spy_collate(examples, pad_token_id):  # type: ignore[no-untyped-def]
        collated.append([example.id for example in examples])
        return original_collate(examples, pad_token_id)

    original_micro = Trainer._micro_step

    def spy_micro(self, batch):  # type: ignore[no-untyped-def]
        value = original_micro(self, batch)  # an OOM raises before the batch is recorded
        pending.extend(collated[-1])
        return value

    original_step = Trainer._optimizer_step

    def spy_step(self):  # type: ignore[no-untyped-def]
        original_step(self)
        committed.extend(pending)
        pending.clear()

    original_oom = Trainer._handle_oom

    def spy_oom(self):  # type: ignore[no-untyped-def]
        pending.clear()
        original_oom(self)

    monkeypatch.setattr(loop_module, "collate", spy_collate)
    monkeypatch.setattr(Trainer, "_micro_step", spy_micro)
    monkeypatch.setattr(Trainer, "_optimizer_step", spy_step)
    monkeypatch.setattr(Trainer, "_handle_oom", spy_oom)
    return committed


def test_first_oom_halves_the_micro_batch_and_keeps_the_effective_batch(
    tiny, tmp_path: Path
) -> None:
    # REQ-010
    _tokenizer, model, _train, _dev = tiny
    _patch_forward_to_oom(model, 1, torch.cuda.OutOfMemoryError("CUDA out of memory"))

    out_dir = tmp_path / "run"
    trainer = _trainer(tiny, out_dir, max_steps=4)
    result = trainer.run()

    assert result.oom_retry is True
    assert result.steps == 4
    assert trainer.micro_batch_size == 1  # 2 halved
    assert trainer.grad_accum == 2  # 1 doubled, so the effective batch holds
    payload = json.loads(result.train_json.read_text(encoding="utf-8"))
    assert payload["oom_retry"] is True
    assert payload["micro_batch_size"] == 1
    assert payload["grad_accum"] == 2
    assert payload["batch_size_effective"] == 2  # unchanged by the halving


def test_oom_mid_accumulation_replays_the_step_and_keeps_the_epoch_intact(
    tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-007
    # REQ-010
    # 40 examples with batch_size 2 span two bucketing chunks (2 * 16), so a halving
    # that re-buckets the epoch would visibly repeat or skip examples.
    tokenizer, model, _train, _dev = tiny
    train_examples = _examples(tokenizer, "train", 0, 40)
    dev_examples = _examples(tokenizer, "dev", 40, 2)
    committed = _record_committed_examples(monkeypatch)
    # call 4 is the second micro-step of optimizer step 2: one step is already done
    # and one micro-batch of the current step is already accumulated.
    _patch_forward_to_oom_at(model, 4, torch.cuda.OutOfMemoryError("CUDA out of memory"))

    trainer = Trainer(
        _recipe(grad_accum=2, eval_every_steps=50),
        model,
        tokenizer,
        train_examples,
        dev_examples,
        tmp_path / "run",
        device="cpu",
        log=lambda message: None,
    )
    expected_steps = trainer.total_steps()
    result = trainer.run()

    assert expected_steps == 10  # 40 examples / (2 * 2) per optimizer step
    assert result.oom_retry is True
    assert result.steps == expected_steps
    assert trainer.micro_batch_size == 1
    assert trainer.grad_accum == 4
    assert trainer.effective_batch_size() == 4
    assert sorted(committed) == sorted(example.id for example in train_examples)
    assert len(committed) == len(train_examples)  # nothing trained on twice
    payload = json.loads(result.train_json.read_text(encoding="utf-8"))
    assert payload["batch_size_effective"] == 4
    assert payload["steps"] == expected_steps


def test_resume_restores_the_micro_batch_and_grad_accum_of_a_halved_run(
    tiny, tmp_path: Path
) -> None:
    # REQ-008
    # REQ-010
    _tokenizer, model, _train, _dev = tiny
    _patch_forward_to_oom(model, 1, torch.cuda.OutOfMemoryError("CUDA out of memory"))
    first_dir = tmp_path / "first"
    first = _trainer(tiny, first_dir, max_steps=4)
    first.run()
    assert (first.micro_batch_size, first.grad_accum) == (1, 2)

    tokenizer, _model, train_examples, dev_examples = tiny
    from transformers import AutoModelForCausalLM

    fresh = AutoModelForCausalLM.from_pretrained(_tiny_path(), dtype=torch.float32)
    second = Trainer(
        _recipe(),
        fresh,
        tokenizer,
        train_examples,
        dev_examples,
        tmp_path / "second",
        device="cpu",
        resume=first_dir / "last",
        log=lambda message: None,
    )
    assert (second.micro_batch_size, second.grad_accum) == (2, 1)  # recipe defaults

    second._load_state(first_dir / "last")

    assert second.micro_batch_size == 1
    assert second.grad_accum == 2
    assert second.batch_size == 2  # the bucketing/effective batch is unchanged
    assert second.effective_batch_size() == 2


def test_calibration_failure_keeps_train_json_and_records_the_error(
    tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-008
    def boom(*args: Any, **kwargs: Any):
        raise OmjError(ErrorCode.E_SCHEMA, "dev logits are unusable")

    monkeypatch.setattr(loop_module, "calibrate_from_dev", boom)
    out_dir = tmp_path / "run"

    result = _trainer(tiny, out_dir, max_steps=2).run()

    assert result.calibration_json is None
    assert not (out_dir / "calibration.json").exists()
    assert result.train_json.is_file()
    payload = json.loads(result.train_json.read_text(encoding="utf-8"))
    assert payload["calibration_error"] == "dev logits are unusable"
    assert payload["steps"] == 2


def test_resume_without_improvement_carries_best_and_still_calibrates(
    tiny, tmp_path: Path
) -> None:
    # REQ-008
    first_dir = tmp_path / "first"
    _trainer(tiny, first_dir, max_steps=2).run()
    state_path = first_dir / "last" / "trainer_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state["best_dev_accuracy"] = 1.0  # nothing the resumed run evaluates can beat this
    torch.save(state, state_path)

    tokenizer, _model, train_examples, dev_examples = tiny
    from transformers import AutoModelForCausalLM

    fresh = AutoModelForCausalLM.from_pretrained(_tiny_path(), dtype=torch.float32)
    second_dir = tmp_path / "second"
    second = Trainer(
        _recipe(),
        fresh,
        tokenizer,
        train_examples,
        dev_examples,
        second_dir,
        device="cpu",
        max_steps=4,
        epochs=2,
        resume=first_dir / "last",
        log=lambda message: None,
    )
    result = second.run()

    assert result.best_dev_accuracy == 1.0  # no checkpoint improved on the resumed best
    assert (second_dir / "best" / "adapter_config.json").is_file()
    assert (second_dir / "best" / "adapter_model.safetensors").is_file()
    assert result.calibration_json is not None
    assert result.calibration_json.is_file()
    calibration = load_calibration(result.calibration_json)
    assert calibration.temperatures["choice"] > 0.0


def test_second_consecutive_oom_raises_e_backend(tiny, tmp_path: Path) -> None:
    # REQ-010
    _tokenizer, model, _train, _dev = tiny
    _patch_forward_to_oom(model, 2, RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))

    with pytest.raises(OmjError) as info:
        _trainer(tiny, tmp_path / "run", max_steps=4).run()

    assert info.value.code is ErrorCode.E_BACKEND
    assert "halving batch size" in info.value.message
