"""LoRA training loop, checkpointing and train.json for ``omj train``.

# REQ-007 dev eval every eval_every_steps and at each epoch end, best/ on improvement
# REQ-008 best/, last/, train.json, calibration.json under out_dir
# REQ-010 peak VRAM accounting and a single automatic micro-batch halving on CUDA OOM
#         (grad_accum doubles with it, so the effective batch and step count hold)
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from omj import __version__
from omj.backends.semif import _supports_logits_to_keep
from omj.bench.metrics import top_label_ece
from omj.errors import ErrorCode, OmjError
from omj.train import require_peft
from omj.train.calibrate import DEV_SUITE, calibrate_from_dev
from omj.train.loss import restricted_brier, restricted_cross_entropy, restricted_probs
from omj.train.recipe import Recipe

ECE_BINS = 15
LOSS_CURVE_EVERY = 10
STATE_FILE = "trainer_state.pt"
ORDER_CHUNK_BATCHES = 16


class TrainExample(Protocol):
    """The shape ``omj.train.data`` produces; duck-typed so this module never imports it."""

    id: str
    row_id: str
    qid: str
    input_ids: list[int]
    label_token_ids: list[int]
    target_index: int
    keys: list[str]
    n_tokens: int


@dataclass
class TrainResult:
    out_dir: Path
    best_step: int
    best_dev_accuracy: float | None
    steps: int
    peak_vram_gb: float | None
    wall_seconds: float
    train_json: Path
    calibration_json: Path | None
    oom_retry: bool


# --------------------------------------------------------------------------- helpers


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_commit() -> str:
    """Short-lived ``git rev-parse HEAD``; "unknown" when git or the checkout is absent."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory failure, however the installed torch spells it."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def collate(examples: Sequence[TrainExample], pad_token_id: int) -> dict[str, Tensor]:
    """Left-pad a micro-batch: the readout is the last position, so padding must precede it.

    ``label_token_ids`` is right-padded to the widest K in the batch and paired with
    ``label_mask`` so questions with different option counts can share a batch.
    """
    if not examples:
        raise ValueError("collate() needs at least one example")
    size = len(examples)
    max_len = max(len(ex.input_ids) for ex in examples)
    max_k = max(len(ex.label_token_ids) for ex in examples)

    input_ids = torch.full((size, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((size, max_len), dtype=torch.long)
    label_token_ids = torch.full((size, max_k), int(pad_token_id), dtype=torch.long)
    label_mask = torch.zeros((size, max_k), dtype=torch.bool)
    target_index = torch.zeros(size, dtype=torch.long)

    for row, example in enumerate(examples):
        ids = list(example.input_ids)
        start = max_len - len(ids)
        input_ids[row, start:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, start:] = 1
        labels = list(example.label_token_ids)
        label_token_ids[row, : len(labels)] = torch.tensor(labels, dtype=torch.long)
        label_mask[row, : len(labels)] = True
        target_index[row] = int(example.target_index)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_token_ids": label_token_ids,
        "label_mask": label_mask,
        "target_index": target_index,
    }


def _enable_gradient_checkpointing(model: Any) -> None:
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if enable is None:
        return
    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:  # older transformers take no kwargs here
        enable()
    require_grads = getattr(model, "enable_input_require_grads", None)
    if require_grads is not None:
        # checkpointed blocks need an input that requires grad, or nothing is recomputed.
        require_grads()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False


def is_quantized(model: Any) -> bool:
    """True for a bitsandbytes 4/8-bit model, which is already placed and must not be ``.to()``-moved."""
    return bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False))


def apply_lora(model: Any, recipe: Recipe, *, adapter_dir: str | Path | None = None) -> Any:
    """Wrap ``model`` in a peft LoRA adapter (or reload one) and freeze the base weights.

    Trainable LoRA parameters are cast to float32 even when the base is bf16 so AdamW's
    moments stay well conditioned; the forward pass is what runs under autocast.
    """
    peft = require_peft()
    if is_quantized(model):
        # QLoRA: casts norms/embeddings to fp32 and wires input grads for checkpointing.
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=recipe.train.gradient_checkpointing
        )
    if adapter_dir is not None:
        wrapped = peft.PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    else:
        config = peft.LoraConfig(
            r=recipe.lora.r,
            lora_alpha=recipe.lora.alpha,
            lora_dropout=recipe.lora.dropout,
            target_modules=list(recipe.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        wrapped = peft.get_peft_model(model, config)

    for param in wrapped.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    if recipe.train.gradient_checkpointing:
        _enable_gradient_checkpointing(wrapped)
    return wrapped


def lr_lambda_for(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """Linear warmup over ``warmup_steps`` then cosine decay to 0 at ``total_steps``."""

    def schedule(current: int) -> float:
        if warmup_steps > 0 and current < warmup_steps:
            return float(current + 1) / float(warmup_steps)
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (current - warmup_steps) / span))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


# --------------------------------------------------------------------------- trainer


class Trainer:
    """Runs the LoRA fine-tune end to end and leaves best/, last/, train.json behind."""

    def __init__(
        self,
        recipe: Recipe,
        model: Any,
        tokenizer: Any,
        train_examples: Sequence[TrainExample],
        dev_examples: Sequence[TrainExample],
        out_dir: Path,
        *,
        device: str = "cuda",
        max_steps: int | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        resume: Path | None = None,
        log: Callable[[str], None] = print,
        backend_name: str = "semif",
        data_stats: dict | None = None,
    ) -> None:
        self.recipe = recipe
        self.tokenizer = tokenizer
        self.train_examples = list(train_examples)
        self.dev_examples = list(dev_examples)
        self.out_dir = Path(out_dir)
        self.device = torch.device(device)
        self.max_steps = max_steps
        self.epochs_total = int(epochs if epochs is not None else recipe.train.epochs)
        self.batch_size = int(batch_size if batch_size is not None else recipe.train.batch_size)
        self.grad_accum = int(recipe.train.grad_accum)
        # An OOM halves the micro-batch and doubles grad_accum, so batch_size (the
        # effective batch and the epoch bucketing chunk) never moves mid-run.
        self.micro_batch_size = self.batch_size
        self.resume = Path(resume) if resume is not None else None
        self.backend_name = backend_name
        self.data_stats = dict(data_stats) if data_stats else {}
        self._log = log

        self._last_position_only = _supports_logits_to_keep(model)
        if getattr(model, "peft_config", None) is not None:
            self.model = model
        else:
            self.model = apply_lora(model, recipe, adapter_dir=self.resume)
        if not is_quantized(model):
            self.model.to(self.device)

        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(tokenizer, "eos_token_id", None)
        if pad is None:
            raise OmjError(
                ErrorCode.E_BACKEND, "tokenizer has neither a pad token nor an eos token"
            )
        self.pad_token_id = int(pad)

        self.step = 0
        self.epoch = 0
        self.cursor = 0
        self.best_step = 0
        self.best_dev_accuracy: float | None = None
        self.oom_retry = False
        self.loss_history: list[tuple[int, float]] = []
        self.loss_curve: list[tuple[int, float]] = []
        self.dev_history: list[tuple[int, float | None, float | None]] = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}-{uuid.uuid4().hex[:6]}"

        self._accum_count = 0
        self._accum_loss = 0.0
        self._step_start_cursor = 0
        self._order: list[int] | None = None
        self._order_epoch = -1
        self._best_dev_logits: list[list[float]] | None = None
        self._step_started = time.perf_counter()
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

    # ------------------------------------------------------------------ setup

    def _trainable(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def effective_batch_size(self) -> int:
        """Examples per optimizer step; an OOM halving keeps this constant (REQ-010)."""
        return max(1, self.micro_batch_size) * max(1, self.grad_accum)

    def total_steps(self) -> int:
        steps_per_epoch = max(
            1, math.ceil(len(self.train_examples) / self.effective_batch_size())
        )
        total = steps_per_epoch * self.epochs_total
        if self.max_steps is not None:
            total = min(total, int(self.max_steps))
        return max(1, total)

    def _seed_everything(self) -> None:
        seed = int(self.recipe.train.seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

    def _build_optimizer(self) -> None:
        total = self.total_steps()
        self.optimizer = torch.optim.AdamW(
            self._trainable(),
            lr=self.recipe.train.lr,
            weight_decay=self.recipe.train.weight_decay,
        )
        warmup = int(round(self.recipe.train.warmup_ratio * total))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda_for(warmup, total)
        )

    @property
    def _order_chunk(self) -> int:
        """Bucketing chunk, pinned to the recipe batch size so an OOM never re-permutes."""
        return max(1, self.batch_size) * ORDER_CHUNK_BATCHES

    def _epoch_order(self, epoch: int) -> list[int]:
        """Shuffled order with length bucketing: each chunk of ``batch_size * 16``
        examples is sorted by token length so micro-batches mix similar lengths
        (scenario prompts are ~1/3 the length of intent prompts; mixing them pads
        every batch to the longest member)."""
        order = list(range(len(self.train_examples)))
        random.Random(int(self.recipe.train.seed) + epoch).shuffle(order)
        chunk = self._order_chunk
        lengths = [getattr(ex, "n_tokens", len(ex.input_ids)) for ex in self.train_examples]
        bucketed: list[int] = []
        for start in range(0, len(order), chunk):
            part = order[start : start + chunk]
            part.sort(key=lambda idx: lengths[idx], reverse=True)
            bucketed.extend(part)
        return bucketed

    def _order_for(self, epoch: int) -> list[int]:
        """The permutation for ``epoch``, computed once and reused for the whole epoch."""
        if self._order is None or self._order_epoch != epoch:
            self._order = self._epoch_order(epoch)
            self._order_epoch = epoch
        return self._order

    # ------------------------------------------------------------------ forward

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    def _last_logits(self, batch: dict[str, Tensor]) -> Tensor:
        kwargs: dict[str, Any] = {}
        if self._last_position_only:
            kwargs["logits_to_keep"] = 1
        outputs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs
        )
        return outputs.logits[:, -1, :]

    def _to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _micro_step(self, batch: dict[str, Tensor]) -> float:
        self.model.train()
        batch = self._to_device(batch)
        with self._autocast():
            last_logits = self._last_logits(batch)
        loss = restricted_cross_entropy(
            last_logits,
            batch["label_token_ids"],
            batch["target_index"],
            label_smoothing=self.recipe.train.label_smoothing,
            label_mask=batch["label_mask"],
        )
        brier_weight = getattr(self.recipe.train, "brier_weight", 0.0)
        if brier_weight:
            loss = loss + brier_weight * restricted_brier(
                last_logits, batch["label_token_ids"], batch["target_index"], label_mask=batch["label_mask"]
            )
        (loss / self.grad_accum).backward()
        return float(loss.detach().item())

    # ------------------------------------------------------------------ eval

    def evaluate(
        self, examples: Sequence[TrainExample]
    ) -> tuple[float | None, float | None, list[list[float]]]:
        """Dev accuracy, 15-bin top-label ECE and the restricted logits per example."""
        if not examples:
            return None, None, []
        self.model.eval()
        logits_rows: list[list[float]] = []
        confidences: list[float] = []
        corrects: list[bool] = []
        chunk_size = max(1, self.micro_batch_size)

        with torch.no_grad():
            for start in range(0, len(examples), chunk_size):
                chunk = list(examples[start : start + chunk_size])
                batch = self._to_device(collate(chunk, self.pad_token_id))
                with self._autocast():
                    last_logits = self._last_logits(batch)
                last_logits = last_logits.float()
                gathered = torch.gather(last_logits, dim=-1, index=batch["label_token_ids"])
                probs = restricted_probs(
                    last_logits, batch["label_token_ids"], batch["label_mask"]
                )
                for row, example in enumerate(chunk):
                    width = len(example.label_token_ids)
                    row_logits = [float(v) for v in gathered[row, :width].tolist()]
                    row_probs = [float(v) for v in probs[row, :width].tolist()]
                    predicted = max(range(width), key=lambda i: row_probs[i])
                    logits_rows.append(row_logits)
                    confidences.append(row_probs[predicted])
                    corrects.append(predicted == int(example.target_index))

        accuracy = sum(1 for hit in corrects if hit) / len(corrects)
        ece = top_label_ece(confidences, corrects, ECE_BINS)
        return accuracy, ece, logits_rows

    def _evaluate_and_checkpoint(self) -> None:
        if not self.dev_examples:
            self._save_last()
            return
        if self.dev_history and self.dev_history[-1][0] == self.step:
            return
        accuracy, ece, logits_rows = self.evaluate(self.dev_examples)
        self.dev_history.append((self.step, accuracy, ece))
        self._log(f"eval step={self.step} dev_accuracy={accuracy:.4f} dev_ece={ece:.4f}")
        if accuracy is not None and (
            self.best_dev_accuracy is None or accuracy > self.best_dev_accuracy
        ):
            self.best_dev_accuracy = accuracy
            self.best_step = self.step
            self._best_dev_logits = logits_rows
            self.model.save_pretrained(str(self.out_dir / "best"))
        self._save_last()

    # ------------------------------------------------------------------ state

    def _save_last(self) -> None:
        last_dir = self.out_dir / "last"
        last_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(last_dir))
        state = {
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "step": self.step,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "batch_size": self.batch_size,
            "micro_batch_size": self.micro_batch_size,
            "grad_accum": self.grad_accum,
            "best_step": self.best_step,
            "best_dev_accuracy": self.best_dev_accuracy,
            "oom_retry": self.oom_retry,
            "loss_history": self.loss_history,
            "loss_curve": self.loss_curve,
            "dev_history": self.dev_history,
            "run_id": self.run_id,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(state, last_dir / STATE_FILE)

    def _load_state(self, directory: Path) -> None:
        state_path = Path(directory) / STATE_FILE
        if not state_path.is_file():
            raise OmjError(
                ErrorCode.E_CONFIG, f"resume checkpoint has no {STATE_FILE}: {directory}"
            )
        # our own checkpoint: it holds RNG tuples, which weights-only loading rejects.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("optimizer") and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state.get("step", 0))
        self.epoch = int(state.get("epoch", 0))
        self.cursor = int(state.get("cursor", 0))
        self._step_start_cursor = self.cursor
        self.batch_size = int(state.get("batch_size", self.batch_size))
        self.grad_accum = int(state.get("grad_accum", self.grad_accum))
        # pre-REQ-010-fix checkpoints stored the halved micro-batch under "batch_size".
        self.micro_batch_size = int(state.get("micro_batch_size", self.batch_size))
        self._order = None
        self._order_epoch = -1
        self.best_step = int(state.get("best_step", 0))
        self.best_dev_accuracy = state.get("best_dev_accuracy")
        self.oom_retry = bool(state.get("oom_retry", False))
        self.loss_history = [tuple(item) for item in state.get("loss_history", [])]
        self.loss_curve = [tuple(item) for item in state.get("loss_curve", [])]
        self.dev_history = [tuple(item) for item in state.get("dev_history", [])]
        rng = state.get("rng") or {}
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

    # ------------------------------------------------------------------ loop

    def _optimizer_step(self) -> None:
        assert self.optimizer is not None
        torch.nn.utils.clip_grad_norm_(self._trainable(), self.recipe.train.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1

        loss = self._accum_loss / max(1, self._accum_count)
        self._accum_count = 0
        self._accum_loss = 0.0
        self.loss_history.append((self.step, loss))
        if self.step == 1 or self.step % LOSS_CURVE_EVERY == 0:
            self.loss_curve.append((self.step, loss))

        total = self.total_steps()
        elapsed = time.perf_counter() - self._step_started
        eta = elapsed / self.step * max(0, total - self.step) if self.step else 0.0
        lr = self.scheduler.get_last_lr()[0]
        self._log(f"step {self.step}/{total} loss={loss:.4f} lr={lr:.2e} eta={eta:.0f}s")

        if self.step % self.recipe.train.eval_every_steps == 0:
            self._evaluate_and_checkpoint()

    def _handle_oom(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self._accum_count = 0
        self._accum_loss = 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.oom_retry:
            raise OmjError(
                ErrorCode.E_BACKEND, "CUDA out of memory even after halving batch size"
            )
        self.oom_retry = True
        halved = max(1, self.micro_batch_size // 2)
        if halved < self.micro_batch_size:
            # doubling grad_accum keeps the effective batch and the optimizer-step
            # count (and therefore the LR schedule) exactly as they were.
            self.grad_accum *= 2
        self.micro_batch_size = halved
        self.cursor = self._step_start_cursor
        self._log(
            f"CUDA OOM: retrying the optimizer step with micro_batch_size="
            f"{self.micro_batch_size} and grad_accum={self.grad_accum} "
            f"(effective batch {self.effective_batch_size()} unchanged)"
        )

    def _stop_step(self) -> float:
        return float(self.max_steps) if self.max_steps is not None else math.inf

    def _train(self) -> None:
        stop = self._stop_step()
        while self.epoch < self.epochs_total and self.step < stop:
            order = self._order_for(self.epoch)
            while self.cursor < len(order) and self.step < stop:
                if self._accum_count == 0:
                    self._step_start_cursor = self.cursor
                indices = order[self.cursor : self.cursor + self.micro_batch_size]
                batch = collate([self.train_examples[i] for i in indices], self.pad_token_id)
                try:
                    loss_value = self._micro_step(batch)
                except BaseException as exc:  # noqa: BLE001 - re-raised unless it is an OOM
                    if not is_oom(exc):
                        raise
                    # rolls the cursor back to the first micro-batch of this
                    # optimizer step, so every example is still seen exactly once.
                    self._handle_oom()
                    continue
                self.cursor += len(indices)
                self._accum_count += 1
                self._accum_loss += loss_value
                if self._accum_count >= self.grad_accum or self.cursor >= len(order):
                    self._optimizer_step()
            if self.cursor >= len(order):
                self.epoch += 1
                self.cursor = 0
                self._evaluate_and_checkpoint()

    # ------------------------------------------------------------------ resumed best

    def _inherit_best(self) -> None:
        """Carry the resumed run's ``best/`` into the new out_dir (REQ-008).

        Without this a resume that never improves leaves out_dir with no best
        adapter and no calibration.json, losing the earlier best checkpoint.
        """
        if self.resume is None:
            return
        source = self.resume.parent / "best"
        target = self.out_dir / "best"
        if not source.is_dir() or target.exists() or source.resolve() == target.resolve():
            return
        shutil.copytree(source, target)
        self._log(f"resume: carried {source} into {target}")

    def _dev_logits_from_best(self) -> list[list[float]] | None:
        """Dev logits under ``out_dir/best``, so a run that never improved still calibrates."""
        best_dir = self.out_dir / "best"
        load_adapter = getattr(self.model, "load_adapter", None)
        set_adapter = getattr(self.model, "set_adapter", None)
        if load_adapter is None or set_adapter is None:
            return None
        name = "omj_best"
        try:
            load_adapter(str(best_dir), adapter_name=name)
            set_adapter(name)
        except Exception as exc:  # noqa: BLE001 - calibration is best effort here
            self._log(f"warning: could not load {best_dir} for calibration: {exc}")
            return None
        try:
            _accuracy, _ece, logits_rows = self.evaluate(self.dev_examples)
            return logits_rows
        finally:
            try:
                set_adapter("default")
                delete_adapter = getattr(self.model, "delete_adapter", None)
                if delete_adapter is not None:
                    delete_adapter(name)
            except Exception as exc:  # noqa: BLE001 - the run is over; only log it
                self._log(f"warning: could not restore the active adapter: {exc}")

    # ------------------------------------------------------------------ run

    def run(self) -> TrainResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._seed_everything()
        self._build_optimizer()
        if self.resume is not None:
            self._load_state(self.resume)
            self._inherit_best()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        started_at = _utc_now()
        self._step_started = time.perf_counter()
        self._train()
        self._evaluate_and_checkpoint()
        wall_seconds = time.perf_counter() - self._step_started
        finished_at = _utc_now()

        peak_vram_gb = None
        if self.device.type == "cuda":
            peak_vram_gb = float(torch.cuda.max_memory_allocated()) / 1e9

        # train.json is written before calibration so a calibration failure can
        # never cost the run its manifest (REQ-008).
        train_json = self._write_train_json(
            started_at=started_at,
            finished_at=finished_at,
            wall_seconds=wall_seconds,
            peak_vram_gb=peak_vram_gb,
        )

        if self.dev_examples and self._best_dev_logits is None and (self.out_dir / "best").is_dir():
            self._best_dev_logits = self._dev_logits_from_best()

        calibration_json: Path | None = None
        calibration_error: str | None = None
        if self.dev_examples and self._best_dev_logits is not None:
            target = self.out_dir / "calibration.json"
            try:
                calibrate_from_dev(
                    self.dev_examples,
                    self._best_dev_logits,
                    backend=self.backend_name,
                    model=self.recipe.base_model,
                    out_path=target,
                )
            except OmjError as exc:
                calibration_error = exc.message
                self._log(f"warning: calibration failed, no calibration.json: {exc.message}")
            else:
                calibration_json = target

        if calibration_error is not None:
            train_json = self._write_train_json(
                started_at=started_at,
                finished_at=finished_at,
                wall_seconds=wall_seconds,
                peak_vram_gb=peak_vram_gb,
                calibration_error=calibration_error,
            )
        self._log(
            f"done steps={self.step} best_step={self.best_step} "
            f"best_dev_accuracy={self.best_dev_accuracy} peak_vram_gb={peak_vram_gb} "
            f"out={self.out_dir}"
        )
        return TrainResult(
            out_dir=self.out_dir,
            best_step=self.best_step,
            best_dev_accuracy=self.best_dev_accuracy,
            steps=self.step,
            peak_vram_gb=peak_vram_gb,
            wall_seconds=wall_seconds,
            train_json=train_json,
            calibration_json=calibration_json,
            oom_retry=self.oom_retry,
        )

    def _write_train_json(
        self,
        *,
        started_at: str,
        finished_at: str,
        wall_seconds: float,
        peak_vram_gb: float | None,
        calibration_error: str | None = None,
    ) -> Path:
        payload = {
            "recipe": self.recipe.model_dump(),
            "run_id": self.run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": wall_seconds,
            "git_commit": git_commit(),
            "omj_version": __version__,
            "device": str(self.device),
            "torch_version": torch.__version__,
            "peak_vram_gb": peak_vram_gb,
            "steps": self.step,
            "loss_curve": [[step, loss] for step, loss in self.loss_curve],
            "dev_history": [[step, accuracy, ece] for step, accuracy, ece in self.dev_history],
            "best_step": self.best_step,
            "best_dev_accuracy": self.best_dev_accuracy,
            "data": self.data_stats,
            "batch_size_effective": self.effective_batch_size(),
            "micro_batch_size": self.micro_batch_size,
            "grad_accum": self.grad_accum,
            "oom_retry": self.oom_retry,
            "max_steps": self.max_steps,
            "dev_suite": DEV_SUITE,
            "calibration_error": calibration_error,
        }
        target = self.out_dir / "train.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return target
