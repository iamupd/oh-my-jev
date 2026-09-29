"""Local decision model: one forward pass per question, option-letter logit readout.

The model never generates text. Each question is rendered into its own prompt,
the batch is left-padded, and the next-token logits at the last position are
narrowed to the option-letter token ids. That keeps question A's wording out of
question B's distribution (REQ-024) and gives the gateway raw logits to calibrate.
"""

from __future__ import annotations

import inspect
import logging
import re
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from omj.backends.adapter_info import ADAPTER_CONFIG_FILE, adapter_base_model  # noqa: F401 (re-exported)
from omj.backends.base import Capabilities, Health, Kind, RawAnswer
from omj.backends.semif_prompt import (
    ASSISTANT_PREFIX,
    LABELS,
    build_messages,
    question_options,
    render_question,
)
from omj.config import BackendSection
from omj.errors import ErrorCode, OmjError
from omj.models.download import download_model

logger = logging.getLogger(__name__)

BATCH_SIZE = 16
DEFAULT_MAX_STATE_TOKENS = 4096
MIN_SINGLE_TOKEN_LABELS = 2


def validate_adapter_dir(path: str | Path) -> Path:
    """The one adapter-directory check shared by `omj serve`, `omj bench` and load (REQ-012)."""
    adapter_dir = Path(path).expanduser()
    if not adapter_dir.is_dir():
        raise OmjError(ErrorCode.E_BACKEND, f"adapter directory does not exist: {adapter_dir}")
    if not (adapter_dir / ADAPTER_CONFIG_FILE).is_file():
        raise OmjError(
            ErrorCode.E_BACKEND,
            f"{adapter_dir} is not a LoRA adapter: {ADAPTER_CONFIG_FILE} is missing",
        )
    return adapter_dir


def check_adapter_base(adapter: str | Path, model: str) -> None:
    """Fail before loading weights when the adapter belongs to a different base model."""
    base = adapter_base_model(Path(adapter))
    if base is None or not model or base.lower() == model.lower():
        return
    raise OmjError(
        ErrorCode.E_CONFIG,
        f"adapter {adapter} was trained on {base}, but the configured model is {model}",
        hint=(
            f"Use a config whose [backend] model is {base}: set OMJ_CONFIG to it, or create one with "
            f"'uv run omj init --model {base} --no-download'. Otherwise pick an adapter trained on {model}."
        ),
    )


@dataclass(frozen=True)
class RenderedPrompt:
    """One question's finished model input: the exact text plus its readout table."""

    text: str
    keys: list[str]
    labels: list[str]
    label_token_ids: list[int]


def single_token_labels(
    tokenizer: Any, labels: list[str] | None = None
) -> tuple[list[str], list[int]]:
    """Keep the labels that encode to exactly one token when preceded by a space.

    The readout position follows "Answer:", so the letter the model would emit
    there carries a leading space; probing " A" rather than "A" matches it.
    """
    candidates = LABELS if labels is None else labels
    kept: list[str] = []
    token_ids: list[int] = []
    for label in candidates:
        encoded = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(encoded) == 1:
            kept.append(label)
            token_ids.append(int(encoded[0]))
    return kept, token_ids


_LABEL_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def cached_single_token_labels(tokenizer: Any) -> tuple[list[str], list[int]]:
    """single_token_labels() memoised per tokenizer: 255 encode() calls per prompt is too many."""
    try:
        cached = _LABEL_CACHE.get(tokenizer)
    except TypeError:  # a tokenizer stub that cannot be weak-referenced
        return single_token_labels(tokenizer)
    if cached is None:
        cached = single_token_labels(tokenizer)
        _LABEL_CACHE[tokenizer] = cached
    return cached


def _apply_template(tokenizer: Any, prompt_body: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        # Base checkpoints ship no chat template; the same body plus the
        # assistant prefix keeps the readout position identical.
        return f"{prompt_body}\n{ASSISTANT_PREFIX}"
    text = tokenizer.apply_chat_template(
        build_messages(prompt_body), tokenize=False, add_generation_prompt=True
    )
    return f"{text}{ASSISTANT_PREFIX}"


def _permuted_question(question: dict, option_order: list[int]) -> dict:
    """Re-express the question as a 'choice' over its options in `option_order`.

    Every kind renders as "<letter>. <key>: <text>" and derives its keys from the
    option order, so a choice over the already-extracted pairs reproduces the
    original rendering exactly while letting the caller permute it.
    """
    options = question_options(question)
    if sorted(option_order) != list(range(len(options))):
        raise ValueError(
            f"option_order must be a permutation of range({len(options)}), got {option_order!r}"
        )
    return {
        "type": "choice",
        "instructions": question["instructions"],
        "criteria": {options[index][0]: options[index][1] for index in option_order},
    }


def render_prompt_text_with(
    tokenizer: Any,
    state: str | dict | list,
    question: dict,
    *,
    option_order: list[int] | None = None,
) -> RenderedPrompt:
    """Render one question exactly as the semif backend feeds it to the model.

    Training calls this with its own tokenizer so its examples and the serving
    path cannot drift apart (REQ-004).
    """
    labels, token_ids = cached_single_token_labels(tokenizer)
    target = question if option_order is None else _permuted_question(question, option_order)
    body, keys, labels_used = render_question(state, target, labels)
    by_label = dict(zip(labels, token_ids))
    return RenderedPrompt(
        text=_apply_template(tokenizer, body),
        keys=list(keys),
        labels=list(labels_used),
        label_token_ids=[by_label[label] for label in labels_used],
    )


def _infer_device(model: Any) -> str:
    try:
        return next(model.parameters()).device.type
    except (AttributeError, StopIteration, TypeError):
        return "cpu"


def _supports_logits_to_keep(model: Any) -> bool:
    """Whether forward() can skip the logits for every position but the last.

    A full [batch, seq, vocab] tensor is several GB for a 2B model at batch 16,
    which alone would blow the 8 GB VRAM budget (REQ-026).
    """
    try:
        return "logits_to_keep" in inspect.signature(model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return False


def _max_state_tokens(model: Any) -> int:
    declared = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(declared, int) and 0 < declared < DEFAULT_MAX_STATE_TOKENS:
        return declared
    return DEFAULT_MAX_STATE_TOKENS


class SemifBackend:
    """transformers-backed backend. Pass `model`/`tokenizer` to skip the download."""

    name = "semif"

    def __init__(
        self,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: str | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._last_position_only = False
        self._loaded = False
        self.model_id = "semif"
        self.capabilities = Capabilities(
            max_options=len(LABELS),
            max_state_tokens=DEFAULT_MAX_STATE_TOKENS,
            supports_batch=True,
            calibrated=False,
            device=device or "cpu",
        )

    def load(self, cfg: BackendSection) -> None:
        self.model_id = f"{cfg.model}@{cfg.revision[:8] or 'main'}"
        if cfg.adapter:
            check_adapter_base(validate_adapter_dir(cfg.adapter), cfg.model)

        if self._model is None:
            self._load_from_hub(cfg)
        if self._tokenizer is None:
            raise OmjError(ErrorCode.E_BACKEND, "semif backend was given a model but no tokenizer")

        if cfg.adapter:
            self._merge_adapter(cfg.adapter)

        self._prepare_tokenizer()

        labels, _token_ids = cached_single_token_labels(self._tokenizer)
        if len(labels) < MIN_SINGLE_TOKEN_LABELS:
            raise OmjError(
                ErrorCode.E_BACKEND,
                f"{self.model_id}: only {len(labels)} of {len(LABELS)} option labels encode to a "
                f"single token; at least {MIN_SINGLE_TOKEN_LABELS} are required",
            )

        if hasattr(self._model, "eval"):
            self._model.eval()
        self._last_position_only = _supports_logits_to_keep(self._model)
        self._device = self._device or _infer_device(self._model)
        self.capabilities = Capabilities(
            max_options=len(labels),
            max_state_tokens=_max_state_tokens(self._model),
            supports_batch=True,
            calibrated=False,
            device=self._device,
        )
        self._loaded = True
        logger.info(
            "semif loaded %s on %s with %d single-token labels",
            self.model_id,
            self._device,
            len(labels),
        )

    def render_prompt_text(
        self,
        state: str | dict | list,
        question: dict,
        *,
        option_order: list[int] | None = None,
    ) -> RenderedPrompt:
        """The prompt this backend would send for one question (REQ-004)."""
        if self._tokenizer is None:
            raise OmjError(ErrorCode.E_BACKEND, "semif backend is not loaded; call load() first")
        return render_prompt_text_with(
            self._tokenizer, state, question, option_order=option_order
        )

    def decide(
        self, state: str | dict | list, questions: dict[str, dict]
    ) -> dict[str, RawAnswer]:
        self._require_loaded()

        rendered: list[tuple[str, Kind, RenderedPrompt]] = []
        for qid, question in questions.items():
            rendered.append((qid, question["type"], self._render(qid, state, question)))

        answers: dict[str, RawAnswer] = {}
        for start in range(0, len(rendered), BATCH_SIZE):
            answers.update(self._forward_chunk(rendered[start : start + BATCH_SIZE]))
        return answers

    def health(self) -> Health:
        if not self._loaded:
            return Health(ok=False, detail="semif backend is not loaded")
        return Health(ok=True, detail=f"{self.model_id} on {self._device}")

    def count_tokens(self, state: str | dict | list, questions: dict[str, dict]) -> int:
        """Longest single prompt, because each question is its own forward pass.

        Summing would overstate the context each pass has to fit, which is what
        the 422 context_length_exceeded budget is about.
        """
        self._require_loaded()
        longest = 0
        for qid, question in questions.items():
            prompt = self._render(qid, state, question)
            encoded = self._tokenizer.encode(prompt.text, add_special_tokens=False)
            longest = max(longest, len(encoded))
        return longest

    def _render(self, qid: str, state: str | dict | list, question: dict) -> RenderedPrompt:
        try:
            return self.render_prompt_text(state, question)
        except ValueError as exc:
            raise OmjError(ErrorCode.E_BACKEND, f"question {qid!r}: {exc}") from exc

    def _merge_adapter(self, adapter: str) -> None:
        """Fold a LoRA adapter into the base weights so inference stays a plain forward (REQ-011)."""
        adapter_dir = validate_adapter_dir(adapter)

        from omj.train import require_peft

        peft = require_peft()
        try:
            wrapped = peft.PeftModel.from_pretrained(self._model, str(adapter_dir))
            # Merging into a 4-bit base would re-quantize the LoRA delta; keep it as a live adapter.
            quantized = getattr(self._model, "is_loaded_in_4bit", False)
            self._model = wrapped if quantized else wrapped.merge_and_unload()
        except Exception as exc:
            raise OmjError(
                ErrorCode.E_BACKEND, f"could not apply adapter {adapter_dir}: {exc}"
            ) from exc

        # An adapter from the omj model cache sits in <org>__<name>/<revision>: show the repo name.
        label = adapter_dir.name
        if (label == "main" or re.fullmatch(r"[0-9a-f]{40}", label)) and "__" in adapter_dir.parent.name:
            label = adapter_dir.parent.name.split("__", 1)[1]
        self.model_id = f"{self.model_id}+{label}"
        logger.info("semif merged adapter %s into %s", adapter_dir, self.model_id)

    def _load_from_hub(self, cfg: BackendSection) -> None:
        path = download_model(cfg.model, cfg.revision, offline=cfg.offline)
        cuda = torch.cuda.is_available()

        if cfg.quant == "4bit-prequant":
            if not cuda:
                raise OmjError(
                    ErrorCode.E_NO_GPU,
                    "quant '4bit-prequant' needs a CUDA GPU; use quant 'bf16' on CPU",
                )
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise OmjError(
                    ErrorCode.E_BACKEND,
                    "quant '4bit-prequant' needs bitsandbytes to read the pre-quantized "
                    "checkpoint; install it or use quant 'bf16'",
                ) from exc
            # The checkpoint is already quantized, so no quantization_config is passed:
            # omj never quantizes a model itself (design 5.4).
            model_kwargs: dict[str, Any] = {"device_map": "cuda"}
        elif cfg.quant == "nf4":
            if not cuda:
                raise OmjError(ErrorCode.E_NO_GPU, "quant 'nf4' needs a CUDA GPU; use quant 'bf16' on CPU")
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise OmjError(ErrorCode.E_BACKEND, "quant 'nf4' needs bitsandbytes; run 'uv sync --extra semif'") from exc
            # Same NF4 settings the QLoRA trainer uses, so a 4B adapter is served on the base it was trained on.
            model_kwargs = {
                "device_map": "cuda",
                "dtype": torch.bfloat16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
            }
        else:
            if not cuda:
                logger.warning(
                    "no CUDA device available; loading %s on CPU in float32 (slow)", cfg.model
                )
            model_kwargs = {"dtype": torch.bfloat16 if cuda else torch.float32}
            if cuda:
                model_kwargs["device_map"] = "cuda"

        self._tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(path), trust_remote_code=False, **model_kwargs
        )
        self._model.eval()

    def _prepare_tokenizer(self) -> None:
        tokenizer = self._tokenizer
        # Left padding keeps the readout position (the last column) aligned with the
        # real end of every prompt in a batch.
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise OmjError(
                    ErrorCode.E_BACKEND,
                    f"{self.model_id}: tokenizer has neither a pad token nor an eos token",
                )
            tokenizer.pad_token = eos_token

    def _forward_chunk(
        self, chunk: list[tuple[str, Kind, RenderedPrompt]]
    ) -> dict[str, RawAnswer]:
        device = torch.device(self._device or "cpu")
        encoded = self._tokenizer(
            [item[2].text for item in chunk],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        forward_kwargs: dict[str, Any] = {}
        if self._last_position_only:
            forward_kwargs["logits_to_keep"] = 1
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids, attention_mask=attention_mask, **forward_kwargs
            )
        last_logits = outputs.logits[:, -1, :].float()
        prompt_tokens = attention_mask.sum(dim=1).tolist()

        answers: dict[str, RawAnswer] = {}
        for row, (qid, kind, prompt) in enumerate(chunk):
            index = torch.tensor(
                prompt.label_token_ids, dtype=torch.long, device=last_logits.device
            )
            values = last_logits[row].index_select(0, index).tolist()
            answers[qid] = RawAnswer(
                qid=qid,
                kind=kind,
                keys=list(prompt.keys),
                logits=[float(value) for value in values],
                calibrated=False,
                meta={"labels": list(prompt.labels), "prompt_tokens": int(prompt_tokens[row])},
            )
        return answers

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise OmjError(ErrorCode.E_BACKEND, "semif backend is not loaded; call load() first")
