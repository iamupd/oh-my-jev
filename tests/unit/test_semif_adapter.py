"""semif LoRA adapter merge, and the prompt renderer training shares with serving.
# REQ-011
# REQ-004
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from omj.backends.semif import (  # noqa: E402
    SemifBackend,
    render_prompt_text_with,
)
from omj.config import BackendSection  # noqa: E402
from omj.errors import ErrorCode, OmjError  # noqa: E402

pytestmark = pytest.mark.slow

TINY_MODEL = "yujiepan/qwen2.5-tiny-random"

STATE = "The customer was charged twice for the same subscription and wants the extra charge back."

ROUTE_Q: dict = {
    "type": "choice",
    "instructions": "Route this ticket.",
    "criteria": {
        "billing": "Invoices, charges, refunds",
        "technical": "Crashes, outages, bugs",
        "account": "Login and profile problems",
    },
}

URGENCY_Q: dict = {
    "type": "score",
    "instructions": "How urgent is this ticket?",
    "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"],
}


@pytest.fixture(scope="module", autouse=True)
def _evict_peft_afterwards() -> Any:
    yield
    # test_train_cmd_deps.py proves require_peft() raises E_BACKEND by making the
    # import fail; importlib would hand back a peft left resident by this module.
    for name in [key for key in sys.modules if key == "peft" or key.startswith("peft.")]:
        del sys.modules[name]


@functools.lru_cache(maxsize=1)
def _tiny_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=TINY_MODEL)


def _fresh(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """A private copy of the tiny checkpoint: peft mutates the model it wraps."""
    # The conftest fixtures block omj's own downloader; this test reads the tiny
    # checkpoint straight from the HF cache instead, so the flag is cleared.
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    try:
        path = _tiny_path()
    except Exception as exc:  # network-less environments cannot fetch the checkpoint
        pytest.skip(f"tiny model {TINY_MODEL} unavailable: {exc}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.eval()
    return tokenizer, model


def _backend(monkeypatch: pytest.MonkeyPatch, adapter: str = "") -> SemifBackend:
    tokenizer, model = _fresh(monkeypatch)
    backend = SemifBackend(model=model, tokenizer=tokenizer, device="cpu")
    backend.load(BackendSection(name="semif", model=TINY_MODEL, adapter=adapter))
    return backend


def _save_lora(dest: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny r=2 LoRA whose lora_B is non-zero, so merging it must move the logits."""
    _tokenizer, model = _fresh(monkeypatch)
    wrapped = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if "lora_B" in name:
                param.normal_(mean=0.0, std=0.5)
    wrapped.save_pretrained(str(dest))
    return dest


def test_adapter_is_merged_and_tagged_into_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-011
    adapter_dir = _save_lora(tmp_path / "massive-ko-r8", monkeypatch)
    assert (adapter_dir / "adapter_config.json").is_file()

    backend = _backend(monkeypatch, adapter=str(adapter_dir))

    assert backend.model_id == f"{TINY_MODEL}@main+massive-ko-r8"
    assert backend.health().ok is True
    # merge_and_unload() leaves a plain transformers model behind, not a PeftModel.
    assert type(backend._model).__name__ != "PeftModel"

    answers = backend.decide(STATE, {"route": ROUTE_Q, "urgency": URGENCY_Q})
    assert answers["route"].keys == ["billing", "technical", "account"]
    assert len(answers["route"].logits) == 3
    assert len(answers["urgency"].logits) == 3


def test_model_id_has_no_adapter_suffix_without_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-011
    assert _backend(monkeypatch).model_id == f"{TINY_MODEL}@main"


def test_merged_adapter_changes_the_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-011
    adapter_dir = _save_lora(tmp_path / "lora-tiny", monkeypatch)

    base = _backend(monkeypatch).decide(STATE, {"route": ROUTE_Q})["route"].logits
    adapted = (
        _backend(monkeypatch, adapter=str(adapter_dir)).decide(STATE, {"route": ROUTE_Q})["route"]
    ).logits

    assert len(base) == len(adapted) == 3
    assert any(abs(left - right) > 1e-4 for left, right in zip(base, adapted))


def test_missing_adapter_directory_is_a_backend_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-011
    with pytest.raises(OmjError) as exc_info:
        _backend(monkeypatch, adapter=str(tmp_path / "absent"))
    assert exc_info.value.code == ErrorCode.E_BACKEND
    assert "absent" in exc_info.value.message


def test_directory_without_adapter_config_is_a_backend_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-011
    empty = tmp_path / "not-an-adapter"
    empty.mkdir()
    with pytest.raises(OmjError) as exc_info:
        _backend(monkeypatch, adapter=str(empty))
    assert exc_info.value.code == ErrorCode.E_BACKEND
    assert "adapter_config.json" in exc_info.value.message


def test_render_prompt_text_is_exactly_what_decide_feeds_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REQ-004
    backend = _backend(monkeypatch)
    captured: list[Any] = []
    original = backend._model.forward

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["input_ids"].clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(backend._model, "forward", spy)
    # One question per batch, so no left padding can pad the captured row.
    backend.decide(STATE, {"route": ROUTE_Q})

    rendered = backend.render_prompt_text(STATE, ROUTE_Q)
    expected = backend._tokenizer(rendered.text, add_special_tokens=False)["input_ids"]

    assert len(captured) == 1
    assert captured[0].shape[0] == 1
    assert captured[0][0].tolist() == expected
    assert rendered.keys == ["billing", "technical", "account"]
    assert rendered.labels == ["A", "B", "C"]
    assert len(rendered.label_token_ids) == 3


def test_module_helper_matches_the_backend_method(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-004
    backend = _backend(monkeypatch)
    assert render_prompt_text_with(backend._tokenizer, STATE, ROUTE_Q) == (
        backend.render_prompt_text(STATE, ROUTE_Q)
    )


def test_option_order_permutes_keys_but_keeps_the_letters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REQ-004
    backend = _backend(monkeypatch)
    plain = backend.render_prompt_text(STATE, ROUTE_Q)
    identity = backend.render_prompt_text(STATE, ROUTE_Q, option_order=[0, 1, 2])
    shuffled = backend.render_prompt_text(STATE, ROUTE_Q, option_order=[2, 0, 1])

    assert identity == plain
    assert shuffled.keys == ["account", "billing", "technical"]
    assert shuffled.labels == plain.labels == ["A", "B", "C"]
    assert shuffled.label_token_ids == plain.label_token_ids
    assert "A. account: Login and profile problems" in shuffled.text
    assert "B. billing: Invoices, charges, refunds" in shuffled.text
    assert "C. technical: Crashes, outages, bugs" in shuffled.text


def test_option_order_permutes_score_levels_by_their_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REQ-004
    backend = _backend(monkeypatch)
    shuffled = backend.render_prompt_text(STATE, URGENCY_Q, option_order=[2, 0, 1])

    assert shuffled.keys == ["2", "0", "1"]
    assert shuffled.labels == ["A", "B", "C"]
    assert "A. 2: Very urgent" in shuffled.text
    assert "C. 1: Somewhat urgent" in shuffled.text


def test_option_order_must_be_a_permutation(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-004
    backend = _backend(monkeypatch)
    with pytest.raises(ValueError, match="permutation"):
        backend.render_prompt_text(STATE, ROUTE_Q, option_order=[0, 0, 1])
