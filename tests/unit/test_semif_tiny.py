"""semif backend end-to-end on a tiny CPU model: readout, determinism, isolation, batching.
# REQ-024
"""

from __future__ import annotations

import functools
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from omj.backends.semif import BATCH_SIZE, SemifBackend  # noqa: E402
from omj.config import BackendSection  # noqa: E402

pytestmark = pytest.mark.slow

TINY_MODEL = "yujiepan/qwen2.5-tiny-random"

STATE = "The customer was charged twice for the same subscription and wants the extra charge back."

QUESTIONS: dict[str, dict] = {
    "refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?",
        "criteria": {"true": "A refund or chargeback is requested", "false": "No refund is requested"},
    },
    "route": {
        "type": "choice",
        "instructions": "Route this ticket.",
        "criteria": {
            "billing": "Invoices, charges, refunds",
            "technical": "Crashes, outages, bugs",
            "account": "Login and profile problems",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"],
    },
}


@functools.lru_cache(maxsize=1)
def _load_tiny() -> tuple[Any, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = snapshot_download(repo_id=TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.eval()
    return tokenizer, model


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> SemifBackend:
    # The fixtures in conftest block omj's own downloader; this test fetches the
    # tiny checkpoint straight from the HF cache instead, so the flag is cleared.
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    try:
        tokenizer, model = _load_tiny()
    except Exception as exc:  # network-less environments cannot fetch the checkpoint
        pytest.skip(f"tiny model {TINY_MODEL} unavailable: {exc}")

    loaded = SemifBackend(model=model, tokenizer=tokenizer, device="cpu")
    loaded.load(BackendSection(name="semif", model=TINY_MODEL, revision=""))
    return loaded


def test_load_finds_single_token_labels(backend: SemifBackend) -> None:
    # REQ-024
    assert backend.capabilities.max_options >= 2
    assert backend.capabilities.device == "cpu"
    assert backend.capabilities.calibrated is False
    assert backend.health().ok is True


def test_decide_returns_one_logit_per_key_for_every_kind(backend: SemifBackend) -> None:
    # REQ-024
    answers = backend.decide(STATE, QUESTIONS)
    assert set(answers) == set(QUESTIONS)

    assert answers["refund"].kind == "noul"
    assert answers["refund"].keys == ["yes", "no"]
    assert answers["route"].keys == ["billing", "technical", "account"]
    assert answers["urgency"].keys == ["0", "1", "2"]

    for qid, answer in answers.items():
        assert answer.qid == qid
        assert answer.probs is None
        assert answer.logits is not None
        assert len(answer.logits) == len(answer.keys)
        assert all(isinstance(value, float) for value in answer.logits)
        assert len(answer.meta["labels"]) == len(answer.keys)
        assert answer.meta["prompt_tokens"] > 0
        assert answer.calibrated is False


def test_decide_is_deterministic(backend: SemifBackend) -> None:
    # REQ-024
    first = backend.decide(STATE, QUESTIONS)
    second = backend.decide(STATE, QUESTIONS)
    for qid in QUESTIONS:
        assert first[qid].logits == second[qid].logits


def test_other_question_text_does_not_change_a_questions_logits(backend: SemifBackend) -> None:
    # REQ-024: each question gets its own prompt, so only left padding (which depends
    # on the longest prompt in the batch) can move the shared question's logits at all.
    baseline = backend.decide(STATE, {"a": QUESTIONS["route"], "b": QUESTIONS["urgency"]})
    swapped = backend.decide(
        STATE,
        {
            "a": QUESTIONS["route"],
            "b": {
                "type": "score",
                "instructions": (
                    "Completely different wording that mentions billing, refunds and outages "
                    "at much greater length so the batch padding changes."
                ),
                "criteria": ["Low", "Medium", "High"],
            },
        },
    )
    alone = backend.decide(STATE, {"a": QUESTIONS["route"]})

    assert baseline["a"].keys == swapped["a"].keys == alone["a"].keys
    for other in (swapped, alone):
        for left, right in zip(baseline["a"].logits, other["a"].logits):
            assert left == pytest.approx(right, abs=1e-4)


def test_twenty_questions_are_processed_in_two_batches(
    backend: SemifBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-024
    assert BATCH_SIZE == 16
    batch_sizes: list[int] = []
    original = backend._model.forward

    def spy(*args: Any, **kwargs: Any) -> Any:
        batch_sizes.append(int(kwargs["input_ids"].shape[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(backend._model, "forward", spy)

    questions = {
        f"q{index}": {
            "type": "choice",
            "instructions": f"Question number {index}: pick an option.",
            "criteria": {"billing": "Invoices and charges", "technical": "Crashes and bugs"},
        }
        for index in range(20)
    }
    answers = backend.decide(STATE, questions)

    assert len(answers) == 20
    assert batch_sizes == [16, 4]


def test_count_tokens_reports_the_longest_prompt(backend: SemifBackend) -> None:
    # REQ-024
    total = backend.count_tokens(STATE, QUESTIONS)
    per_question = [
        backend.count_tokens(STATE, {qid: question}) for qid, question in QUESTIONS.items()
    ]
    assert total == max(per_question)
    assert total > 0


def test_forward_runs_without_a_kv_cache(backend: SemifBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    # A single prefill per prompt: a KV cache would only hold device memory.
    seen: list[object] = []
    original = backend._model.forward

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("use_cache"))
        return original(*args, **kwargs)

    monkeypatch.setattr(backend._model, "forward", spy)
    backend.decide(STATE, QUESTIONS)
    assert seen and all(value is False for value in seen)


def test_out_of_memory_batch_is_split_and_still_answered(
    backend: SemifBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = backend.decide(STATE, QUESTIONS)
    original = backend._model.forward
    batch_sizes: list[int] = []

    def tight_memory(*args: Any, **kwargs: Any) -> Any:
        batch_sizes.append(int(kwargs["input_ids"].shape[0]))
        if kwargs["input_ids"].shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return original(*args, **kwargs)

    monkeypatch.setattr(backend._model, "forward", tight_memory)
    answers = backend.decide(STATE, QUESTIONS)

    assert batch_sizes[0] == len(QUESTIONS)
    assert set(answers) == set(QUESTIONS)
    for qid in QUESTIONS:
        for left, right in zip(expected[qid].logits, answers[qid].logits):
            assert left == pytest.approx(right, abs=1e-4)
