"""semif backend on real CUDA hardware: bf16 load, three-question smoke, VRAM ceiling.
# REQ-024
# REQ-026
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="semif extra is not installed")

from omj.backends.semif import SemifBackend  # noqa: E402
from omj.config import BackendSection  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
VRAM_CEILING_BYTES = 8 * 1024**3

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


def _allow_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the multi-GB checkpoint in the real $OMJ_HOME instead of pytest's tmp_path,
    # otherwise every run re-downloads it.
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    monkeypatch.setenv("OMJ_HOME", str(Path.home() / ".omj"))


def _assert_smoke(answers: dict) -> None:
    assert set(answers) == set(QUESTIONS)
    assert answers["refund"].keys == ["yes", "no"]
    assert answers["route"].keys == ["billing", "technical", "account"]
    assert answers["urgency"].keys == ["0", "1", "2"]
    for answer in answers.values():
        assert answer.logits is not None
        assert len(answer.logits) == len(answer.keys)
        assert all(value == value for value in answer.logits)  # no NaN
        assert answer.calibrated is False


def test_bf16_load_and_smoke_stays_under_8gb(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-024, REQ-026
    _allow_downloads(monkeypatch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    backend = SemifBackend()
    backend.load(BackendSection(name="semif", model=MODEL, revision=REVISION, quant="bf16"))

    assert backend.capabilities.device == "cuda"
    assert backend.capabilities.max_options >= 2
    assert backend.model_id == f"{MODEL}@{REVISION[:8]}"
    assert backend.health().ok is True

    _assert_smoke(backend.decide(STATE, QUESTIONS))

    peak = torch.cuda.max_memory_reserved()
    assert peak < VRAM_CEILING_BYTES, f"peak VRAM {peak / 1024**3:.2f} GiB exceeds the 8 GiB budget"


def test_prequant_4bit_load_and_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-026: only a user-supplied, already-quantized bitsandbytes checkpoint is supported.
    prequant_model = os.environ.get("OMJ_PREQUANT_MODEL")
    if not prequant_model:
        pytest.skip("set OMJ_PREQUANT_MODEL to a pre-quantized bnb-4bit checkpoint id")
    pytest.importorskip("bitsandbytes", reason="bitsandbytes is required for 4bit-prequant")

    _allow_downloads(monkeypatch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    backend = SemifBackend()
    backend.load(
        BackendSection(
            name="semif",
            model=prequant_model,
            revision=os.environ.get("OMJ_PREQUANT_REVISION", ""),
            quant="4bit-prequant",
        )
    )

    assert backend.capabilities.device == "cuda"
    _assert_smoke(backend.decide(STATE, QUESTIONS))

    peak = torch.cuda.max_memory_reserved()
    assert peak < VRAM_CEILING_BYTES, f"peak VRAM {peak / 1024**3:.2f} GiB exceeds the 8 GiB budget"
