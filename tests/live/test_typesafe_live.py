"""Live smoke test against the real TypeSafe Jev API. Skipped without a real JEV_KEY.
# REQ-023
"""

from __future__ import annotations

import os

import pytest

from omj.backends.base import RawAnswer
from omj.backends.typesafe import TypeSafeBackend
from omj.config import BackendSection

pytestmark = pytest.mark.live

_HAS_LIVE_KEY = bool(os.environ.get("JEV_KEY")) and os.environ.get("OMJ_NO_NETWORK") != "1"

QUESTIONS = {
    "wants_refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?",
        "criteria": None,
    },
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "invoice or payment issues", "technical": "bugs, crashes, outages"},
    },
    "severity": {
        "type": "score",
        "instructions": "Rate the severity of this ticket.",
        "criteria": ["low", "medium", "high"],
    },
}


@pytest.mark.skipif(not _HAS_LIVE_KEY, reason="requires a real JEV_KEY and OMJ_NO_NETWORK unset")
def test_live_decide_returns_three_shaped_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-023: tests/conftest.py's autouse block_network fixture force-sets
    # OMJ_NO_NETWORK=1 for every test; undo that here so this test can make
    # its one real upstream call.
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)

    backend = TypeSafeBackend()
    backend.load(BackendSection(provider="typesafe"))

    answers = backend.decide(
        "The customer says the app crashes on login and wants their money back.",
        QUESTIONS,
    )

    assert set(answers.keys()) == set(QUESTIONS.keys())

    refund = answers["wants_refund"]
    assert isinstance(refund, RawAnswer)
    assert refund.kind == "noul"
    assert refund.keys == ["yes", "no"]
    assert refund.probs is not None
    assert len(refund.probs) == 2
    assert refund.calibrated is True

    route = answers["route"]
    assert route.kind == "choice"
    assert route.keys == ["billing", "technical"]
    assert route.probs is not None
    assert len(route.probs) == 2

    severity = answers["severity"]
    assert severity.kind == "score"
    assert severity.keys == ["0", "1", "2"]
    assert severity.probs is not None
    assert len(severity.probs) == 3

    for answer in answers.values():
        assert answer.meta.get("upstream_model")
        assert answer.meta.get("usage") is not None
