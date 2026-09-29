"""API key masking: mask(), redact(), RedactingFilter, and end-to-end leak checks.
# REQ-044
"""

from __future__ import annotations

import json
import logging

import pytest

from omj.gateway.decision_log import DecisionLogger, DecisionLogRow, state_sha256
from omj.logging_setup import setup_logging
from omj.redaction import SECRET_ENV_VARS, RedactingFilter, mask, redact

# Synthetic value for tests only; matches the REQ-044 example verbatim.
FAKE_JEV_KEY = "sk-live-" + "abcdefgh1234"


@pytest.fixture
def _reset_root_logger():
    import omj.logging_setup as logging_setup_module

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_filters = list(root.filters)
    saved_level = root.level
    saved_configured = logging_setup_module._configured
    # Other tests (e.g. gateway create_app) leave stderr handlers bound to
    # streams pytest has since closed; writing to those raises inside logging
    # and its "--- Logging error ---" dump would bypass redaction. Start clean.
    root.handlers[:] = []
    root.filters[:] = []
    logging_setup_module._configured = False
    yield
    root.handlers[:] = saved_handlers
    root.filters[:] = saved_filters
    root.level = saved_level
    logging_setup_module._configured = saved_configured


def test_secret_env_vars_are_the_three_upstream_keys() -> None:
    # REQ-044
    assert SECRET_ENV_VARS == ("JEV_KEY", "OPENROUTER_KEY", "TYPESAFE_API_KEY")


def test_mask_short_secret_is_fully_masked() -> None:
    # REQ-044
    assert mask("short") == "****"
    assert mask("1234567") == "****"


def test_mask_long_secret_keeps_first_four_chars() -> None:
    # REQ-044
    assert mask(FAKE_JEV_KEY) == "sk-l****"


def test_redact_replaces_env_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-044
    monkeypatch.setenv("JEV_KEY", FAKE_JEV_KEY)
    text = f"calling upstream with key={FAKE_JEV_KEY}"

    result = redact(text)

    assert FAKE_JEV_KEY not in result
    assert "sk-l****" in result


def test_redact_replaces_extra_secrets_param() -> None:
    # REQ-044
    extra = "request-scoped-" + "secret-1"
    text = f"key={extra}"

    result = redact(text, extra_secrets=[extra])

    assert extra not in result
    assert mask(extra) in result


def test_redact_masks_bearer_tokens() -> None:
    # REQ-044
    text = "Authorization: Bearer " + "abcdefghijklmnop"

    result = redact(text)

    assert "abcdefghijklmnop" not in result
    assert "Bearer abcd****" in result


def test_redact_bearer_value_stops_at_json_delimiters() -> None:
    # REQ-044: a greedy \S+ eats the closing quote and comma of a JSON string,
    # which corrupts the serialized decision-log row.
    fake_bearer = "secret" + "token123"
    text = json.dumps({"state": f"Authorization: Bearer {fake_bearer}", "backend": "kev"})

    result = redact(text)

    assert fake_bearer not in result
    assert "Bearer secr****" in result
    payload = json.loads(result)
    assert payload["backend"] == "kev"


def test_redact_bearer_value_before_closing_brace_and_bracket() -> None:
    # REQ-044
    assert redact('{"h":"Bearer abcdefghij"}') == '{"h":"Bearer abcd****"}'
    assert redact("[Bearer abcdefghij]") == "[Bearer abcd****]"


def test_redact_ignores_empty_env_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-044
    monkeypatch.delenv("JEV_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    text = "no secrets here"
    assert redact(text) == text


def test_redacting_filter_masks_message_built_from_args(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-044
    monkeypatch.setenv("JEV_KEY", FAKE_JEV_KEY)
    record = logging.LogRecord(
        name="omj.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="upstream key=%s",
        args=(FAKE_JEV_KEY,),
        exc_info=None,
    )

    kept = RedactingFilter().filter(record)

    assert kept is True
    assert FAKE_JEV_KEY not in record.getMessage()
    assert "sk-l****" in record.msg
    assert not record.args


def test_setup_logging_never_leaks_secret_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _reset_root_logger: None,
) -> None:
    # REQ-044
    import omj.logging_setup as logging_setup_module

    monkeypatch.setenv("JEV_KEY", FAKE_JEV_KEY)
    logging_setup_module._configured = False

    setup_logging(level="INFO")
    logging.getLogger("omj.upstream").info("sending request with key=%s", FAKE_JEV_KEY)

    captured = capsys.readouterr()
    assert FAKE_JEV_KEY not in captured.err
    assert "sk-l****" in captured.err


def test_setup_logging_never_leaks_secret_logged_directly_via_root(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    _reset_root_logger: None,
) -> None:
    # REQ-044: records logged straight through the root logger (rather than
    # propagated up from a child logger) hit the RedactingFilter that is
    # attached to the root logger itself, so caplog's own handler sees the
    # already-redacted record too.
    import omj.logging_setup as logging_setup_module

    monkeypatch.setenv("OPENROUTER_KEY", FAKE_JEV_KEY)
    logging_setup_module._configured = False

    setup_logging(level="INFO")
    with caplog.at_level(logging.INFO):
        logging.getLogger().info("sending request with key=%s", FAKE_JEV_KEY)

    assert FAKE_JEV_KEY not in caplog.text
    assert "sk-l****" in caplog.text


def test_setup_logging_never_leaks_secret_via_logger_exception_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _reset_root_logger: None,
) -> None:
    # REQ-044: logger.exception() renders the secret into the traceback text,
    # which the RedactingFilter must mask via record.exc_info/exc_text too,
    # not just record.msg.
    import omj.logging_setup as logging_setup_module

    monkeypatch.setenv("JEV_KEY", FAKE_JEV_KEY)
    logging_setup_module._configured = False

    setup_logging(level="INFO")
    try:
        raise ValueError(f"leak: {FAKE_JEV_KEY}")
    except ValueError:
        logging.getLogger("omj.upstream").exception("boom")

    captured = capsys.readouterr()
    assert FAKE_JEV_KEY not in captured.err
    assert "sk-l****" in captured.err


def test_setup_logging_is_idempotent(_reset_root_logger: None) -> None:
    # REQ-044
    import omj.logging_setup as logging_setup_module

    logging_setup_module._configured = False
    setup_logging()
    handler_count_after_first_call = len(logging.getLogger().handlers)

    setup_logging()
    assert len(logging.getLogger().handlers) == handler_count_after_first_call


def test_decision_log_line_never_contains_secret_embedded_in_question_instructions(
    monkeypatch: pytest.MonkeyPatch, omj_home
) -> None:
    # REQ-044
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_JEV_KEY)

    row = DecisionLogRow(
        ts="2026-09-22T12:00:00+00:00",
        request_id="22222222-2222-2222-2222-222222222222",
        backend="typesafe",
        model="qwen3.5-2b",
        state_sha256=state_sha256("state"),
        questions={"q1": {"type": "noul", "instructions": f"use key {FAKE_JEV_KEY} to verify"}},
        answers={"q1": {"type": "noul", "noul": 0.5}},
        latency_ms=10.0,
        calibrated=False,
        upstream_extra={"note": f"debug key {FAKE_JEV_KEY}"},
    )

    logger = DecisionLogger()
    path = logger.write(row)

    raw_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert FAKE_JEV_KEY not in raw_line
    assert "sk-l****" in raw_line

    payload = json.loads(raw_line)
    assert FAKE_JEV_KEY not in json.dumps(payload, ensure_ascii=False)
