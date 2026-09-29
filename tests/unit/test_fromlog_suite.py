"""Tests for the fromlog loader: load decision items from decision logs."""
# REQ-043

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from omj.bench.suites.fromlog import load_from_log, load_from_log_with_stats
from omj.gateway.decision_log import DecisionLogRow, DecisionLogger


@pytest.fixture
def decision_logs_dir(tmp_path: Path) -> Path:
    """Create a temporary logs directory with some decision logs."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return logs_dir


def test_fromlog_loads_decision_items(decision_logs_dir):
    """REQ-043: load_from_log converts decision log rows to DecisionItems."""
    # Write two valid decision logs
    logger = DecisionLogger(decision_logs_dir, include_state=False)

    row1 = DecisionLogRow(
        ts=datetime.now(UTC).isoformat(),
        request_id="req-001",
        backend="openrouter",
        model="qwen/qwen-2.5-7b",
        state_sha256="abc123",
        questions={
            "q1": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"a": "Option A", "b": "Option B"},
            }
        },
        answers={"q1": {"choice": "a", "probabilities": {"a": 0.8, "b": 0.2}}},
        latency_ms=100.0,
        calibrated=False,
        upstream_extra={},
    )

    row2 = DecisionLogRow(
        ts=datetime.now(UTC).isoformat(),
        request_id="req-002",
        backend="typesafe",
        model="gpt-4o",
        state_sha256="def456",
        questions={
            "q2": {
                "type": "noul",
                "instructions": "Yes or no?",
            }
        },
        answers={"q2": {"noul": 0.6}},
        latency_ms=50.0,
        calibrated=True,
        upstream_extra={},
    )

    logger.write(row1)
    logger.write(row2)

    # Load from log
    items = load_from_log(decision_logs_dir)

    assert len(items) == 2

    # Check first item
    assert items[0].id == "req-001"
    assert items[0].suite == "from-log"
    assert items[0].expected is None
    assert items[0].expected_shape == "unknown"
    assert "from-log" in items[0].tags
    assert "openrouter" in items[0].tags
    assert items[0].source.startswith("decision-log:")

    # Check second item
    assert items[1].id == "req-002"
    assert items[1].suite == "from-log"
    assert items[1].expected is None
    assert items[1].expected_shape == "unknown"
    assert "from-log" in items[1].tags
    assert "typesafe" in items[1].tags


def test_fromlog_with_state(decision_logs_dir):
    """REQ-043: load_from_log handles state field when present."""
    logger = DecisionLogger(decision_logs_dir, include_state=True)

    row = DecisionLogRow(
        ts=datetime.now(UTC).isoformat(),
        request_id="req-with-state",
        backend="openrouter",
        model="qwen/qwen-2.5-7b",
        state_sha256="abc123",
        state={"text": "This is the state content"},
        questions={
            "q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}
        },
        answers={"q": {"choice": "a", "probabilities": {"a": 0.7, "b": 0.3}}},
        latency_ms=100.0,
        calibrated=False,
        upstream_extra={},
    )

    logger.write(row)
    items = load_from_log(decision_logs_dir)

    assert len(items) == 1
    assert items[0].state == {"text": "This is the state content"}


def test_fromlog_state_sha256_fallback(decision_logs_dir):
    """REQ-043: load_from_log uses state_sha256 when state is not present."""
    logger = DecisionLogger(decision_logs_dir, include_state=False)

    row = DecisionLogRow(
        ts=datetime.now(UTC).isoformat(),
        request_id="req-no-state",
        backend="openrouter",
        model="qwen/qwen-2.5-7b",
        state_sha256="abc123def456",
        questions={
            "q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A"}}
        },
        answers={},
        latency_ms=100.0,
        calibrated=False,
        upstream_extra={},
    )

    logger.write(row)
    items = load_from_log(decision_logs_dir)

    assert len(items) == 1
    assert items[0].state == {"state_sha256": "abc123def456"}


def test_fromlog_skip_rows_without_questions(decision_logs_dir):
    """REQ-043: rows without questions are skipped."""
    # Manually write a row without questions field
    logs_dir = decision_logs_dir
    import json

    date_str = datetime.now(UTC).strftime("%Y%m%d")
    log_file = logs_dir / f"decisions-{date_str}.jsonl"

    # Valid row
    valid_row = {
        "ts": datetime.now(UTC).isoformat(),
        "request_id": "req-valid",
        "backend": "openrouter",
        "model": "qwen",
        "state_sha256": "abc",
        "questions": {
            "q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A"}}
        },
        "answers": {},
        "latency_ms": 100.0,
        "calibrated": False,
        "upstream_extra": {},
    }

    # Invalid row (missing questions)
    invalid_row = {
        "ts": datetime.now(UTC).isoformat(),
        "request_id": "req-invalid",
        "backend": "openrouter",
        "model": "qwen",
        "state_sha256": "def",
        "answers": {},
        "latency_ms": 100.0,
        "calibrated": False,
        "upstream_extra": {},
    }

    log_file.write_text(json.dumps(valid_row) + "\n" + json.dumps(invalid_row) + "\n")

    # Load with stats
    items, stats = load_from_log_with_stats(logs_dir)

    assert len(items) == 1
    assert items[0].id == "req-valid"
    assert stats["rows"] == 2
    assert stats["skipped"] == 1


def test_fromlog_with_stats(decision_logs_dir):
    """REQ-043: load_from_log_with_stats returns items and stats."""
    logger = DecisionLogger(decision_logs_dir, include_state=False)

    row = DecisionLogRow(
        ts=datetime.now(UTC).isoformat(),
        request_id="req-001",
        backend="openrouter",
        model="qwen",
        state_sha256="abc",
        questions={
            "q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A"}}
        },
        answers={},
        latency_ms=100.0,
        calibrated=False,
        upstream_extra={},
    )

    logger.write(row)
    items, stats = load_from_log_with_stats(decision_logs_dir)

    assert len(items) == 1
    assert stats["rows"] == 1
    assert stats["skipped"] == 0


def test_fromlog_multiple_log_files(tmp_path: Path):
    """REQ-043: load_from_log handles multiple decision-*.jsonl files."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    import json
    from datetime import timedelta

    # Write rows with different dates
    now = datetime.now(UTC)

    # First file
    date1 = now.strftime("%Y%m%d")
    file1 = logs_dir / f"decisions-{date1}.jsonl"
    row1 = {
        "ts": now.isoformat(),
        "request_id": "req-001",
        "backend": "openrouter",
        "model": "qwen",
        "state_sha256": "abc",
        "questions": {"q": {"type": "noul", "instructions": "Yes/No?"}},
        "answers": {},
        "latency_ms": 100.0,
        "calibrated": False,
        "upstream_extra": {},
    }
    file1.write_text(json.dumps(row1) + "\n")

    # Second file (different date, if testing with multiple files)
    # For simplicity, just add more rows to same date
    row2 = {
        "ts": now.isoformat(),
        "request_id": "req-002",
        "backend": "typesafe",
        "model": "gpt-4o",
        "state_sha256": "def",
        "questions": {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A"}}},
        "answers": {},
        "latency_ms": 50.0,
        "calibrated": True,
        "upstream_extra": {},
    }
    file1.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n")

    items = load_from_log(logs_dir)
    assert len(items) == 2
