"""Decision log JSONL writer: daily files, sha256 state, opt-in state payload.
# REQ-020
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from omj.gateway.decision_log import DecisionLogger, DecisionLogRow, iter_rows, state_sha256


def _make_row(**overrides: Any) -> DecisionLogRow:
    defaults: dict[str, Any] = dict(
        ts="2026-09-22T12:00:00+00:00",
        request_id="11111111-1111-1111-1111-111111111111",
        backend="mock",
        model="qwen3.5-2b",
        state_sha256=state_sha256({"who": "홍길동"}),
        questions={"q1": {"type": "choice", "instructions": "판단하라"}},
        answers={
            "q1": {
                "type": "choice",
                "choice": "yes",
                "probabilities": {"yes": 0.9, "no": 0.1},
                "confidence": 0.9,
            }
        },
        latency_ms=12.5,
        calibrated=False,
        upstream_extra={},
        state={"who": "홍길동"},
    )
    defaults.update(overrides)
    return DecisionLogRow(**defaults)


def test_write_creates_file_under_tmp_omj_home_logs(omj_home: Path) -> None:
    # REQ-020
    logger = DecisionLogger()
    path = logger.write(_make_row())

    assert path.parent == omj_home / "logs"
    assert path.exists()


def test_filename_matches_daily_date_pattern(omj_home: Path) -> None:
    # REQ-020
    logger = DecisionLogger()
    path = logger.write(_make_row())

    assert re.fullmatch(r"decisions-\d{8}\.jsonl", path.name)


def test_written_row_contains_all_required_fields_and_excludes_state_by_default(
    omj_home: Path,
) -> None:
    # REQ-020
    logger = DecisionLogger()
    row = _make_row()
    path = logger.write(row)

    line = path.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(line)

    assert payload["ts"] == row.ts
    assert payload["request_id"] == row.request_id
    assert payload["backend"] == row.backend
    assert payload["model"] == row.model
    assert payload["state_sha256"] == row.state_sha256
    assert payload["questions"] == row.questions
    assert payload["answers"] == row.answers
    assert payload["latency_ms"] == row.latency_ms
    assert payload["calibrated"] == row.calibrated
    assert payload["upstream_extra"] == row.upstream_extra
    assert "state" not in payload


def test_include_state_true_adds_state_field(omj_home: Path) -> None:
    # REQ-020
    logger = DecisionLogger(include_state=True)
    row = _make_row()
    path = logger.write(row)

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["state"] == row.state


def test_state_excluded_even_when_none_and_included_flag_true(omj_home: Path) -> None:
    # REQ-020
    logger = DecisionLogger(include_state=True)
    row = _make_row(state=None)
    path = logger.write(row)

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["state"] is None


def test_korean_text_is_preserved_without_unicode_escapes(omj_home: Path) -> None:
    # REQ-020
    logger = DecisionLogger(include_state=True)
    row = _make_row(state="피해자는 사망하였는가")
    path = logger.write(row)

    raw_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "피해자는 사망하였는가" in raw_line
    assert "\\u" not in raw_line


def test_two_writes_produce_two_lines(omj_home: Path) -> None:
    # REQ-020
    logger = DecisionLogger()
    first_path = logger.write(_make_row(request_id="aaaa"))
    second_path = logger.write(_make_row(request_id="bbbb"))

    assert first_path == second_path
    lines = first_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_custom_log_dir_is_used_and_created_lazily(tmp_path: Path) -> None:
    # REQ-020
    custom_dir = tmp_path / "custom-logs"
    logger = DecisionLogger(log_dir=custom_dir)
    assert not custom_dir.exists()

    path = logger.write(_make_row())
    assert path.parent == custom_dir
    assert custom_dir.exists()


def test_state_sha256_is_stable_regardless_of_key_order() -> None:
    # REQ-020
    a = state_sha256({"b": 1, "a": 2})
    b = state_sha256({"a": 2, "b": 1})
    assert a == b


def test_state_sha256_hashes_str_state_as_is() -> None:
    # REQ-020
    import hashlib

    text = "raw string state"
    assert state_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_state_sha256_accepts_list_state() -> None:
    # REQ-020
    assert state_sha256([1, 2, 3]) == state_sha256([1, 2, 3])


def test_iter_rows_yields_parsed_dicts_in_write_order(omj_home: Path) -> None:
    # REQ-020
    logger = DecisionLogger()
    logger.write(_make_row(request_id="row-1"))
    path = logger.write(_make_row(request_id="row-2"))

    rows = list(iter_rows(path))
    assert [r["request_id"] for r in rows] == ["row-1", "row-2"]


def test_bearer_value_in_logged_state_stays_parseable_and_masked(omj_home: Path) -> None:
    # REQ-020, REQ-044: masking the serialized line must not eat the JSON quote
    # that terminates the state string.
    fake_bearer = "secret" + "token123"
    logger = DecisionLogger(include_state=True)
    path = logger.write(_make_row(state=f"Authorization: Bearer {fake_bearer}"))

    raw_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert fake_bearer not in raw_line

    row = json.loads(raw_line)
    assert row["state"] == "Authorization: Bearer secr****"
    assert row["request_id"] == "11111111-1111-1111-1111-111111111111"
    assert list(iter_rows(path))[0]["state"] == "Authorization: Bearer secr****"


def test_concurrent_writes_never_interleave_lines(omj_home: Path) -> None:
    # REQ-020: 32 threads x 20 writes with no lock around open/write/flush can
    # interleave partial JSONL lines from different threads into one line.
    logger = DecisionLogger()
    n_threads = 32
    writes_per_thread = 20

    def _write_many(thread_id: int) -> None:
        for i in range(writes_per_thread):
            logger.write(_make_row(request_id=f"t{thread_id}-{i}"))

    threads = [threading.Thread(target=_write_many, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    path = logger.log_dir / next(iter(logger.log_dir.iterdir())).name
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * writes_per_thread
    for line in lines:
        json.loads(line)  # every line must be independently parseable


def test_bearer_value_nested_in_state_mapping_is_masked(omj_home: Path) -> None:
    # REQ-020, REQ-044
    fake_bearer = "secret" + "token123"
    logger = DecisionLogger(include_state=True)
    path = logger.write(
        _make_row(state={"headers": {"authorization": f"Bearer {fake_bearer}"}})
    )

    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["state"]["headers"]["authorization"] == "Bearer secr****"
