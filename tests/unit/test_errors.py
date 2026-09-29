"""Error code registry, hints, formatting, and a source-wide code allowlist scan.
# REQ-008
"""

from __future__ import annotations

import re
from pathlib import Path

from omj.errors import HINTS, ErrorCode, OmjError, format_error

EXPECTED_CODES = {
    "E_NO_GPU",
    "E_SCHEMA",
    "E_DOWNLOAD",
    "E_CONFIG",
    "E_BACKEND",
    "E_AUTH",
    "E_NET",
}

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "omj"
CODE_PATTERN = re.compile(r"(?<![A-Za-z0-9_])E_[A-Z][A-Z_]*(?![A-Za-z0-9_])")


def test_error_code_set_matches_the_seven_approved_codes() -> None:
    # REQ-008
    assert {code.value for code in ErrorCode} == EXPECTED_CODES


def test_every_code_has_a_non_empty_hint() -> None:
    for code in ErrorCode:
        assert code in HINTS
        assert HINTS[code].strip()


def test_format_error_produces_code_message_hint() -> None:
    err = OmjError(ErrorCode.E_CONFIG, "bad field", hint="Fix it.")
    assert format_error(err) == "E_CONFIG: bad field. Fix it."


def test_omjerror_defaults_hint_from_table() -> None:
    err = OmjError(ErrorCode.E_NET, "upstream unreachable")
    assert err.hint == HINTS[ErrorCode.E_NET]
    assert err.exit_code == 1


def test_source_tree_only_references_approved_error_codes() -> None:
    # REQ-008: grep every source file for E_<CODE>-shaped tokens and require
    # each one to be a member of the approved seven-code table.
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in CODE_PATTERN.finditer(text):
            assert match.group(0) in EXPECTED_CODES, f"unknown code {match.group(0)!r} in {path}"


def test_code_pattern_ignores_identifiers_that_merely_contain_e_underscore() -> None:
    # REQ-008: regression for false positives on identifiers that embed "E_"
    # mid-token (e.g. COVERAGE_RISK_BUDGET) rather than starting with it.
    sample = (
        'MASSIVE_URL = 1\n'
        'COVERAGE_RISK_BUDGET = 0.05\n'
        'if TYPE_CHECKING:\n'
        '    pass\n'
        'headers["TYPESAFE_API_KEY"] = value\n'
    )
    assert CODE_PATTERN.findall(sample) == []
    assert CODE_PATTERN.findall("ErrorCode.E_CONFIG") == ["E_CONFIG"]
    assert CODE_PATTERN.findall("raise E_BOGUS") == ["E_BOGUS"]
