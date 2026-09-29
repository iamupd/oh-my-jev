"""Shipped source must not carry an em dash; the house style uses ':' or parentheses.

# REQ-014
"""

from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "omj"
EM_DASH = "—"


def test_src_dir_is_where_we_think_it_is() -> None:
    # REQ-014: a mislocated root would make the scan below vacuously green.
    assert SRC_DIR.is_dir(), SRC_DIR
    assert (SRC_DIR / "errors.py").is_file()


def test_no_em_dash_anywhere_under_src_omj() -> None:
    # REQ-014
    offenders: list[str] = []
    scanned = 0
    for path in sorted(SRC_DIR.rglob("*.py")):
        scanned += 1
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if EM_DASH in line:
                offenders.append(f"{path.relative_to(SRC_DIR).as_posix()}:{number}: {line.strip()}")

    assert scanned > 10, f"only {scanned} source files scanned"
    assert offenders == [], "replace U+2014 with ':' or parentheses:\n" + "\n".join(offenders)
