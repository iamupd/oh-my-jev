"""Repo-level artifacts: LICENSE and README disclaimer. # REQ-048"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_license_is_apache() -> None:
    # REQ-048
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.exists()
    lines = license_path.read_text(encoding="utf-8").splitlines()
    first_non_empty = next(line for line in lines if line.strip())
    assert "Apache License" in first_non_empty


def test_readme_has_typesafe_disclaimer() -> None:
    # REQ-048
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists()
    text = readme_path.read_text(encoding="utf-8")
    assert "Not affiliated with or endorsed by TypeSafe AI" in text
