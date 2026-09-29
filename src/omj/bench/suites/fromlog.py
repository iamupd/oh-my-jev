"""Load decision items from decision log JSONL files for re-evaluation without ground truth.

This module reads decision log files and converts them into benchmark items.
Since logs do not include ground truth answers, items are loaded with expected=None
and expected_shape="unknown", suitable only for underdetermined metrics.
"""

from __future__ import annotations

from pathlib import Path

from omj.bench.suites.base import DecisionItem
from omj.gateway.decision_log import iter_rows


def load_from_log(log_dir: Path | str) -> list[DecisionItem]:
    """Load decision items from decision log JSONL files.

    Iterates all decisions-*.jsonl files in the directory and converts each row
    with a questions field to a DecisionItem. Rows without questions are skipped.

    Args:
        log_dir: Directory containing decision-*.jsonl files.

    Returns:
        List of DecisionItems with suite="from-log", expected=None,
        expected_shape="unknown", and tags ["from-log", backend].
    """
    items: list[DecisionItem] = []
    log_path = Path(log_dir)

    # Find all decision log files
    for log_file in sorted(log_path.glob("decisions-*.jsonl")):
        for row in iter_rows(log_file):
            # Skip rows without questions
            if "questions" not in row:
                continue

            # Determine state: use state field if present, otherwise use state_sha256
            if "state" in row and row["state"] is not None:
                state = row["state"]
            else:
                state = {"state_sha256": row["state_sha256"]}

            item = DecisionItem(
                id=row["request_id"],
                suite="from-log",
                state=state,
                questions=row["questions"],
                expected=None,
                tags=["from-log", row["backend"]],
                expected_shape="unknown",
                source=f"decision-log:{log_file.name}",
            )

            items.append(item)

    return items


def load_from_log_with_stats(log_dir: Path | str) -> tuple[list[DecisionItem], dict[str, int]]:
    """Load decision items with statistics about skipped rows.

    Args:
        log_dir: Directory containing decision-*.jsonl files.

    Returns:
        (items, stats) where stats is {"rows": total_rows, "skipped": skipped_count}.
    """
    items: list[DecisionItem] = []
    log_path = Path(log_dir)
    total_rows = 0
    skipped_rows = 0

    # Find all decision log files
    for log_file in sorted(log_path.glob("decisions-*.jsonl")):
        for row in iter_rows(log_file):
            total_rows += 1

            # Skip rows without questions
            if "questions" not in row:
                skipped_rows += 1
                continue

            # Determine state: use state field if present, otherwise use state_sha256
            if "state" in row and row["state"] is not None:
                state = row["state"]
            else:
                state = {"state_sha256": row["state_sha256"]}

            item = DecisionItem(
                id=row["request_id"],
                suite="from-log",
                state=state,
                questions=row["questions"],
                expected=None,
                tags=["from-log", row["backend"]],
                expected_shape="unknown",
                source=f"decision-log:{log_file.name}",
            )

            items.append(item)

    return items, {"rows": total_rows, "skipped": skipped_rows}
