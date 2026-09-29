"""Terminal progress for `omj bench`: one bar per suite on stderr, off when stderr is not a terminal."""

from __future__ import annotations

import sys
from types import TracebackType
from typing import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class BenchProgress:
    """Context manager; `suite(name, total)` returns the per-item callback for run_suite."""

    def __init__(self, enabled: bool, console: Console | None = None) -> None:
        self.enabled = enabled
        self._console = console if console is not None else Console(stderr=True)
        self._progress: Progress | None = None

    def __enter__(self) -> "BenchProgress":
        if self.enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("errors {task.fields[errors]}"),
                TimeElapsedColumn(),
                TextColumn("eta"),
                TimeRemainingColumn(),
                console=self._console,
            )
            self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def note(self, message: str) -> None:
        """A one-line status message (e.g. while the backend loads)."""
        if self.enabled:
            self._console.print(message, highlight=False)

    def suite(self, name: str, total: int) -> Callable[[bool], None] | None:
        if self._progress is None:
            return None
        progress = self._progress
        task = progress.add_task(name, total=total, errors=0)
        errors = 0

        def advance(ok: bool) -> None:
            nonlocal errors
            if not ok:
                errors += 1
            progress.update(task, advance=1, errors=errors)

        return advance


def progress_enabled() -> bool:
    """Show progress only on an interactive stderr, so logs, pipes and CI stay clean."""
    return sys.stderr.isatty()
