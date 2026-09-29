"""Root logger bootstrap: one stderr handler, redaction wired on both the logger and it."""

from __future__ import annotations

import logging
import sys

from omj.redaction import RedactingFilter

_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured

    root = logging.getLogger()
    root.setLevel(level)

    if _configured:
        return

    redacting_filter = RedactingFilter()
    root.addFilter(redacting_filter)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(redacting_filter)
    root.addHandler(handler)

    _configured = True
