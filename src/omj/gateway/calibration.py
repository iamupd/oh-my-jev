"""Calibration file loading and temperature application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from omj.errors import ErrorCode, OmjError


@dataclass(frozen=True)
class Calibration:
    version: int
    backend: str
    model: str
    temperatures: dict[str, float]
    fitted_on: dict
    ece_before: float | None
    ece_after: float | None
    created_at: str


def load_calibration(path: str | Path) -> Calibration:
    """Load calibration from JSON file.

    Raises OmjError(E_CONFIG) if file is malformed, missing, or has invalid temperature.
    Missing temperature keys default to 1.0; temperature ≤ 0 is invalid.
    """
    try:
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
    except FileNotFoundError as e:
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"calibration: file not found at {path}",
        ) from e
    except json.JSONDecodeError as e:
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"calibration: invalid JSON in {path}",
        ) from e
    except Exception as e:
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"calibration: failed to read {path}",
        ) from e

    try:
        temperatures = data.get("temperatures", {})
        if not isinstance(temperatures, dict):
            raise ValueError("calibration: temperatures must be a dict")

        temperatures_filled = {
            "noul": temperatures.get("noul", 1.0),
            "choice": temperatures.get("choice", 1.0),
            "score": temperatures.get("score", 1.0),
        }

        for kind, temp in temperatures_filled.items():
            if not isinstance(temp, (int, float)) or temp <= 0:
                raise ValueError(
                    f"calibration: temperature '{kind}' must be a positive number, got {temp}"
                )

        cal = Calibration(
            version=data["version"],
            backend=data["backend"],
            model=data["model"],
            temperatures=temperatures_filled,
            fitted_on=data["fitted_on"],
            ece_before=data.get("ece_before"),
            ece_after=data.get("ece_after"),
            created_at=data["created_at"],
        )
        return cal
    except ValueError as e:
        raise OmjError(ErrorCode.E_CONFIG, str(e)) from e
    except KeyError as e:
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"calibration: missing required field {e}",
        ) from e
    except Exception as e:
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"calibration: {str(e)}",
        ) from e


def temperature_for(cal: Calibration | None, kind: str) -> float:
    """Get temperature for a question kind.

    Returns the temperature from calibration if available, else 1.0.
    kind should be one of: "noul", "choice", "score".
    """
    if cal is None:
        return 1.0
    return cal.temperatures.get(kind, 1.0)


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    """Apply temperature scaling to logits: divide each by T.

    Preserves argmax (dividing by a positive scalar is monotone).
    """
    return [logit / temperature for logit in logits]


def save_calibration(cal: Calibration, path: str | Path) -> None:
    """Save calibration to JSON file with indent 2."""
    file_path = Path(path)
    data = {
        "version": cal.version,
        "backend": cal.backend,
        "model": cal.model,
        "temperatures": cal.temperatures,
        "fitted_on": cal.fitted_on,
        "ece_before": cal.ece_before,
        "ece_after": cal.ece_after,
        "created_at": cal.created_at,
    }
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
