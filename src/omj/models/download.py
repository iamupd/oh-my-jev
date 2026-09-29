"""Hugging Face model downloader with exponential backoff and retry logic."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from omj.config import omj_home
from omj.errors import ErrorCode, OmjError

Downloader = Callable[[str, str, str], str]


def hf_snapshot_download(repo_id: str, revision: str, local_dir: str) -> str:
    """Default downloader using huggingface_hub.snapshot_download."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)


def download_model(
    model_id: str,
    revision: str,
    dest_root: Path | None = None,
    downloader: Downloader = hf_snapshot_download,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 3,
    offline: bool = False,
) -> Path:
    """Download a model from Hugging Face with exponential backoff retries.

    Args:
        model_id: Model identifier (e.g., "bert-base-uncased")
        revision: Git revision (e.g., "main", "v1.0"). An empty value means the
            default branch and is normalized to "main" for the downloader.
        dest_root: Destination root directory. Defaults to omj_home()/"models"
        downloader: Function to call for downloading. Signature: (repo_id, revision, local_dir) -> str
        sleep: Sleep function (default: time.sleep)
        attempts: Number of retry attempts (default: 3)
        offline: Use the cache only; never reach the network.

    Returns:
        Path to the downloaded model directory.

    Raises:
        OmjError: If the cache is missing while offline or OMJ_NO_NETWORK=1, or
            all retries are exhausted.
    """
    if dest_root is None:
        dest_root = omj_home() / "models"

    model_dir = model_id.replace("/", "__")
    # snapshot_download 404s on an empty revision; "" means the default branch.
    revision_ref = revision if revision else "main"
    dest = dest_root / model_dir / revision_ref

    marker_file = dest / ".omj-complete"
    if marker_file.exists():
        return dest

    if offline:
        raise OmjError(
            ErrorCode.E_DOWNLOAD,
            f"{model_id}@{revision_ref}: model not cached; run without --no-download",
        )

    if os.environ.get("OMJ_NO_NETWORK") == "1":
        raise OmjError(
            ErrorCode.E_DOWNLOAD,
            "network disabled (OMJ_NO_NETWORK=1)",
        )

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            downloader(model_id, revision_ref, str(dest))

            dest.mkdir(parents=True, exist_ok=True)
            marker_file.write_text("")
            return dest
        except Exception as e:
            last_error = e
            if attempt < attempts - 1:
                sleep_seconds = 2.0 ** attempt
                sleep(sleep_seconds)

    raise OmjError(
        ErrorCode.E_DOWNLOAD,
        f"{model_id}@{revision_ref}: {last_error}",
    )
