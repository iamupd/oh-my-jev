"""Tests for Hugging Face model downloading with retry logic. REQ-009."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omj.errors import ErrorCode, OmjError
from omj.models.download import download_model


# REQ-009: Fixed revision, exponential backoff, 3-attempt retries
class TestDownloadModel:
    def test_success_after_retries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failure twice, then success → returns path, sleeps [1, 2]."""
        call_count = 0
        sleep_calls = []

        def fake_downloader(repo_id: str, revision: str, local_dir: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"Download attempt {call_count} failed")
            return local_dir

        def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)

        result = download_model(
            "bert-base-uncased",
            "main",
            dest_root=tmp_path,
            downloader=fake_downloader,
            sleep=fake_sleep,
            attempts=3,
        )

        assert call_count == 3
        assert sleep_calls == [1.0, 2.0]
        assert result.exists()

    def test_all_retries_exhausted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three failures → E_DOWNLOAD with message containing model id."""
        def fake_downloader(repo_id: str, revision: str, local_dir: str) -> str:
            raise RuntimeError("Network error")

        monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)

        with pytest.raises(OmjError) as exc_info:
            download_model(
                "my-model/test",
                "v1.0",
                dest_root=tmp_path,
                downloader=fake_downloader,
                sleep=lambda x: None,
                attempts=3,
            )

        assert exc_info.value.code == ErrorCode.E_DOWNLOAD
        assert "my-model/test" in exc_info.value.message
        assert "v1.0" in exc_info.value.message

    def test_cached_marker_skips_download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cached marker → downloader not called."""
        downloader_called = False

        def fake_downloader(repo_id: str, revision: str, local_dir: str) -> str:
            nonlocal downloader_called
            downloader_called = True
            return local_dir

        monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)

        result = download_model(
            "cached-model",
            "main",
            dest_root=tmp_path,
            downloader=fake_downloader,
            sleep=lambda x: None,
            attempts=3,
        )

        assert result.exists()
        assert (result / ".omj-complete").exists()

        downloader_called = False

        result2 = download_model(
            "cached-model",
            "main",
            dest_root=tmp_path,
            downloader=fake_downloader,
            sleep=lambda x: None,
            attempts=3,
        )

        assert not downloader_called
        assert result == result2

    def test_network_disabled_without_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """OMJ_NO_NETWORK=1 without cache → E_DOWNLOAD."""
        monkeypatch.setenv("OMJ_NO_NETWORK", "1")

        with pytest.raises(OmjError) as exc_info:
            download_model(
                "uncached-model",
                "main",
                dest_root=tmp_path,
                downloader=lambda r, rev, d: d,
                sleep=lambda x: None,
                attempts=3,
            )

        assert exc_info.value.code == ErrorCode.E_DOWNLOAD
        assert "network disabled" in exc_info.value.message.lower()

    def test_revision_passed_through_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A named revision is passed through unchanged; empty string becomes "main"."""
        received_revisions = []

        def fake_downloader(repo_id: str, revision: str, local_dir: str) -> str:
            received_revisions.append(revision)
            return local_dir

        monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)

        download_model(
            "model",
            "v1.2.3",
            dest_root=tmp_path,
            downloader=fake_downloader,
            sleep=lambda x: None,
            attempts=1,
        )

        assert received_revisions == ["v1.2.3"]

        received_revisions.clear()
        download_model(
            "model",
            "",
            dest_root=tmp_path,
            downloader=fake_downloader,
            sleep=lambda x: None,
            attempts=1,
        )

        # REQ-009: "" means the default branch; snapshot_download 404s on "".
        assert received_revisions == ["main"]

    def test_model_id_slash_replaced_by_underscores(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """model_id with '/' → replaced by '__' in path."""
        def fake_downloader(repo_id: str, revision: str, local_dir: str) -> str:
            return local_dir

        monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)

        result = download_model(
            "org/model-name",
            "main",
            dest_root=tmp_path,
            downloader=fake_downloader,
            sleep=lambda x: None,
            attempts=1,
        )

        expected = tmp_path / "org__model-name" / "main"
        assert result == expected

    def test_empty_revision_reaches_downloader_as_main(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-009: an empty revision must be normalized to 'main' for the downloader."""
        received: list[tuple[str, str, str]] = []

        def fake_downloader(repo_id: str, revision: str, local_dir: str) -> str:
            received.append((repo_id, revision, local_dir))
            return local_dir

        monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)

        dest = download_model(
            "org/model-name",
            "",
            dest_root=tmp_path,
            downloader=fake_downloader,
            sleep=lambda x: None,
            attempts=1,
        )

        assert received == [("org/model-name", "main", str(dest))]
        assert dest == tmp_path / "org__model-name" / "main"


class TestOfflineMode:
    """REQ-009: --no-download must also stop an implicit backend download."""

    def test_offline_without_cache_raises_e_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
        called = False

        def fake_downloader(repo_id: str, revision: str, local_dir: str) -> str:
            nonlocal called
            called = True
            return local_dir

        with pytest.raises(OmjError) as exc_info:
            download_model(
                "org/model-name",
                "main",
                dest_root=tmp_path,
                downloader=fake_downloader,
                sleep=lambda x: None,
                attempts=1,
                offline=True,
            )

        assert exc_info.value.code == ErrorCode.E_DOWNLOAD
        assert "not cached" in exc_info.value.message
        assert called is False

    def test_offline_with_cache_returns_cached_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
        cached = tmp_path / "org__model-name" / "main"
        cached.mkdir(parents=True)
        (cached / ".omj-complete").write_text("")

        def fail_downloader(repo_id: str, revision: str, local_dir: str) -> str:
            raise AssertionError("downloader must not run in offline mode")

        result = download_model(
            "org/model-name",
            "main",
            dest_root=tmp_path,
            downloader=fail_downloader,
            sleep=lambda x: None,
            attempts=1,
            offline=True,
        )

        assert result == cached


def test_download_model_explains_hf_hub_offline(tmp_path, monkeypatch) -> None:
    import pytest

    from omj.errors import ErrorCode, OmjError
    from omj.models.download import download_model

    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = []
    with pytest.raises(OmjError) as err:
        download_model("Qwen/Qwen3.5-0.8B", "", dest_root=tmp_path, downloader=lambda *a: calls.append(a))
    assert err.value.code == ErrorCode.E_DOWNLOAD and calls == []
    assert "HF_HUB_OFFLINE" in err.value.message and "Remove-Item Env:HF_HUB_OFFLINE" in err.value.hint
