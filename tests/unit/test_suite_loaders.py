"""External suite loaders: jevbench raw download and MASSIVE tarball. # REQ-033"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from omj.bench.suites import base, jevbench, massive
from omj.bench.suites.base import Downloader, http_fetch
from omj.errors import ErrorCode, OmjError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
JEV_MINI = FIXTURES / "jevbench_mini.jsonl"
MASSIVE_MINI = FIXTURES / "massive_mini.tar.gz"


class RecordingDownloader:
    """Downloader stub that serves fixed bytes and records every URL it is given."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.payload


@pytest.fixture()
def jev_downloader() -> RecordingDownloader:
    return RecordingDownloader(JEV_MINI.read_bytes())


@pytest.fixture()
def massive_downloader() -> RecordingDownloader:
    return RecordingDownloader(MASSIVE_MINI.read_bytes())


@pytest.fixture()
def massive_hash_ok(monkeypatch: pytest.MonkeyPatch) -> str:
    digest = hashlib.sha256(MASSIVE_MINI.read_bytes()).hexdigest()
    monkeypatch.setattr(massive, "MASSIVE_SHA256", digest)
    return digest


# --------------------------------------------------------------- downloader ---


def test_http_fetch_refuses_when_network_is_disabled() -> None:
    # REQ-033
    with pytest.raises(OmjError) as excinfo:
        http_fetch("https://example.invalid/whatever.jsonl")
    assert excinfo.value.code is ErrorCode.E_DOWNLOAD
    assert "OMJ_NO_NETWORK" in excinfo.value.message


def test_default_downloader_is_typed_as_a_url_to_bytes_callable() -> None:
    # REQ-033
    fetch: Downloader = http_fetch
    assert callable(fetch)


def test_cache_root_defaults_under_omj_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # REQ-033
    monkeypatch.setenv("OMJ_HOME", str(tmp_path / "home"))
    assert base.cache_root(None) == tmp_path / "home" / "cache"
    assert base.cache_root(tmp_path / "elsewhere") == tmp_path / "elsewhere"


# ----------------------------------------------------------------- jevbench ---


def test_jevbench_loader_refuses_without_an_injected_downloader(tmp_path: Path) -> None:
    # REQ-033
    with pytest.raises(OmjError) as excinfo:
        jevbench.load_jevbench_public(cache_dir=tmp_path)
    assert excinfo.value.code is ErrorCode.E_DOWNLOAD
    assert not list(tmp_path.rglob("*.jsonl"))


def test_jevbench_pins_the_commit_in_every_url() -> None:
    # REQ-033
    assert jevbench.JEVBENCH_COMMIT == "75e6224ed8103bbc3485ca74820a2eaf7ce8abe0"
    assert list(jevbench.JEVBENCH_FILES) == ["original", "easy", "hard"]
    for name, url in jevbench.JEVBENCH_FILES.items():
        assert url == (
            "https://raw.githubusercontent.com/fstandhartinger/jevbench/"
            f"{jevbench.JEVBENCH_COMMIT}/datasets/public/{name}.jsonl"
        )


def test_jevbench_loader_converts_rows_and_tags_them_per_file(
    tmp_path: Path, jev_downloader: RecordingDownloader
) -> None:
    # REQ-033
    items = jevbench.load_jevbench_public(cache_dir=tmp_path, downloader=jev_downloader)

    # three files x three fixture rows
    assert len(jev_downloader.urls) == 3
    assert len(items) == 9
    assert {item.suite for item in items} == {"jevbench-public"}
    assert {item.source for item in items} == {f"fstandhartinger/jevbench@{jevbench.JEVBENCH_COMMIT}"}
    assert {item.license for item in items} == {"MIT"}

    file_tags = [t for item in items for t in item.tags if t in {"original", "easy", "hard"}]
    assert sorted(file_tags) == ["easy"] * 3 + ["hard"] * 3 + ["original"] * 3

    by_id = {(item.tags[0], item.id): item for item in items}
    hard_noul = by_id[("hard", "jev-mini-noul-1")]
    assert list(hard_noul.questions) == ["decision"]
    assert hard_noul.questions["decision"]["type"] == "noul"
    assert hard_noul.expected == {"decision": "yes"}
    assert "refund" in hard_noul.tags

    hard_score = by_id[("hard", "jev-mini-score-1")]
    assert hard_score.questions["decision"]["type"] == "score"
    assert hard_score.expected == {"decision": "3"}
    assert hard_score.questions["decision"]["criteria"][3].startswith("Total outage")


def test_jevbench_loader_preserves_state_and_question_verbatim(
    tmp_path: Path, jev_downloader: RecordingDownloader
) -> None:
    # REQ-033
    raw = {json.loads(line)["id"]: json.loads(line) for line in JEV_MINI.read_text(encoding="utf-8").splitlines()}
    for item in jevbench.load_jevbench_public(cache_dir=tmp_path, downloader=jev_downloader):
        source_row = raw[item.id]
        assert item.state == source_row["state"]
        assert item.questions["decision"] == source_row["question"]
        assert item.expected == {"decision": source_row["expected"]}


def test_jevbench_loader_writes_a_commit_scoped_cache_and_reuses_it(
    tmp_path: Path, jev_downloader: RecordingDownloader
) -> None:
    # REQ-033
    jevbench.load_jevbench_public(cache_dir=tmp_path, downloader=jev_downloader)
    cache_dir = tmp_path / "jevbench" / jevbench.JEVBENCH_COMMIT
    assert sorted(p.name for p in cache_dir.glob("*.jsonl")) == ["easy.jsonl", "hard.jsonl", "original.jsonl"]

    first_call_count = len(jev_downloader.urls)
    again = jevbench.load_jevbench_public(cache_dir=tmp_path, downloader=jev_downloader)
    assert len(jev_downloader.urls) == first_call_count
    assert len(again) == 9


# ------------------------------------------------------------------ massive ---


def test_massive_constants_are_pinned() -> None:
    # REQ-033
    assert massive.MASSIVE_URL == (
        "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
    )
    assert massive.MASSIVE_SHA256 == "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
    assert len(massive.SCENARIOS) == 18
    assert len(massive.INTENTS) == 60


def test_massive_download_refuses_without_an_injected_downloader(tmp_path: Path) -> None:
    # REQ-033
    with pytest.raises(OmjError) as excinfo:
        massive.download_massive(cache_dir=tmp_path)
    assert excinfo.value.code is ErrorCode.E_DOWNLOAD


def test_massive_download_rejects_a_sha256_mismatch(
    tmp_path: Path, massive_downloader: RecordingDownloader
) -> None:
    # REQ-033 -- the real pinned hash does not match the mini fixture
    with pytest.raises(OmjError) as excinfo:
        massive.download_massive(cache_dir=tmp_path, downloader=massive_downloader)
    assert excinfo.value.code is ErrorCode.E_DOWNLOAD
    assert "sha256 mismatch" in excinfo.value.message
    assert not (tmp_path / "massive" / "1.1" / "data").exists()


def test_massive_download_extracts_only_the_two_locale_files(
    tmp_path: Path, massive_downloader: RecordingDownloader, massive_hash_ok: str
) -> None:
    # REQ-033
    root = massive.download_massive(cache_dir=tmp_path, downloader=massive_downloader)

    assert massive_downloader.urls == [massive.MASSIVE_URL]
    assert root == tmp_path / "massive" / "1.1"
    assert sorted(p.name for p in (root / "data").iterdir()) == ["en-US.jsonl", "ko-KR.jsonl"]
    assert not (root / "LICENSE").exists()
    assert (root / ".complete").is_file()


def test_massive_download_skips_the_fetch_when_the_marker_exists(
    tmp_path: Path, massive_downloader: RecordingDownloader, massive_hash_ok: str
) -> None:
    # REQ-033
    massive.download_massive(cache_dir=tmp_path, downloader=massive_downloader)
    massive.download_massive(cache_dir=tmp_path, downloader=massive_downloader)
    assert massive_downloader.urls == [massive.MASSIVE_URL]


# ----------------------------------------------------------- stratified ids ---


def _synthetic_test_rows() -> list[dict]:
    rng = random.Random(7)
    rows: list[dict] = []
    # Reproduce the real shape: 2,974 test rows spread unevenly over the 18 scenarios.
    weights = [57 + 20 * i for i in range(18)]
    total = sum(weights)
    counts = [max(1, round(w * 2974 / total)) for w in weights]
    counts[-1] += 2974 - sum(counts)
    next_id = 0
    for scenario, count in zip(massive.SCENARIOS, counts, strict=True):
        for _ in range(count):
            next_id += 1
            rows.append({"id": str(next_id), "scenario": scenario, "partition": "test"})
    rng.shuffle(rows)
    return rows


def test_stratified_ids_is_deterministic_and_sums_to_n() -> None:
    # REQ-033
    rows = _synthetic_test_rows()
    assert len(rows) == 2974

    first = massive.stratified_ids(rows, n=300, seed=20260922)
    second = massive.stratified_ids(list(reversed(rows)), n=300, seed=20260922)

    assert first == second
    assert len(first) == 300
    assert len(set(first)) == 300
    assert first == sorted(first)
    assert all(isinstance(i, int) for i in first)


def test_stratified_ids_changes_with_the_seed() -> None:
    # REQ-033
    rows = _synthetic_test_rows()
    assert massive.stratified_ids(rows, n=300, seed=20260922) != massive.stratified_ids(rows, n=300, seed=1)


def test_stratified_ids_covers_every_scenario_proportionally() -> None:
    # REQ-033
    rows = _synthetic_test_rows()
    by_id = {int(row["id"]): row["scenario"] for row in rows}
    picked = massive.stratified_ids(rows, n=300, seed=20260922)

    per_scenario: dict[str, int] = {}
    for item_id in picked:
        per_scenario[by_id[item_id]] = per_scenario.get(by_id[item_id], 0) + 1
    assert set(per_scenario) == set(massive.SCENARIOS)

    totals: dict[str, int] = {}
    for row in rows:
        totals[row["scenario"]] = totals.get(row["scenario"], 0) + 1
    for scenario, quota in per_scenario.items():
        ideal = 300 * totals[scenario] / 2974
        assert abs(quota - ideal) < 1.0
