"""MASSIVE -> DecisionItem conversion and the authored criteria files. # REQ-034"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from omj.bench.suites import load_suite, massive
from omj.bench.suites.local import SUITES_DIR
from omj.errors import ErrorCode, OmjError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MASSIVE_MINI = FIXTURES / "massive_mini.tar.gz"

MINI_TEST_IDS = [1, 2, 3, 4, 5, 6]
EXPECTED_LABELS = {
    1: ("alarm", "alarm_set"),
    2: ("alarm", "alarm_remove"),
    3: ("weather", "weather_query"),
    4: ("weather", "weather_query"),
    5: ("calendar", "calendar_set"),
    6: ("calendar", "calendar_query"),
}
MINI_UTTS = {
    "ko": {1: "내일 아침 일곱 시에 깨워 줘", 3: "오늘 서울에 비 와"},
    "en": {1: "wake me up at seven tomorrow", 3: "will it rain in seoul today"},
}


class RecordingDownloader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.payload


@pytest.fixture()
def mini_ids_file(tmp_path: Path) -> Path:
    path = tmp_path / "mini-ids.txt"
    path.write_text(
        "# source=amazon-massive-dataset-1.1 test split ko-KR/en-US\n"
        "# seed=20260922 strata=scenario n=6\n" + "".join(f"{i}\n" for i in MINI_TEST_IDS),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def mini_downloader(monkeypatch: pytest.MonkeyPatch) -> RecordingDownloader:
    payload = MASSIVE_MINI.read_bytes()
    monkeypatch.setattr(massive, "MASSIVE_SHA256", hashlib.sha256(payload).hexdigest())
    return RecordingDownloader(payload)


def _load(locale: str, tmp_path: Path, downloader: RecordingDownloader, ids_file: Path):
    return massive.load_massive(locale, cache_dir=tmp_path, downloader=downloader, ids_file=ids_file)


# ------------------------------------------------------------- criteria json ---


def test_criteria_files_hold_exactly_the_label_sets() -> None:
    # REQ-034
    for locale in ("ko", "en"):
        path = SUITES_DIR / f"massive-criteria.{locale}.json"
        assert path.is_file(), path
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"scenario", "intent"}
        assert list(data["scenario"]) == list(massive.SCENARIOS)
        assert list(data["intent"]) == list(massive.INTENTS)
        for group in ("scenario", "intent"):
            for key, description in data[group].items():
                assert isinstance(description, str)
                assert description.strip()
                assert "\n" not in description
                assert description.strip() != key


def test_criteria_descriptions_differ_between_the_two_languages() -> None:
    # REQ-034
    ko = json.loads((SUITES_DIR / "massive-criteria.ko.json").read_text(encoding="utf-8"))
    en = json.loads((SUITES_DIR / "massive-criteria.en.json").read_text(encoding="utf-8"))
    for group in ("scenario", "intent"):
        for key in ko[group]:
            assert ko[group][key] != en[group][key]


def test_committed_ids_file_holds_three_hundred_stratified_ids() -> None:
    # REQ-034
    path = SUITES_DIR / "massive-ids.txt"
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [line for line in lines if line.startswith("#")]
    assert any("amazon-massive-dataset-1.1" in line for line in header)
    assert any("seed=20260922" in line and "strata=scenario" in line for line in header)

    ids = [int(line) for line in lines if line.strip() and not line.startswith("#")]
    assert len(ids) == 300
    assert len(set(ids)) == 300
    assert ids == sorted(ids)


# ---------------------------------------------------------------- conversion ---


@pytest.mark.parametrize("locale", ["ko", "en"])
def test_conversion_produces_two_choice_questions_with_fixed_values(
    locale: str, tmp_path: Path, mini_downloader: RecordingDownloader, mini_ids_file: Path
) -> None:
    # REQ-034
    items = _load(locale, tmp_path, mini_downloader, mini_ids_file)
    assert len(items) == 6
    assert [item.pair_id for item in items] == [str(i) for i in MINI_TEST_IDS]

    for item in items:
        row_id = int(item.pair_id)
        scenario, intent = EXPECTED_LABELS[row_id]

        assert item.suite == f"massive-{locale}"
        assert item.source == "AmazonScience/massive@1.1"
        assert item.license == "CC-BY-4.0"
        assert item.tags == ["massive", locale, scenario]
        assert item.expected == {"scenario": scenario, "intent": intent}
        assert item.expected_shape is None

        assert list(item.questions) == ["scenario", "intent"]
        assert item.questions["scenario"]["type"] == "choice"
        assert item.questions["intent"]["type"] == "choice"
        assert list(item.questions["scenario"]["criteria"]) == list(massive.SCENARIOS)
        assert list(item.questions["intent"]["criteria"]) == list(massive.INTENTS)

    assert {item.questions["scenario"]["instructions"] for item in items} == {
        "이 발화가 속한 분야는?" if locale == "ko" else "Which domain does this utterance belong to?"
    }
    assert {item.questions["intent"]["instructions"] for item in items} == {
        "사용자의 의도는?" if locale == "ko" else "What is the user's intent?"
    }


@pytest.mark.parametrize("locale", ["ko", "en"])
def test_state_is_the_raw_utterance_and_train_rows_are_dropped(
    locale: str, tmp_path: Path, mini_downloader: RecordingDownloader, mini_ids_file: Path
) -> None:
    # REQ-034
    items = {int(item.pair_id): item for item in _load(locale, tmp_path, mini_downloader, mini_ids_file)}
    for row_id, utt in MINI_UTTS[locale].items():
        assert items[row_id].state == utt
    # id 7 exists in the fixture but only in the train partition
    assert 7 not in items


def test_ko_and_en_items_are_paired_by_the_massive_row_id(
    tmp_path: Path, mini_downloader: RecordingDownloader, mini_ids_file: Path
) -> None:
    # REQ-034
    ko = _load("ko", tmp_path, mini_downloader, mini_ids_file)
    en = _load("en", tmp_path, mini_downloader, mini_ids_file)

    assert [item.pair_id for item in ko] == [item.pair_id for item in en]
    assert [item.expected for item in ko] == [item.expected for item in en]
    assert {item.id for item in ko}.isdisjoint({item.id for item in en})
    for ko_item, en_item in zip(ko, en, strict=True):
        assert ko_item.state != en_item.state


def test_unknown_locale_is_a_schema_error(tmp_path: Path) -> None:
    # REQ-034
    with pytest.raises(OmjError) as excinfo:
        massive.load_massive("de", cache_dir=tmp_path)
    assert excinfo.value.code is ErrorCode.E_SCHEMA


def test_load_suite_dispatches_the_two_massive_names(
    tmp_path: Path, mini_downloader: RecordingDownloader, monkeypatch: pytest.MonkeyPatch, mini_ids_file: Path
) -> None:
    # REQ-034
    monkeypatch.setattr(massive, "DEFAULT_IDS_FILE", mini_ids_file)
    for name, locale in (("massive-ko", "ko"), ("massive-en", "en")):
        items = load_suite(name, cache_dir=tmp_path, downloader=mini_downloader)
        assert len(items) == 6
        assert {item.suite for item in items} == {f"massive-{locale}"}
