"""Training data build: MASSIVE row loading, example rendering, shuffle, dev
sampling, the test-leak guard, and DataStats.

# REQ-003
# REQ-005
# REQ-017
# REQ-020
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="training tests need the semif extra (uv sync --extra semif)")

from omj.bench.suites import massive as massive_suite  # noqa: E402
from omj.errors import ErrorCode, OmjError
from omj.train import data
from omj.train.recipe import DataSection, LoraSection, Recipe, TrainSection

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MASSIVE_MINI = FIXTURES / "massive_mini.tar.gz"
TINY_MODEL = "yujiepan/qwen2.5-tiny-random"


class RecordingDownloader:
    """Downloader stub that serves fixed bytes, matching test_suite_loaders.py."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.payload


@pytest.fixture()
def massive_downloader() -> RecordingDownloader:
    return RecordingDownloader(MASSIVE_MINI.read_bytes())


@pytest.fixture()
def massive_hash_ok(monkeypatch: pytest.MonkeyPatch) -> str:
    digest = hashlib.sha256(MASSIVE_MINI.read_bytes()).hexdigest()
    monkeypatch.setattr(massive_suite, "MASSIVE_SHA256", digest)
    return digest


@pytest.fixture()
def criteria(massive_hash_ok: str) -> dict:
    # Reads the real committed suites/massive-criteria.ko.json; independent of the
    # mini tarball, but grouped with it since both back a build_examples() call.
    return data.load_criteria("ko-KR")


@functools.lru_cache(maxsize=1)
def _tiny_tokenizer_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=TINY_MODEL)


@pytest.fixture()
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> Any:
    # The conftest fixtures block omj's own downloader; this fetches the tiny
    # tokenizer straight from the HF cache instead, so the flag is cleared.
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    try:
        path = _tiny_tokenizer_path()
    except Exception as exc:  # network-less environments cannot fetch the checkpoint
        pytest.skip(f"tiny tokenizer {TINY_MODEL} unavailable: {exc}")

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def _recipe(*, seed: int, shuffle_options: bool, max_seq_len: int = 4096, locale: str = "ko-KR") -> Recipe:
    return Recipe(
        name="t",
        base_model=TINY_MODEL,
        data=DataSection(
            locale=locale, questions=["scenario", "intent"], shuffle_options=shuffle_options, dev_sample=1
        ),
        lora=LoraSection(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        train=TrainSection(
            lr=1e-3,
            epochs=1,
            batch_size=2,
            grad_accum=1,
            max_seq_len=max_seq_len,
            label_smoothing=0.0,
            seed=seed,
            eval_every_steps=2,
        ),
    )


# ----------------------------------------------------------------- row load ---


def test_load_massive_rows_splits_by_partition_for_both_locales(
    tmp_path: Path, massive_downloader: RecordingDownloader, massive_hash_ok: str
) -> None:
    # REQ-003
    for locale in ("ko-KR", "en-US"):
        train_rows = data.load_massive_rows(locale, "train", cache_dir=tmp_path, downloader=massive_downloader)
        test_rows = data.load_massive_rows(locale, "test", cache_dir=tmp_path, downloader=massive_downloader)
        dev_rows = data.load_massive_rows(locale, "dev", cache_dir=tmp_path, downloader=massive_downloader)

        assert [row["id"] for row in train_rows] == ["7"]
        assert len(test_rows) == 6
        assert dev_rows == []
        assert all(row["partition"] == "train" for row in train_rows)
        assert all(row["partition"] == "test" for row in test_rows)


def test_load_massive_rows_rejects_an_unknown_locale(tmp_path: Path) -> None:
    # REQ-003
    with pytest.raises(OmjError) as excinfo:
        data.load_massive_rows("fr-FR", "train", cache_dir=tmp_path, downloader=lambda url: b"")
    assert excinfo.value.code is ErrorCode.E_SCHEMA


# --------------------------------------------------------------- criteria -----


def test_load_criteria_covers_every_scenario_and_intent_label(massive_hash_ok: str) -> None:
    # REQ-003
    loaded = data.load_criteria("ko-KR")
    assert set(loaded["scenario"]) == set(massive_suite.SCENARIOS)
    assert set(loaded["intent"]) == set(massive_suite.INTENTS)


def test_question_dict_carries_the_locale_instructions_and_raw_criteria(massive_hash_ok: str) -> None:
    # REQ-003
    loaded = data.load_criteria("ko-KR")
    question = data.question_dict("scenario", loaded, "ko-KR")
    assert question["type"] == "choice"
    assert question["instructions"] == massive_suite.INSTRUCTIONS["ko"]["scenario"]
    assert question["criteria"] == loaded["scenario"]


# ------------------------------------------------------------------ sample_dev ---


def test_sample_dev_is_deterministic_regardless_of_input_order() -> None:
    # REQ-017
    rows = [{"id": str(i)} for i in range(50)]
    shuffled = list(reversed(rows))

    first = data.sample_dev(rows, n=10, seed=1)
    second = data.sample_dev(shuffled, n=10, seed=1)
    assert first == second
    assert len(first) == 10


def test_sample_dev_changes_with_the_seed_and_caps_at_row_count() -> None:
    # REQ-017
    rows = [{"id": str(i)} for i in range(20)]
    assert data.sample_dev(rows, n=5, seed=1) != data.sample_dev(rows, n=5, seed=2)
    assert len(data.sample_dev(rows, n=1000, seed=1)) == 20


# ----------------------------------------------------------------- check_no_leak ---


def test_check_no_leak_passes_when_there_is_no_overlap(tmp_path: Path) -> None:
    # REQ-003
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# comment\n1\n2\n3\n", encoding="utf-8")
    assert data.check_no_leak([10, 11], ids_file=ids_file) == 0


def test_check_no_leak_raises_e_config_mentioning_the_overlap_count(tmp_path: Path) -> None:
    # REQ-003
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("7\n", encoding="utf-8")  # the mini tarball's one train row

    with pytest.raises(OmjError) as excinfo:
        data.check_no_leak([7, 99], ids_file=ids_file)
    assert excinfo.value.code is ErrorCode.E_CONFIG
    assert "1" in excinfo.value.message
    assert "7" in excinfo.value.message


def test_check_no_leak_non_strict_returns_the_overlap_count(tmp_path: Path) -> None:
    # REQ-020: the count is what DataStats.test_overlap records, so it must be real.
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("7\n8\n", encoding="utf-8")

    assert data.check_no_leak([7, 99], ids_file=ids_file, strict=False) == 1
    assert data.check_no_leak([7, 8, 99], ids_file=ids_file, strict=False) == 2
    assert data.check_no_leak([99], ids_file=ids_file, strict=False) == 0


# ----------------------------------------------------------------- build_examples ---


def test_build_examples_count_equals_rows_times_questions(
    tmp_path: Path,
    massive_downloader: RecordingDownloader,
    massive_hash_ok: str,
    criteria: dict,
    tokenizer: Any,
) -> None:
    # REQ-003
    rows = data.load_massive_rows("ko-KR", "train", cache_dir=tmp_path, downloader=massive_downloader)
    recipe = _recipe(seed=1, shuffle_options=False)

    examples, stats = data.build_examples(rows, recipe, tokenizer, criteria, split="train")

    assert len(examples) == len(rows) * len(recipe.data.questions)
    assert stats.n_rows == len(rows)
    assert stats.n_examples == len(examples)
    assert {example.qid for example in examples} == set(recipe.data.questions)


@pytest.mark.parametrize("shuffle_options", [False, True])
def test_target_index_points_at_the_original_label(
    tmp_path: Path,
    massive_downloader: RecordingDownloader,
    massive_hash_ok: str,
    criteria: dict,
    tokenizer: Any,
    shuffle_options: bool,
) -> None:
    # REQ-005
    rows = data.load_massive_rows("ko-KR", "train", cache_dir=tmp_path, downloader=massive_downloader)
    by_id = {int(row["id"]): row for row in rows}
    recipe = _recipe(seed=42, shuffle_options=shuffle_options)

    examples, _ = data.build_examples(rows, recipe, tokenizer, criteria, split="train")

    assert len(examples) > 0
    for example in examples:
        row = by_id[example.row_id]
        assert example.keys[example.target_index] == row[example.qid]
        assert len(example.label_token_ids) == len(example.keys)


def test_build_examples_is_deterministic_for_the_same_recipe(
    tmp_path: Path,
    massive_downloader: RecordingDownloader,
    massive_hash_ok: str,
    criteria: dict,
    tokenizer: Any,
) -> None:
    # REQ-017
    rows = data.load_massive_rows("ko-KR", "train", cache_dir=tmp_path, downloader=massive_downloader)
    recipe = _recipe(seed=7, shuffle_options=True)

    first, first_stats = data.build_examples(rows, recipe, tokenizer, criteria, split="train")
    second, second_stats = data.build_examples(rows, recipe, tokenizer, criteria, split="train")

    assert first == second
    assert first_stats == second_stats


def test_shuffle_changes_option_order_for_a_different_seed(
    tmp_path: Path,
    massive_downloader: RecordingDownloader,
    massive_hash_ok: str,
    criteria: dict,
    tokenizer: Any,
) -> None:
    # REQ-005
    rows = data.load_massive_rows("ko-KR", "train", cache_dir=tmp_path, downloader=massive_downloader)
    recipe_a = _recipe(seed=1, shuffle_options=True)
    recipe_b = _recipe(seed=2, shuffle_options=True)

    examples_a, _ = data.build_examples(rows, recipe_a, tokenizer, criteria, split="train")
    examples_b, _ = data.build_examples(rows, recipe_b, tokenizer, criteria, split="train")

    by_id_a = {example.id: example for example in examples_a}
    by_id_b = {example.id: example for example in examples_b}
    assert set(by_id_a) == set(by_id_b)
    assert any(by_id_a[example_id].keys != by_id_b[example_id].keys for example_id in by_id_a)


def test_no_shuffle_keeps_the_authored_criteria_order(
    tmp_path: Path,
    massive_downloader: RecordingDownloader,
    massive_hash_ok: str,
    criteria: dict,
    tokenizer: Any,
) -> None:
    # REQ-005
    rows = data.load_massive_rows("ko-KR", "train", cache_dir=tmp_path, downloader=massive_downloader)
    recipe = _recipe(seed=1, shuffle_options=False)

    examples, _ = data.build_examples(rows, recipe, tokenizer, criteria, split="train")

    for example in examples:
        assert example.keys == list(criteria[example.qid])


def test_prompt_matches_render_prompt_text_with_for_the_same_option_order(
    tmp_path: Path,
    massive_downloader: RecordingDownloader,
    massive_hash_ok: str,
    criteria: dict,
    tokenizer: Any,
) -> None:
    # REQ-005
    from omj.backends.semif import render_prompt_text_with

    rows = data.load_massive_rows("ko-KR", "train", cache_dir=tmp_path, downloader=massive_downloader)
    recipe = _recipe(seed=1, shuffle_options=False)

    examples, _ = data.build_examples(rows, recipe, tokenizer, criteria, split="train")
    row = rows[0]

    for example in examples:
        question = data.question_dict(example.qid, criteria, recipe.data.locale)
        expected = render_prompt_text_with(tokenizer, row["utt"], question, option_order=None)
        expected_ids = tokenizer(expected.text, add_special_tokens=False)["input_ids"]

        assert example.input_ids == expected_ids
        assert example.keys == expected.keys
        assert example.label_token_ids == expected.label_token_ids


def test_examples_over_max_seq_len_are_dropped_and_counted(
    tmp_path: Path,
    massive_downloader: RecordingDownloader,
    massive_hash_ok: str,
    criteria: dict,
    tokenizer: Any,
) -> None:
    # REQ-003
    rows = data.load_massive_rows("ko-KR", "train", cache_dir=tmp_path, downloader=massive_downloader)
    recipe = _recipe(seed=1, shuffle_options=False, max_seq_len=64)  # 64 is the recipe schema's floor

    examples, stats = data.build_examples(rows, recipe, tokenizer, criteria, split="train")

    total_pairs = len(rows) * len(recipe.data.questions)
    assert stats.n_dropped_too_long > 0
    assert len(examples) + stats.n_dropped_too_long == total_pairs
    assert stats.n_examples == len(examples)
    assert all(example.n_tokens <= recipe.train.max_seq_len for example in examples)


def test_data_stats_fields_and_to_dict(
    tmp_path: Path,
    massive_downloader: RecordingDownloader,
    massive_hash_ok: str,
    criteria: dict,
    tokenizer: Any,
) -> None:
    # REQ-020
    rows = data.load_massive_rows("ko-KR", "train", cache_dir=tmp_path, downloader=massive_downloader)
    row = rows[0]
    recipe = _recipe(seed=123, shuffle_options=True)

    examples, stats = data.build_examples(rows, recipe, tokenizer, criteria, split="train")

    assert stats.n_rows == 1
    assert stats.n_examples == len(examples)
    assert stats.n_dropped_too_long == 0
    assert stats.test_overlap == 0
    assert stats.seed == recipe.train.seed
    assert stats.label_counts["scenario"] == {row["scenario"]: 1}
    assert stats.label_counts["intent"] == {row["intent"]: 1}

    as_dict = stats.to_dict()
    assert as_dict == {
        "n_rows": stats.n_rows,
        "n_examples": stats.n_examples,
        "n_dropped_too_long": stats.n_dropped_too_long,
        "label_counts": stats.label_counts,
        "test_overlap": stats.test_overlap,
        "seed": stats.seed,
    }


def test_build_examples_raises_when_a_row_id_is_a_pinned_test_id(
    tmp_path: Path,
    massive_downloader: RecordingDownloader,
    massive_hash_ok: str,
    criteria: dict,
    tokenizer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REQ-003
    rows = data.load_massive_rows("ko-KR", "train", cache_dir=tmp_path, downloader=massive_downloader)
    recipe = _recipe(seed=1, shuffle_options=False)

    leaking_ids_file = tmp_path / "leaking-ids.txt"
    leaking_ids_file.write_text("7\n", encoding="utf-8")
    # check_no_leak() resolves the default via omj.train.data's own imported
    # binding, not massive_suite's, so that is the name to patch here.
    monkeypatch.setattr(data, "DEFAULT_IDS_FILE", leaking_ids_file)

    with pytest.raises(OmjError) as excinfo:
        data.build_examples(rows, recipe, tokenizer, criteria, split="train")
    assert excinfo.value.code is ErrorCode.E_CONFIG
