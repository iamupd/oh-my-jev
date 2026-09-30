"""Mixed training sources (recipe #2): converters, sampling, caching, recipe schema,
and record-based example building. All inputs are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="training tests need the semif extra (uv sync --extra semif)")

from omj.errors import OmjError  # noqa: E402
from omj.train import data, sources
from omj.train.recipe import DataSection, LoraSection, MixSource, Recipe, TrainSection, load_recipe
from tests.unit.test_train_data import tokenizer  # noqa: F401  (fixture re-export)


def _recipe(max_seq_len: int = 1024) -> Recipe:
    return Recipe(
        name="mix-test",
        base_model="fake/base",
        data=DataSection(),
        lora=LoraSection(r=4, alpha=8, dropout=0.0, target_modules=["q_proj"]),
        train=TrainSection(lr=1e-4, epochs=1, batch_size=2, grad_accum=1, max_seq_len=max_seq_len,
                           label_smoothing=0.0, seed=7, eval_every_steps=1),
    )


def _tsv(*rows: tuple[str, str, str]) -> str:
    lines = ["id\tdocument\tlabel"] + ["\t".join(row) for row in rows]
    return "\n".join(lines) + "\n"


# --- converters ---------------------------------------------------------------


def test_ynat_maps_label_index_to_korean_topic() -> None:
    recs = sources.ynat_records([{"guid": "g1", "title": "코스피 상승", "label": 1}], "train")
    assert recs[0].expected == "경제"
    assert set(recs[0].question["criteria"]) == set(sources.YNAT_LABELS)


def test_nli_skips_unlabeled_rows_and_keeps_premise_hypothesis() -> None:
    rows = [
        {"guid": "a", "premise": "p", "hypothesis": "h", "label": 2},
        {"guid": "b", "premise": "p", "hypothesis": "h", "label": -1},
    ]
    recs = sources.nli_records(rows, "train")
    assert [r.expected for r in recs] == ["contradiction"]
    assert recs[0].state == {"전제": "p", "가설": "h"}


def test_nsmc_parses_tsv_and_drops_empty_documents() -> None:
    text = _tsv(("1", "최고의 영화", "1"), ("2", "", "0"), ("3", "지루함", "0"))
    recs = sources.nsmc_records(text, "train")
    assert [(r.id, r.expected) for r in recs] == [("nsmc:1", "yes"), ("nsmc:3", "no")]
    assert recs[0].question["type"] == "noul"


def test_banking77_dev_holdout_is_disjoint_and_deterministic() -> None:
    rows = [{"text": f"t{i}", "label": i % 77} for i in range(2000)]
    train = sources.banking77_records(rows, "train")
    dev = sources.banking77_records(rows, "dev")
    assert not {r.id for r in train} & {r.id for r in dev}
    assert len(train) + len(dev) == 2000
    assert 0.02 < len(dev) / 2000 < 0.08
    assert [r.id for r in dev] == [r.id for r in sources.banking77_records(rows, "dev")]


def _oj(**kw: Any) -> dict:
    row = {"id": "x1", "source": "workflow-controls-v1/invoice_processing", "kind": "choice",
           "question": "Which action?", "options": ["approve: pay it", "hold: wait"],
           "target": [0.0, 1.0], "state_json": json.dumps({"amount": 10})}
    row.update(kw)
    return row


def test_openjev_choice_splits_key_description_and_picks_argmax() -> None:
    [rec] = sources.openjev_records([_oj()], "train")
    assert rec.question["criteria"] == {"approve": "pay it", "hold": "wait"}
    assert rec.expected == "hold"
    assert rec.state == {"amount": 10}


def test_openjev_noul_and_score() -> None:
    noul = _oj(id="n", kind="noul", options=["no", "yes"], target=[0.0, 1.0])
    score = _oj(id="s", kind="score", options=["low", "mid", "high"], target=[0.0, 0.0, 1.0])
    recs = {r.id: r for r in sources.openjev_records([noul, score], "train")}
    assert recs["openjev:n"].expected == "yes"
    assert recs["openjev:s"].expected == "2"
    assert recs["openjev:s"].question["criteria"] == ["low", "mid", "high"]


def test_openjev_filters_games_and_soft_targets() -> None:
    rows = [_oj(id="g", source="snake-v1/x"), _oj(id="soft", target=[0.4, 0.6]),
            _oj(id="tie", target=[1.0, 1.0])]
    assert sources.openjev_records(rows, "train") == []


# --- sampling / caching / schema ---------------------------------------------


def _choice_and_noul(n_choice: int, n_noul: int) -> list[sources.DecisionRecord]:
    rows = [_oj(id=f"c{i}") for i in range(n_choice)]
    rows += [_oj(id=f"n{i}", kind="noul", options=["no", "yes"]) for i in range(n_noul)]
    return sources.openjev_records(rows, "train")


def test_balanced_sample_balances_kinds_and_is_order_independent() -> None:
    recs = _choice_and_noul(50, 50)
    picked = sources.balanced_sample(recs, 20, seed=3)
    kinds = [r.question["type"] for r in picked]
    assert kinds.count("choice") == kinds.count("noul") == 10
    again = sources.balanced_sample(list(reversed(recs)), 20, seed=3)
    assert [r.id for r in picked] == [r.id for r in again]


def test_balanced_sample_fills_from_spare_when_a_kind_is_short() -> None:
    assert len(sources.balanced_sample(_choice_and_noul(3, 30), 20, seed=1)) == 20


def test_fetch_cached_downloads_once(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return b"payload"

    for _ in range(2):
        assert sources.fetch_cached("http://x/y", "y.bin", cache_dir=tmp_path, fetch=fetch) == b"payload"
    assert calls == ["http://x/y"]


def test_unknown_source_raises() -> None:
    with pytest.raises(OmjError):
        sources.load_source("nope", "train")


def test_mix_recipe_validation() -> None:
    with pytest.raises(ValueError):
        DataSection(source="mix")
    with pytest.raises(ValueError):
        DataSection(source="massive", mix=[MixSource(name="nsmc", train_n=1)])
    with pytest.raises(ValueError):
        DataSection(source="mix", mix=[MixSource(name="nsmc", train_n=1), MixSource(name="nsmc", train_n=2)])


# --- record-based example building -------------------------------------------


def _mixed_records() -> list[sources.DecisionRecord]:
    ynat = [{"guid": f"g{i}", "title": f"제목 {i}", "label": i % 7} for i in range(6)]
    score = _oj(id="s", kind="score", options=["a", "b", "c"], target=[0.0, 1.0, 0.0])
    return (
        sources.ynat_records(ynat, "train")
        + sources.nsmc_records(_tsv(("1", "좋다", "1"), ("2", "별로", "0")), "train")
        + sources.openjev_records([score], "train")
    )


def test_build_examples_from_records_targets_match_expected(tokenizer: Any) -> None:  # noqa: F811
    recs = _mixed_records()
    examples, stats = data.build_examples_from_records(recs, _recipe(), tokenizer, split="train")
    by_id = {r.id: r for r in recs}
    assert stats.n_examples == len(recs) and stats.test_overlap == 0
    assert set(stats.label_counts) == {"klue-ynat", "nsmc", "openjev-business"}
    for ex in examples:
        assert ex.keys[ex.target_index] == by_id[ex.id].expected
        assert len(ex.label_token_ids) == len(ex.keys)


def test_build_examples_from_records_is_deterministic(tokenizer: Any) -> None:  # noqa: F811
    recs = _mixed_records()
    a, _ = data.build_examples_from_records(recs, _recipe(), tokenizer, split="train")
    b, _ = data.build_examples_from_records(list(reversed(recs)), _recipe(), tokenizer, split="train")
    assert [(e.id, e.keys, e.input_ids) for e in a] == [(e.id, e.keys, e.input_ids) for e in b]


def test_build_examples_from_records_drops_too_long(tokenizer: Any) -> None:  # noqa: F811
    long = sources.nsmc_records(_tsv(("9", "아주 긴 리뷰 " * 200, "1")), "train")
    examples, stats = data.build_examples_from_records(long, _recipe(max_seq_len=64), tokenizer, split="dev")
    assert examples == [] and stats.n_dropped_too_long == 1


def test_omj_config_env_overrides_config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from omj.config import config_path, load_config

    alt = tmp_path / "config-4b.toml"
    alt.write_text('[backend]\nname = "semif"\nmodel = "Qwen/Qwen3.5-4B"\nquant = "nf4"\n', encoding="utf-8")
    monkeypatch.setenv("OMJ_CONFIG", str(alt))
    assert config_path() == alt
    cfg = load_config()
    assert cfg.backend.model == "Qwen/Qwen3.5-4B" and cfg.backend.quant == "nf4"
