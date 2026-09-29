"""Your own decisions in the suite format: training source, recipe rules and bench suite."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from omj.errors import OmjError
from omj.train.recipe import load_recipe
from omj.train.sources import jsonl_records

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "support-decisions.jsonl"


def _recipe(tmp_path: Path, mix: str) -> Path:
    path = tmp_path / "mine.toml"
    path.write_text(f'''base_model = "Qwen/Qwen3.5-0.8B"
[data]
source = "mix"
{mix}
[lora]
r = 4
alpha = 8
dropout = 0.0
target_modules = ["q_proj"]
[train]
lr = 1e-4
epochs = 1
batch_size = 1
grad_accum = 1
max_seq_len = 256
label_smoothing = 0.0
seed = 1
eval_every_steps = 10
''', encoding="utf-8")
    return path


def test_jsonl_records_split_by_item_and_keep_labels() -> None:
    train, dev = jsonl_records(EXAMPLE, "train"), jsonl_records(EXAMPLE, "dev")
    items = lambda recs: {r.id.split(":")[2] for r in recs}  # noqa: E731
    assert train and dev and not (items(train) & items(dev))
    assert len(items(train) | items(dev)) == 40 and len(train) + len(dev) == 120
    rec = next(r for r in train if r.id.endswith(":urgency"))
    assert rec.question["type"] == "score" and rec.expected in {"0", "1", "2", "3"}


def test_jsonl_rows_are_validated_like_suites(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}},
                               "expected": {"q": "maybe"}}) + "\n", encoding="utf-8")
    with pytest.raises(OmjError) as err:
        jsonl_records(bad, "train")
    assert "line 1" in err.value.message and "maybe" in err.value.message


def test_recipe_resolves_the_jsonl_path_next_to_the_recipe(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    recipe = load_recipe(_recipe(tmp_path, '[[data.mix]]\nname = "jsonl"\npath = "data.jsonl"'))
    entry = recipe.data.mix[0]
    assert Path(entry.path) == (tmp_path / "data.jsonl").resolve() and entry.train_n is None


@pytest.mark.parametrize("mix", [
    '[[data.mix]]\nname = "jsonl"',                                   # no path
    '[[data.mix]]\nname = "banking77"\ntrain_n = 10\npath = "x.jsonl"',  # path on a public source
    '[[data.mix]]\nname = "banking77"',                                # train_n required outside jsonl
])
def test_recipe_rejects_bad_jsonl_entries(tmp_path: Path, mix: str) -> None:
    with pytest.raises(OmjError):
        load_recipe(_recipe(tmp_path, mix))


def test_load_mix_records_uses_every_jsonl_record_and_warns_on_benchmark_overlap(tmp_path: Path, capsys) -> None:
    from omj.bench.suites import load_suite
    from omj.cli.train_cmd import load_mix_records

    data = tmp_path / "data.jsonl"
    lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
    leaked = load_suite("omj-holdout")[0]
    row = json.loads(lines[0]); row["id"] = "leak"; row["state"] = leaked.state
    data.write_text("\n".join(lines + [json.dumps(row)]) + "\n", encoding="utf-8")
    recipe = load_recipe(_recipe(tmp_path, '[[data.mix]]\nname = "jsonl"\npath = "data.jsonl"\ndev_n = 5'))

    train, dev = load_mix_records(recipe)

    assert len(train) == len(jsonl_records(data, "train")) and len(dev) <= 5
    assert "omj-holdout" in capsys.readouterr().err


def test_bench_runs_a_jsonl_suite_under_its_file_name(tmp_path: Path) -> None:
    from omj.cli.bench_cmd import BenchOptions, run_bench

    report = run_bench(BenchOptions(backend="mock", suites=[str(EXAMPLE)], out=tmp_path / "run"), out=io.StringIO())
    assert list(report["suites"]) == ["support-decisions"] and report["suites"]["support-decisions"]["n"] == 120
