<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.png">
  <img src="docs/images/logo.png" alt="oh-my-jev logo" width="120">
</picture>

# oh-my-jev

**The easiest way to play with, benchmark and train your own Jev-style model.**

Try *System One* decision models in a web playground, measure accuracy and calibration against Jev, and fine-tune your own on a JSONL file. A decision model answers a typed question with a calibrated probability instead of free text.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Release](https://img.shields.io/badge/release-v0.1.1-2ea44f)](https://github.com/iamupd/oh-my-jev/releases)
[![TypeSafe compatible](https://img.shields.io/badge/API-TypeSafe%20%2Fv1%2Fsystemone-6f42c1)](#use-it-from-the-typesafe-sdk)

[Quick start](#quick-start) · [Commands](#commands) · [Suites](#benchmark-suites) · [Train](#train-your-own-decision-model) · [Limitations](#known-limitations) · [한국어](README.ko.md)

<img src="docs/images/playground.png" alt="Web playground: a fine-tuned 4B model and a base 2B model judge the same support ticket, with probability bars and a policy verdict" width="760">
<br><sub>Web playground: two models judge the same ticket, with probabilities and a policy verdict (<code>omj ui</code>)</sub>
<br><br>
<img src="docs/images/reports.png" alt="Reports page: a fine-tuned run compared with Jev 1.13, with differences and 95% intervals" width="760">
<br><sub>Reports page: every run, compared with Jev, with 95% intervals, per-tag weak spots and reliability (<code>omj bench --view</code>)</sub>
<br><br>
<img src="docs/images/bench.png" alt="omj bench terminal output: metrics, per-tag breakdown, reliability table and summary" width="760">
<br><sub>Terminal: the same results straight from <code>omj bench</code></sub>

</div>

## Why

Agents make many small calls: *which team gets this ticket, is this answer grounded, should a human look at it.* A generated answer gives you no threshold to act on. A decision model returns a probability for each option, so you can act automatically above 0.9 and escalate below it, provided the probabilities are honest.

oh-my-jev is the workbench for that loop: run a decision model behind a TypeSafe-compatible endpoint, measure accuracy **and** calibration, fine-tune your own, and compare it with a reference.

## Highlights

- 🚀 **One-pass readout.** Local models answer from the option-label logits of a single forward pass: no generation, one prefill per decision.
- 🎯 **Calibration, not just accuracy.** ECE, Brier, NLL, coverage at 5% risk, reliability tables, 95% confidence intervals and temperature fitting.
- 🔌 **Drop-in endpoint.** `omj serve` speaks TypeSafe's `/v1/systemone`, so the official SDK works by changing `base_url`.
- 🧪 **Train your own.** LoRA / QLoRA recipes with label-restricted cross-entropy and a Brier term; 4-bit loading for small GPUs.
- ⚖️ **Measured against Jev.** Every result shows its difference from Jev, marked `≈` when it is inside the 95% interval. Compare runs in the terminal or on the Reports page, and send one situation to several models in the web playground (English / 한국어).
- 🧰 **Works on first install.** A mock backend needs no GPU and no key, so every command runs before you download a model.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/iamupd/oh-my-jev.git
cd oh-my-jev
uv sync --extra dev
uv run omj init --yes                 # detects your GPU and keys, writes ~/.omj/config.toml
uv run omj bench --suite omj-smoke --view    # 30 items; --view opens the result in your browser
uv run omj serve                      # http://127.0.0.1:8799, runs until Ctrl+C
```

Then, in a second terminal:

```bash
curl -s http://127.0.0.1:8799/v1/systemone -H "Content-Type: application/json" -d '{"model": "jev-latest", "state": "Customer: I was charged twice.", "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payment problems", "shipping": "delivery problems"}}}}'
```

<details>
<summary>Windows PowerShell</summary>

`curl` is an alias of `Invoke-WebRequest` there; send the same request with:

```powershell
$body = '{"model": "jev-latest", "state": "Customer: I was charged twice.", "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payment problems", "shipping": "delivery problems"}}}}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8799/v1/systemone -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

</details>

<details>
<summary>How <code>omj init</code> picks a backend</summary>

- CUDA GPU with 6 to 24 GB and the `semif` extra installed: a local Qwen3.5 model (see [Local models](#local-models))
- Without the `semif` extra: the remote backend when `JEV_KEY` or `OPENROUTER_KEY` is set, otherwise the mock backend, plus a hint on enabling local models
- Pin one yourself: `uv run omj init --yes --backend mock` (or `typesafe`, `semif`)
- `omj bench` uses the configured backend unless you pass `--backend NAME`, `--endpoint URL`, `--model org/name` or `--adapter <dir>` (the last two run locally on semif)

</details>

## Commands

| Command | What you get |
|---|---|
| `omj init` | Hardware and key detection, a smoke test, and `~/.omj/config.toml` |
| `omj serve` | A TypeSafe-compatible `/v1/systemone` endpoint over a local model, a remote API or a mock |
| `omj bench` | Accuracy, calibration, per-type and per-tag breakdowns, a reliability table and the difference from Jev, saved as `report.md` and `report.json`; `--view` opens it on the Reports page |
| `omj compare` | Two or more reports side by side, with differences against the first |
| `omj train` | LoRA / QLoRA fine-tuning from a TOML recipe, on your own JSONL decisions and/or public datasets |
| `omj ui` | A local web playground for one or more models, with policy thresholds, plus a Reports page for every finished run |

Run each as `uv run omj …`, or activate `.venv` and call `omj …` directly.

| Backend | Runs | Needs |
|---|---|---|
| `semif` | open weights in-process, option-logit readout, optional LoRA adapter | CUDA GPU, `uv sync --extra semif` |
| `typesafe` | the hosted Jev API, directly or through OpenRouter | `JEV_KEY` or `OPENROUTER_KEY` |
| `kev` | a running [Kev](https://github.com/jaredpalmer/kev) server | the Kev server |
| `mock` | keyword overlap; a test double, not a model | nothing |

## Benchmark suites

Pick one or more with `--suite` (repeatable); the default is `omj-smoke`. A suite is run on the model from `omj init` unless you name one with `--model org/name`, `--adapter <dir>` or `--backend`.

| Suite | What it measures | Size | Jev values bundled | Data |
|---|---|---|---|---|
| `omj-smoke` | Quick check across routing, sentiment, safety and other small decisions | 30 | yes | bundled |
| `jevbench-public` | The public JevBench items: easy, standard and hard decisions over long policies, extraction, multi-hop and date arithmetic | 231 | yes | downloaded once from the JevBench repository (pinned commit) |
| `omj-holdout` | Out-of-distribution questions from BoolQ, ARC-Challenge, CommonsenseQA and SVAMP; use it to pick checkpoints | 300 | no | bundled |
| `massive-ko` / `massive-en` | Korean / English intent and scenario classification on a fixed MASSIVE 1.1 test sample; running both adds the KO-EN gap | 600 answers each | yes | MASSIVE 1.1 archive downloaded once (checksum verified) |
| `underdetermined` | Questions with no right answer; measures overconfidence instead of accuracy | 20 | yes | bundled |
| `order` | The choice questions of the first suite in the run, with options shuffled; measures how much the probabilities move | 5 shuffles per question | no | derived |
| `all` | Every suite above | | | |
| `path/to/file.jsonl` | Your own decisions in the same format (see [Train your own](#train-your-own-decision-model)); reported under the file name | any | no | yours |

```bash
uv run omj bench --model Qwen/Qwen3.5-2B --suite jevbench-public --suite omj-holdout --view   # the two most informative suites
uv run omj bench --model Qwen/Qwen3.5-2B --suite massive-ko --suite massive-en                 # adds the KO-EN gap and agreement
uv run omj bench --suite jevbench-public --suite order        # the model from omj init, plus option-order robustness
```

## Compare with Jev and other runs

Every `omj bench` result ends with a **vs Jev** table (accuracy, ECE and Brier differences), also written to `report.md` and shown on the Reports page. The Jev numbers come from, in order:

1. a finished local Jev run on the same suite,
2. a fresh Jev run on the same items when `JEV_KEY` or `OPENROUTER_KEY` is set (run once, saved to `~/.omj/runs/jev-reference-*` and reused),
3. Jev 1.13 values bundled with omj (JevBench public, MASSIVE ko/en, omj-smoke; measured by the omj authors on 2026-09-22, not official TypeSafe figures).

`≈` means Jev lies inside the run's 95% interval, so the difference is not significant; `▲` / `▼` mean better / worse. `--reference none` turns the table off and `--reference <report.json>` compares with another run instead.

To put finished runs side by side:

```bash
uv run omj compare jev/report.json run-a/report.json run-b/report.json --labels jev-1.13,model-a,model-b --out runs/cmp
```

<div align="center">
<img src="docs/images/compare.png" alt="omj compare output: metrics of three reports with coloured differences" width="760">
</div>

Differences are coloured by each metric's direction (green is better), and the best value in each row is bold. The numbers above were measured in September 2026 on the 231 public JevBench items. The Reports page does the same for the runs you tick.

## Local models

```bash
uv sync --extra semif --extra dev    # torch, transformers, peft (several GB)
uv run omj init --yes                # 6-24 GB CUDA GPU: downloads a Qwen3.5 base (2B is about 4.5 GB)
uv run omj serve
```

- `--no-download` writes the config without fetching weights.
- A model that would not fit the GPU in bf16 is loaded in 4-bit (nf4) automatically; `uv run omj init --quant nf4` forces it.

## Benchmark any Hugging Face model

Pass a model id with `--model`; omj downloads it on first use, builds the option prompt and reads the label-token probabilities, so an untrained model gets a zero-shot score to compare against. No config file needed.

```bash
uv run omj bench --model Qwen/Qwen3.5-0.8B --suite jevbench-public --suite omj-holdout --out runs/qwen-0.8b
uv run omj bench --model Qwen/Qwen3.5-2B   --suite jevbench-public --suite omj-holdout --out runs/qwen-2b
uv run omj compare runs/qwen-0.8b/report.json runs/qwen-2b/report.json --labels qwen-0.8b,qwen-2b --out runs/cmp
```

<details>
<summary>Which models work</summary>

- Any `AutoModelForCausalLM` checkpoint whose tokenizer encodes `" A"`, `" B"`, ... as single tokens. Models that need `trust_remote_code` are not loaded.
- The model is loaded in bf16, or in 4-bit when its size tag (for example `4B`) says it would not fit your GPU in bf16.
- A LoRA adapter repo works too: `--model your-org/your-lora-adapter` downloads the adapter and loads it on the base model it was trained on. `--adapter <dir>` does the same for a local adapter.
- Gated models (for example Llama or Gemma) need `hf auth login` and an accepted licence first.
- Models trained by other projects with their own prompt format (Kev, Open-Jev and similar) are best measured through their own server: `uv run omj bench --endpoint http://127.0.0.1:<port>`.

</details>

## Web playground

Send one situation and its questions to one or more models and see their probabilities side by side, then apply a policy threshold. Give each model with `--target`:

```bash
uv run omj ui                                                            # the model in config.toml (+ Jev when JEV_KEY is set)
uv run omj ui --target Qwen/Qwen3.5-0.8B --target Qwen/Qwen3.5-2B       # two Hugging Face models
uv run omj ui --target semif --target semif:~/.omj/adapters/<recipe>/<run_id>/best   # base model vs your adapter
uv run omj ui --target semif --target typesafe                            # your model vs the hosted Jev API
```

| `--target` | Runs |
|---|---|
| `org/model` | that Hugging Face model, downloaded on first use |
| `semif` | the model in `config.toml` |
| `semif:<adapter dir>` | a LoRA adapter on the base model it was trained on |
| `typesafe` / `kev` / `mock` | the hosted Jev API / a Kev server / the test double |
| `@<config.toml>` | the `[backend]` of another config file (advanced) |

Each target is labelled on screen from what it runs (`qwen3.5-0.8b`, `semif`, the adapter's recipe name); prefix `name=` to choose the label yourself, for example `--target baseline=Qwen/Qwen3.5-2B`.

- Opens on `http://127.0.0.1:8800`, in English; add `?lang=ko` or use the **EN / 한국어** switch for Korean. Four presets (memory promotion, code review, content policy, support routing) are built in.
- Every target loads its own model, so the GPU has to hold all of them at once.
- The **Reports** tab lists every finished run (`~/.omj/runs` and any `--out` folder): summary with 95% intervals, per-tag weak spots, a reliability chart, and side-by-side comparison of the runs you tick. `omj bench --view` opens the same page, reusing a running `omj ui` or starting a reports-only server (no models loaded) until Ctrl+C.

## Train your own decision model

Write your decisions as JSONL: **one JSON object per line**, one line per situation, in the same format the bench suites use. A two-line file (the full [`examples/support-decisions.jsonl`](examples/support-decisions.jsonl) has 40 lines and four teams):

```jsonl
{"id": "support-001", "tags": ["billing"], "state": {"channel": "email", "message": "I was charged twice this month."}, "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems"}}, "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"}, "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Can wait a week", "Within a few days", "Today", "Blocking right now"]}}, "expected": {"team": "billing", "refund": "yes", "urgency": "2"}}
{"id": "support-002", "tags": ["shipping"], "state": {"channel": "chat", "message": "Where is my parcel? It should have arrived on Friday."}, "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems"}}, "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"}, "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Can wait a week", "Within a few days", "Today", "Blocking right now"]}}, "expected": {"team": "shipping", "refund": "no", "urgency": "2"}}
```

| Field | Meaning |
|---|---|
| `id` | Unique per line |
| `state` | The situation: text, or any JSON object |
| `questions` | One or more questions. `choice`: `criteria` maps each option key to a description. `noul`: a yes/no question (optional `criteria` with `true` / `false` descriptions). `score`: `criteria` lists one description per level, from 0 up |
| `expected` | The right answer per question: an option key, `yes` / `no`, or the level number as a string. Questions without one are skipped in training and scored as unlabelled in bench |
| `tags` | Optional labels; the report breaks accuracy down by them |

Point a recipe at the file and train. A recipe is a TOML file; [`recipes/example-custom-en.toml`](recipes/example-custom-en.toml) is a complete starting point:

<details open>
<summary><code>recipes/example-custom-en.toml</code></summary>

```toml
# omj example recipe: example-custom-en
# Train on your own decisions: examples/support-decisions.jsonl (40 synthetic support tickets, CC0,
# three questions each) mixed with Banking77 so the adapter does not overfit a handful of items.
# Replace the jsonl path with your file; it uses the same format as the bench suites.
#
# Data: examples/support-decisions.jsonl (synthetic, CC0-1.0); Banking77 (PolyAI, CC-BY-4.0, downloaded at run time)

name = "example-custom-en"
base_model = "Qwen/Qwen3.5-0.8B"
revision = "2fc06364715b967f1860aea9cf38778875588b17"

[data]
source = "mix"
locale = "en-US"
shuffle_options = true

[[data.mix]]
name = "jsonl"
path = "../examples/support-decisions.jsonl"   # relative to this recipe file; train_n left out = every record
dev_n = 20

[[data.mix]]
name = "banking77"
train_n = 1000
dev_n = 50

[lora]
r = 8
alpha = 16
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]

[train]
lr = 2e-4
epochs = 2
batch_size = 4
grad_accum = 4
max_seq_len = 1024
label_smoothing = 0.05
brier_weight = 0.5
seed = 20260929
eval_every_steps = 50
```

</details>

| Section | What it sets |
|---|---|
| `name`, `base_model`, `revision` | Run name (adapters go to `~/.omj/adapters/<name>/`) and the Hugging Face base model, pinned to a revision |
| `[data]` | `source = "mix"` combines the `[[data.mix]]` entries; `shuffle_options` shuffles option order during training |
| `[[data.mix]]` | One per source. `name = "jsonl"` with `path` is your file (relative to the recipe); `train_n` samples that many records (left out = all for jsonl), `dev_n` sets the dev share |
| `[lora]` | Adapter rank, scaling, dropout and the layers it attaches to |
| `[train]` | Learning rate, epochs, batch size × `grad_accum` (effective batch), context length, `brier_weight` for calibration, seed and how often to evaluate |

```bash
uv sync --extra semif --extra dev
uv run omj train --recipe recipes/example-custom-en.toml                    # a recipe file (any path)
uv run omj bench --adapter ~/.omj/adapters/example-custom-en/<run_id>/best --suite my-test.jsonl --view
```

`--recipe` takes a path to a `.toml` file, or the bare name of a recipe bundled in `recipes/` (`--recipe example-custom-en`). Copy the example, change `name` and `path`, and run it with `--recipe my-recipe.toml`.

- Every row is validated before training starts; a bad row fails with its line number.
- 10% of your items (by `id`) are held out for the dev split that picks the best checkpoint.
- Keep a separate test file for `bench --suite my-test.jsonl`. Training warns when your items share 8-word runs with the bundled evaluation suites, since those scores would be inflated.
- Public sources you can mix in: `massive`, `massive-en`, `banking77`, `klue-ynat`, `klue-nli`, `nsmc`, `openjev-business` (downloaded on first use). [`recipes/example-intent-en.toml`](recipes/example-intent-en.toml) uses only public data.
- With `--adapter`, `bench` and `serve` run locally on semif and load the base model the adapter was trained on, in 4-bit when it would not fit the GPU.
- Pick checkpoints on `omj-holdout` or your own test file, not on JevBench; `omj-holdout` is a public development suite, not a sealed test set.

## Use it from the TypeSafe SDK

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8799")
```

For side-by-side comparisons in a browser, see [Web playground](#web-playground).

## Known limitations

- Tested on Windows 11 with an NVIDIA GPU. Linux is expected to work; local models need CUDA, so macOS is not supported for them.
- `omj train` fits calibration temperatures as if every question were a `choice`; yes/no and `score` questions in a mixed recipe share that temperature. The bundled example uses `choice` only.
- The MASSIVE suites and training download their data on first use.
- `omj compare` compares finished reports; it does not run models or test whether a difference is significant. The `≈` mark against Jev is an interval check, not a paired test.
- The bundled Jev values date from 2026-09-22 (Jev 1.13); a newer Jev may score differently. Measure it yourself with a key for current numbers.

## Acknowledgements

oh-my-jev builds on ideas and interfaces from these projects; no code is copied from them:
[TypeSafe](https://docs.typesafe.ai) (System One wire protocol and SDK),
[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (single-pass option logit readout),
[Kev](https://github.com/jaredpalmer/kev) and [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) (LoRA-trained open decision models),
and [JevBench](https://github.com/fstandhartinger/jevbench) (public items and scoring conventions).

## License

Code: [Apache-2.0](LICENSE). Bundled suite data keeps its own licenses ([suites/NOTICE.md](suites/NOTICE.md)); dependencies and datasets are listed in [THIRD-PARTY.md](THIRD-PARTY.md).

> Not affiliated with or endorsed by TypeSafe AI. "Jev" and "System One" are names used by TypeSafe AI; this is an independent project implementing a compatible interface.
