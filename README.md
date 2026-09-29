<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.png">
  <img src="docs/images/logo.png" alt="oh-my-jev logo" width="120">
</picture>

# oh-my-jev

**Build a decision model. Measure when to trust it.**

Serve, benchmark, train and compare open *System One* decision models: small models that answer a typed question with a calibrated probability instead of free text.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Release](https://img.shields.io/badge/release-v0.1.0-2ea44f)](https://github.com/iamupd/oh-my-jev/releases)
[![TypeSafe compatible](https://img.shields.io/badge/API-TypeSafe%20%2Fv1%2Fsystemone-6f42c1)](#use-it-from-the-typesafe-sdk)

[Quick start](#quick-start) · [Commands](#commands) · [Train](#train-your-own-decision-model) · [Limitations](#known-limitations) · [한국어](README.ko.md)

<img src="docs/images/bench.png" alt="omj bench output: metrics, per-tag breakdown, reliability table and summary" width="760">

</div>

## Why

Agents make many small calls: *which team gets this ticket, is this answer grounded, should a human look at it.* A generated answer gives you no threshold to act on. A decision model returns a probability for each option, so you can act automatically above 0.9 and escalate below it, provided the probabilities are honest.

oh-my-jev is the workbench for that loop: run a decision model behind a TypeSafe-compatible endpoint, measure accuracy **and** calibration, fine-tune your own, and compare it with a reference.

## Highlights

- 🚀 **One-pass readout.** Local models answer from the option-label logits of a single forward pass: no generation, one prefill per decision.
- 🎯 **Calibration, not just accuracy.** ECE, Brier, NLL, coverage at 5% risk, reliability tables, 95% confidence intervals and temperature fitting.
- 🔌 **Drop-in endpoint.** `omj serve` speaks TypeSafe's `/v1/systemone`, so the official SDK works by changing `base_url`.
- 🧪 **Train your own.** LoRA / QLoRA recipes with label-restricted cross-entropy and a Brier term; 4-bit loading for small GPUs.
- ⚖️ **Side by side.** Compare reports in the terminal, or send one situation to two models in the web playground (English / 한국어).
- 🧰 **Works on first install.** A mock backend needs no GPU and no key, so every command runs before you download a model.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/iamupd/oh-my-jev.git
cd oh-my-jev
uv sync --extra dev
uv run omj init --yes                 # detects your GPU and keys, writes ~/.omj/config.toml
uv run omj bench --suite omj-smoke    # 30 items on the backend init chose
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
- `omj bench` uses the configured backend unless you pass `--backend NAME` or `--endpoint URL`

</details>

## Commands

| Command | What you get |
|---|---|
| `omj init` | Hardware and key detection, a smoke test, and `~/.omj/config.toml` |
| `omj serve` | A TypeSafe-compatible `/v1/systemone` endpoint over a local model, a remote API or a mock |
| `omj bench` | Accuracy, calibration, per-type and per-tag breakdowns and a reliability table, saved as `report.md` and `report.json` |
| `omj compare` | Two or more reports side by side, with differences against the first |
| `omj train` | LoRA / QLoRA fine-tuning from a TOML recipe |
| `omj ui` | A local web playground for one or two models, with policy thresholds |

Run each as `uv run omj …`, or activate `.venv` and call `omj …` directly.

| Backend | Runs | Needs |
|---|---|---|
| `semif` | open weights in-process, option-logit readout, optional LoRA adapter | CUDA GPU, `uv sync --extra semif` |
| `typesafe` | the hosted Jev API, directly or through OpenRouter | `JEV_KEY` or `OPENROUTER_KEY` |
| `kev` | a running [Kev](https://github.com/jaredpalmer/kev) server | the Kev server |
| `mock` | keyword overlap; a test double, not a model | nothing |

## Compare against a reference

```bash
uv run omj compare jev/report.json coco-v13/report.json coco-v14/report.json --labels jev-1.13,coco-v13-nf4,coco-v14-nf4 --out runs/cmp
```

<div align="center">
<img src="docs/images/compare.png" alt="omj compare output: metrics of three reports with coloured differences" width="760">
</div>

Differences are coloured by each metric's direction (green is better), and the best value in each row is bold. The numbers above were measured in September 2026 on the 231 public JevBench items.

## Local models

```bash
uv sync --extra semif --extra dev    # torch, transformers, peft (several GB)
uv run omj init --yes                # 6-24 GB CUDA GPU: downloads a Qwen3.5 base (2B is about 4.5 GB)
uv run omj serve
```

- `--no-download` writes the config without fetching weights.
- On an 8 GB GPU, set `quant = "nf4"` under `[backend]` in `config.toml` to load a 4B base in 4-bit.

## Train your own decision model

```bash
uv sync --extra semif --extra dev
uv run omj train --recipe example-intent-en    # small untuned example on Qwen3.5-0.8B
uv run omj bench --adapter ~/.omj/adapters/example-intent-en/<run_id>/best --suite omj-holdout
```

- A recipe lists its data sources (`[[data.mix]]`), LoRA settings and training settings. The bundled example is a starting point; swap in your own decisions.
- With `--adapter`, `bench` and `serve` load the base model the adapter was trained on.
- Pick checkpoints on `omj-holdout` (300 items from BoolQ, ARC-Challenge, CommonsenseQA and SVAMP, none used by the bundled training sources). It is a public development suite, not a sealed test set; report public benchmark numbers, but do not select on them.

## Use it from the TypeSafe SDK

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8799")
```

The web playground (`uv run omj ui --target a=mock --target b=mock`) opens in English; add `?lang=ko` or use the **EN / 한국어** switch for Korean.

## Known limitations

- Tested on Windows 11 with an NVIDIA GPU. Linux is expected to work; local models need CUDA, so macOS is not supported for them.
- `omj train` fits calibration temperatures as if every question were a `choice`; yes/no and `score` questions in a mixed recipe share that temperature. The bundled example uses `choice` only.
- The MASSIVE suites and training download their data on first use.
- `omj compare` compares finished reports; it does not run models or test whether a difference is significant.

## Acknowledgements

oh-my-jev builds on ideas and interfaces from these projects; no code is copied from them:
[TypeSafe](https://docs.typesafe.ai) (System One wire protocol and SDK),
[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (single-pass option logit readout),
[Kev](https://github.com/jaredpalmer/kev) and [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) (LoRA-trained open decision models),
and [JevBench](https://github.com/fstandhartinger/jevbench) (public items and scoring conventions).

## License

Code: [Apache-2.0](LICENSE). Bundled suite data keeps its own licenses ([suites/NOTICE.md](suites/NOTICE.md)); dependencies and datasets are listed in [THIRD-PARTY.md](THIRD-PARTY.md).

> Not affiliated with or endorsed by TypeSafe AI. "Jev" and "System One" are names used by TypeSafe AI; this is an independent project implementing a compatible interface.
