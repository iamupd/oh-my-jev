# Third-party notices

omj is Apache-2.0. It depends on, downloads, or refers to the following third-party work, each under its own licence.

## Python dependencies (installed, not vendored)

| Package | Licence |
|---|---|
| typer, fastapi, uvicorn, pydantic, httpx, huggingface-hub, numpy, tomli-w | MIT / BSD-3-Clause / Apache-2.0 (see each package) |
| torch | BSD-3-Clause |
| transformers, accelerate, peft | Apache-2.0 |
| bitsandbytes | MIT |
| pyarrow | Apache-2.0 |
| flash-linear-attention (optional, installed separately) | MIT |
| typesafe-sdk (dev only) | MIT |

## Datasets (downloaded at run time, not redistributed)

| Dataset | Used for | Licence |
|---|---|---|
| MASSIVE 1.1 (Amazon) | `massive-ko` / `massive-en` suites, `massive` training source | CC-BY-4.0 |
| JevBench public items (Benchmark Heaven) | `jevbench-public` suite | see the JevBench repository |
| Banking77 (PolyAI) | training source | CC-BY-4.0 |
| KLUE YNAT / NLI | training sources | CC-BY-SA-4.0 |
| NSMC | training source | CC0-1.0 |
| Open-Jev release-v2 redistributable | training source (business families) | CC0-1.0 |

## Bundled suite content

`suites/omj-holdout.jsonl` contains items derived from the following datasets; each row carries its `source` and `license`. Authors, source splits, the changes made, the ShareAlike terms and the MIT license text are in [suites/NOTICE.md](suites/NOTICE.md), which is shipped with the package:

| Dataset | Licence |
|---|---|
| BoolQ (Google) | CC-BY-SA-3.0 |
| ARC-Challenge (AI2) | CC-BY-SA-4.0 |
| CommonsenseQA | MIT |
| SVAMP | MIT |

`suites/omj-smoke.jsonl`, `suites/underdetermined.jsonl` and the MASSIVE criteria files are original synthetic content (CC0-1.0).

## Models

Model weights are downloaded from their publishers at run time (for example Qwen3.5, Apache-2.0) and are not redistributed.
