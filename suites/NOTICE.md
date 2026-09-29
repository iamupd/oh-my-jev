# Notices for bundled suite data

The oh-my-jev source code is licensed under Apache-2.0 (see `LICENSE`). The data files in this
directory are **not** covered by that license; each is distributed under the terms below.

## omj-holdout.jsonl

300 items (75 per source) sampled deterministically (seed 20260924) from four public datasets by
`scripts/make_holdout_suite.py`, after dropping candidates that share an 8-word sequence with the
bundled training sources. Every row records its `source` and `license`.

Changes made to the original data: sampling; conversion into the omj decision format (situation in
`state`, question in `instructions`, answer options relabelled as `criteria` keys, expected answer
in `expected`); for SVAMP, three wrong numbers derived from the correct answer (small offsets, doubling,
halving, times ten) were added to form a 4-way choice. No text was otherwise rewritten.

| Rows | Source | Split | Authors / copyright | License |
|---|---|---|---|---|
| `holdout-boolq-*` | [google/boolq](https://huggingface.co/datasets/google/boolq) | validation | Clark et al., "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions", NAACL 2019; Google | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| `holdout-arc-*` | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) (ARC-Challenge) | test | Clark et al., "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge", 2018; Allen Institute for AI | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `holdout-commonsense_qa-*` | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | validation | Talmor, Herzig, Lourie and Berant, "CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge", NAACL 2019 | MIT ([license statement](https://github.com/jonathanherzig/commonsenseqa/issues/5)) |
| `holdout-svamp-*` | [ChilleD/SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP), from [arkilpatel/SVAMP](https://github.com/arkilpatel/SVAMP) | test | Patel, Bhattamishra and Goyal, "Are NLP Models really able to Solve Simple Math Word Problems?", NAACL 2021; Copyright (c) 2021 Arkil Patel | MIT (text below) |

ShareAlike: the rows derived from BoolQ and ARC are adapted material and are distributed under the
same CC BY-SA license as their source (3.0 and 4.0 respectively). The other rows keep the MIT license
of their source.

### MIT license text (SVAMP; CommonsenseQA is released under the same terms by its authors)

```
MIT License

Copyright (c) 2021 Arkil Patel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Other files

- `omj-smoke.jsonl`, `underdetermined.jsonl`, `massive-criteria.*.json`: original synthetic content, CC0-1.0.
- `massive-ids.txt`: item identifiers of the MASSIVE 1.1 dataset (Amazon, CC BY 4.0); the MASSIVE text itself is downloaded at run time and not bundled.
