"""Regenerate suites/massive-ids.txt: a seeded, scenario-stratified sample of MASSIVE test ids.

Run manually (it downloads the 40 MB MASSIVE tarball):

    uv run python scripts/make_massive_ids.py

The committed output is what the massive-ko / massive-en loaders read, so rerunning
this with the same seed must reproduce the file byte for byte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from omj.bench.suites import massive  # noqa: E402
from omj.errors import OmjError, format_error  # noqa: E402

HEADER_SOURCE = "# source=amazon-massive-dataset-1.1 test split ko-KR/en-US"
HEADER_SAMPLE = "# seed={seed} strata=scenario n={n}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "suites" / "massive-ids.txt",
        help="where to write the id list (default: suites/massive-ids.txt)",
    )
    parser.add_argument("--n", type=int, default=massive.DEFAULT_SAMPLE_SIZE, help="how many ids to sample")
    parser.add_argument("--seed", type=int, default=massive.DEFAULT_SEED, help="sampling seed")
    parser.add_argument("--cache-dir", type=Path, default=None, help="override $OMJ_HOME/cache")
    parser.add_argument("--dry-run", action="store_true", help="print the quota table without writing")
    return parser.parse_args(argv)


def check_locales_are_aligned(ko_rows: list[dict], en_rows: list[dict]) -> None:
    """MASSIVE pairs the locales by id; the loaders rely on that, so verify it here."""
    ko_ids = [row["id"] for row in ko_rows]
    en_ids = [row["id"] for row in en_rows]
    if ko_ids != en_ids:
        raise SystemExit("ko-KR and en-US test splits do not share the same id order")

    ko_labels = {row["id"]: (row["scenario"], row["intent"]) for row in ko_rows}
    mismatched = [row["id"] for row in en_rows if ko_labels[row["id"]] != (row["scenario"], row["intent"])]
    if mismatched:
        raise SystemExit(f"{len(mismatched)} ids carry different labels per locale (first: {mismatched[0]})")


def render(ids: list[int], n: int, seed: int) -> str:
    header = [HEADER_SOURCE, HEADER_SAMPLE.format(seed=seed, n=n)]
    return "\n".join(header + [str(i) for i in ids]) + "\n"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    ko_rows = massive.read_locale_rows("ko", cache_dir=args.cache_dir)
    en_rows = massive.read_locale_rows("en", cache_dir=args.cache_dir)
    check_locales_are_aligned(ko_rows, en_rows)

    ids = massive.stratified_ids(ko_rows, n=args.n, seed=args.seed)
    table = massive.scenario_quota_table(ko_rows, n=args.n, seed=args.seed)

    print(f"test rows: {len(ko_rows)} per locale (ko-KR, en-US)")
    print(f"{'scenario':<16}{'test rows':>10}{'sampled':>9}")
    for scenario, total, sampled in table:
        print(f"{scenario:<16}{total:>10}{sampled:>9}")
    print(f"{'total':<16}{len(ko_rows):>10}{len(ids):>9}")

    if args.dry_run:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(ids, args.n, args.seed), encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(ids)} ids)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OmjError as error:
        print(format_error(error), file=sys.stderr)
        raise SystemExit(error.exit_code) from error
