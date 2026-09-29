"""Loader for the MASSIVE 1.1 ko-KR / en-US test splits, converted to decision items."""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import tarfile
from pathlib import Path

from omj.bench.suites.base import (
    DecisionItem,
    Downloader,
    cache_root,
    resolve_downloader,
    validate_item,
)
from omj.bench.suites.local import SUITES_DIR
from omj.errors import ErrorCode, OmjError

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
MASSIVE_VERSION = "1.1"
MASSIVE_SOURCE = "AmazonScience/massive@1.1"
MASSIVE_LICENSE = "CC-BY-4.0"

LOCALES: dict[str, str] = {"ko": "ko-KR", "en": "en-US"}
MEMBERS: dict[str, str] = {code: f"{MASSIVE_VERSION}/data/{tag}.jsonl" for code, tag in LOCALES.items()}

TEST_PARTITION = "test"
DEFAULT_SAMPLE_SIZE = 300
DEFAULT_SEED = 20260922

DEFAULT_IDS_FILE = SUITES_DIR / "massive-ids.txt"
CRITERIA_FILE = "massive-criteria.{locale}.json"

INSTRUCTIONS: dict[str, dict[str, str]] = {
    "ko": {"scenario": "이 발화가 속한 분야는?", "intent": "사용자의 의도는?"},
    "en": {
        "scenario": "Which domain does this utterance belong to?",
        "intent": "What is the user's intent?",
    },
}

#: The 18 MASSIVE scenario labels, sorted so that criteria order is reproducible.
SCENARIOS: tuple[str, ...] = (
    "alarm",
    "audio",
    "calendar",
    "cooking",
    "datetime",
    "email",
    "general",
    "iot",
    "lists",
    "music",
    "news",
    "play",
    "qa",
    "recommendation",
    "social",
    "takeaway",
    "transport",
    "weather",
)

#: The 60 MASSIVE intent labels, sorted for the same reason.
INTENTS: tuple[str, ...] = (
    "alarm_query",
    "alarm_remove",
    "alarm_set",
    "audio_volume_down",
    "audio_volume_mute",
    "audio_volume_other",
    "audio_volume_up",
    "calendar_query",
    "calendar_remove",
    "calendar_set",
    "cooking_query",
    "cooking_recipe",
    "datetime_convert",
    "datetime_query",
    "email_addcontact",
    "email_query",
    "email_querycontact",
    "email_sendemail",
    "general_greet",
    "general_joke",
    "general_quirky",
    "iot_cleaning",
    "iot_coffee",
    "iot_hue_lightchange",
    "iot_hue_lightdim",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "iot_hue_lightup",
    "iot_wemo_off",
    "iot_wemo_on",
    "lists_createoradd",
    "lists_query",
    "lists_remove",
    "music_dislikeness",
    "music_likeness",
    "music_query",
    "music_settings",
    "news_query",
    "play_audiobook",
    "play_game",
    "play_music",
    "play_podcasts",
    "play_radio",
    "qa_currency",
    "qa_definition",
    "qa_factoid",
    "qa_maths",
    "qa_stock",
    "recommendation_events",
    "recommendation_locations",
    "recommendation_movies",
    "social_post",
    "social_query",
    "takeaway_order",
    "takeaway_query",
    "transport_query",
    "transport_taxi",
    "transport_ticket",
    "transport_traffic",
    "weather_query",
)


def massive_cache_dir(cache_dir: Path | str | None = None) -> Path:
    return cache_root(cache_dir) / "massive" / MASSIVE_VERSION


def download_massive(
    cache_dir: Path | str | None = None,
    downloader: Downloader | None = None,
) -> Path:
    """Fetch and verify the MASSIVE tarball, extracting only the two locale files."""
    root = massive_cache_dir(cache_dir)
    marker = root / ".complete"
    if marker.is_file():
        return root

    fetch = resolve_downloader(downloader)
    payload = fetch(MASSIVE_URL)

    digest = hashlib.sha256(payload).hexdigest()
    if digest != MASSIVE_SHA256:
        raise OmjError(
            ErrorCode.E_DOWNLOAD,
            f"sha256 mismatch for {MASSIVE_URL}: expected {MASSIVE_SHA256}, got {digest}",
        )

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for code, member_name in MEMBERS.items():
            try:
                member = tar.getmember(member_name)
            except KeyError as exc:
                raise OmjError(ErrorCode.E_DOWNLOAD, f"{member_name} is missing from the tarball") from exc
            extracted = tar.extractfile(member)
            if extracted is None:
                raise OmjError(ErrorCode.E_DOWNLOAD, f"{member_name} is not a regular file in the tarball")
            # Written by hand rather than via tar.extract(): the destination path is
            # ours, so a crafted member name can never escape the cache directory.
            (data_dir / f"{LOCALES[code]}.jsonl").write_bytes(extracted.read())

    marker.write_text(f"{MASSIVE_SHA256}\n", encoding="utf-8", newline="\n")
    return root


def stratified_ids(
    rows_test: list[dict],
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Pick n test ids, stratified by scenario, deterministically for a given seed."""
    by_scenario: dict[str, list[int]] = {}
    for row in rows_test:
        by_scenario.setdefault(str(row["scenario"]), []).append(int(row["id"]))

    total = sum(len(ids) for ids in by_scenario.values())
    if total == 0:
        return []
    n = min(n, total)

    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for scenario in sorted(by_scenario):
        exact = n * len(by_scenario[scenario]) / total
        quotas[scenario] = math.floor(exact)
        fractions.append((exact - math.floor(exact), scenario))

    # Largest-remainder: hand the leftover slots to the largest fractional parts,
    # breaking ties on scenario name so the result never depends on dict order.
    leftover = n - sum(quotas.values())
    for _, scenario in sorted(fractions, key=lambda pair: (-pair[0], pair[1]))[:leftover]:
        quotas[scenario] += 1

    picked: list[int] = []
    for scenario in sorted(by_scenario):
        candidates = sorted(by_scenario[scenario])
        quota = min(quotas[scenario], len(candidates))
        picked.extend(random.Random(seed).sample(candidates, quota))

    return sorted(picked)


def scenario_quota_table(
    rows_test: list[dict],
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[tuple[str, int, int]]:
    """(scenario, rows in the test split, sampled rows) for reporting."""
    totals: dict[str, int] = {}
    for row in rows_test:
        totals[str(row["scenario"])] = totals.get(str(row["scenario"]), 0) + 1

    by_id = {int(row["id"]): str(row["scenario"]) for row in rows_test}
    sampled: dict[str, int] = {}
    for item_id in stratified_ids(rows_test, n=n, seed=seed):
        sampled[by_id[item_id]] = sampled.get(by_id[item_id], 0) + 1

    return [(scenario, totals[scenario], sampled.get(scenario, 0)) for scenario in sorted(totals)]


def read_locale_rows(
    locale: str,
    cache_dir: Path | str | None = None,
    downloader: Downloader | None = None,
    partition: str | None = TEST_PARTITION,
) -> list[dict]:
    """Rows of one locale file, optionally narrowed to a single partition."""
    tag = _locale_tag(locale)
    root = download_massive(cache_dir=cache_dir, downloader=downloader)
    path = root / "data" / f"{tag}.jsonl"
    if not path.is_file():
        raise OmjError(ErrorCode.E_DOWNLOAD, f"{path} is missing from the MASSIVE cache")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OmjError(ErrorCode.E_SCHEMA, f"{path} line {lineno}: {exc}") from exc
            if partition is not None and row.get("partition") != partition:
                continue
            rows.append(row)
    return rows


def read_ids_file(path: Path | str) -> list[int]:
    target = Path(path)
    if not target.is_file():
        raise OmjError(ErrorCode.E_SCHEMA, f"MASSIVE id list not found: {target}")

    ids: list[int] = []
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            ids.append(int(stripped))
        except ValueError as exc:
            raise OmjError(ErrorCode.E_SCHEMA, f"{target} line {lineno}: {stripped!r} is not an id") from exc
    return ids


def load_criteria(locale: str) -> dict[str, dict[str, str]]:
    """The authored option descriptions for one language."""
    code = _locale_code(locale)
    path = SUITES_DIR / CRITERIA_FILE.format(locale=code)
    if not path.is_file():
        raise OmjError(ErrorCode.E_SCHEMA, f"MASSIVE criteria file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    for group, labels in (("scenario", SCENARIOS), ("intent", INTENTS)):
        options = data.get(group)
        if not isinstance(options, dict) or set(options) != set(labels):
            raise OmjError(
                ErrorCode.E_SCHEMA,
                f"{path}: '{group}' must describe exactly the {len(labels)} MASSIVE {group} labels",
            )
    return data


def load_massive(
    locale: str,
    cache_dir: Path | str | None = None,
    downloader: Downloader | None = None,
    ids_file: Path | str | None = None,
) -> list[DecisionItem]:
    """Convert the sampled MASSIVE test rows of one locale into DecisionItems."""
    code = _locale_code(locale)
    criteria = load_criteria(code)
    instructions = INSTRUCTIONS[code]
    # Resolved here rather than as a default argument so that the committed id list
    # stays overridable after import (tests, and a future --ids-file flag).
    wanted = read_ids_file(DEFAULT_IDS_FILE if ids_file is None else ids_file)
    wanted_set = set(wanted)

    rows = {int(row["id"]): row for row in read_locale_rows(code, cache_dir, downloader)}

    # Every item asks the same two questions, so the option maps are built once and
    # shared. Treat them as read-only: nothing downstream may mutate criteria in place.
    scenario_question = {
        "type": "choice",
        "instructions": instructions["scenario"],
        "criteria": {key: criteria["scenario"][key] for key in SCENARIOS},
    }
    intent_question = {
        "type": "choice",
        "instructions": instructions["intent"],
        "criteria": {key: criteria["intent"][key] for key in INTENTS},
    }

    items: list[DecisionItem] = []
    for row_id in wanted:
        row = rows.get(row_id)
        if row is None:
            continue
        item = DecisionItem(
            id=f"massive-{code}-{row_id}",
            suite=f"massive-{code}",
            state=row["utt"],
            questions={"scenario": scenario_question, "intent": intent_question},
            expected={"scenario": row["scenario"], "intent": row["intent"]},
            tags=["massive", code, row["scenario"]],
            pair_id=str(row_id),
            source=MASSIVE_SOURCE,
            license=MASSIVE_LICENSE,
        )
        validate_item(item)
        items.append(item)

    missing = wanted_set - {int(item.pair_id) for item in items if item.pair_id is not None}
    if missing:
        raise OmjError(
            ErrorCode.E_SCHEMA,
            f"massive-{code}: {len(missing)} sampled ids are absent from the test split "
            f"(first: {sorted(missing)[0]})",
        )
    return items


def _locale_code(locale: str) -> str:
    if locale in LOCALES:
        return locale
    raise OmjError(
        ErrorCode.E_SCHEMA,
        f"unknown MASSIVE locale {locale!r} (expected one of {', '.join(sorted(LOCALES))})",
    )


def _locale_tag(locale: str) -> str:
    return LOCALES[_locale_code(locale)]
