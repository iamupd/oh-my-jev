"""Mixed training sources for ``omj train`` (recipe data.source = "mix").

Every source is converted to the same ``DecisionRecord`` (state, question,
expected key) and then rendered with the serving prompt renderer, so training
and serving never diverge. Downloads are cached under ``$OMJ_HOME/cache/p2``.

Sources and licenses:
- massive:        AmazonScience MASSIVE 1.1 (CC-BY-4.0), reuses omj.train.data
- banking77:      PolyAI Banking77 via legacy-datasets/banking77 parquet (CC-BY-4.0)
- klue-ynat:      KLUE YNAT topic classification (CC-BY-SA-4.0)
- klue-nli:       KLUE NLI (CC-BY-SA-4.0)
- nsmc:           Naver Sentiment Movie Corpus, e9t/nsmc on GitHub (CC0-1.0)
- openjev-business: ZefanCai/Open-Jev release-v2 redistributable (CC0-1.0),
                  business-decision families only (games are excluded)
"""

from __future__ import annotations

import csv
import io
import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from omj.config import omj_home
from omj.errors import ErrorCode, OmjError

Split = Literal["train", "dev"]

HF = "https://huggingface.co/api/datasets"
URLS = {
    "banking77": {"train": f"{HF}/legacy-datasets/banking77/parquet/default/train/0.parquet"},
    "klue-ynat": {
        "train": f"{HF}/klue/klue/parquet/ynat/train/0.parquet",
        "dev": f"{HF}/klue/klue/parquet/ynat/validation/0.parquet",
    },
    "klue-nli": {
        "train": f"{HF}/klue/klue/parquet/nli/train/0.parquet",
        "dev": f"{HF}/klue/klue/parquet/nli/validation/0.parquet",
    },
    "nsmc": {
        "train": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
        "dev": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
    },
    "openjev-business": {
        "train": f"{HF}/ZefanCai/Open-Jev/parquet/release-v2-redistributable/train/0.parquet",
        "dev": f"{HF}/ZefanCai/Open-Jev/parquet/release-v2-redistributable/validation/0.parquet",
    },
}
SOURCE_NAMES = ("massive", "massive-en", "banking77", "klue-ynat", "klue-nli", "nsmc", "openjev-business")

YNAT_LABELS = {
    "IT과학": "IT, 과학, 인터넷, 모바일 기술 관련 뉴스",
    "경제": "금융, 산업, 기업, 부동산 등 경제 뉴스",
    "사회": "사건사고, 교육, 노동, 환경 등 사회 뉴스",
    "생활문화": "건강, 여행, 공연, 음식 등 생활·문화 뉴스",
    "세계": "해외 국가와 국제 관계 뉴스",
    "스포츠": "경기, 선수, 구단 등 스포츠 뉴스",
    "정치": "정부, 국회, 정당, 선거 등 정치 뉴스",
}
NLI_LABELS = {
    "entailment": "전제가 참이면 가설도 반드시 참이다",
    "neutral": "전제만으로는 가설의 참거짓을 알 수 없다",
    "contradiction": "전제가 참이면 가설은 거짓이다",
}
BANKING77_NAMES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working",
    "country_support", "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details",
    "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
    "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
    "top_up_limits", "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", "verify_top_up",
    "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]
OPENJEV_BUSINESS_PREFIXES = ("workflow-controls", "customer-control", "reasoning-control")
BANKING77_DEV_FRACTION = 0.05


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    source: str
    state: Any
    question: dict
    expected: str
    group: int = 0  # stable integer used to seed the option shuffle


Fetch = Callable[[str], bytes]


def _cache_dir(cache_dir: Path | str | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else omj_home() / "cache" / "p2"


def fetch_cached(url: str, name: str, *, cache_dir: Path | str | None = None, fetch: Fetch | None = None) -> bytes:
    """Return the bytes of ``url``, downloading once into the cache."""
    target = _cache_dir(cache_dir) / name
    if target.is_file() and target.stat().st_size > 0:
        return target.read_bytes()
    if fetch is None:
        from omj.bench.suites.base import http_fetch as fetch  # honors OMJ_NO_NETWORK
    data = fetch(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def _parquet_rows(data: bytes) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise OmjError(ErrorCode.E_BACKEND, "mixed recipes need pyarrow; run 'uv sync --extra semif'") from exc
    return pq.read_table(io.BytesIO(data)).to_pylist()


def _group(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


# ---------------------------------------------------------------- converters


def banking77_records(rows: Iterable[dict], split: Split) -> list[DecisionRecord]:
    rows = list(rows)
    criteria = {name: name.replace("_", " ").strip("?") for name in BANKING77_NAMES}
    question = {"type": "choice", "instructions": "What is the customer's banking request?", "criteria": criteria}
    out = []
    for idx, row in enumerate(rows):
        in_dev = (_group(f"b77:{idx}") % 1000) < int(BANKING77_DEV_FRACTION * 1000)
        if (split == "dev") != in_dev:
            continue
        out.append(DecisionRecord(f"banking77:{idx}", "banking77", row["text"], question,
                                  BANKING77_NAMES[int(row["label"])], _group(f"b77:{idx}")))
    return out


def ynat_records(rows: Iterable[dict], split: Split) -> list[DecisionRecord]:
    names = list(YNAT_LABELS)
    question = {"type": "choice", "instructions": "이 뉴스 제목의 분야는?", "criteria": dict(YNAT_LABELS)}
    return [DecisionRecord(f"ynat:{r['guid']}", "klue-ynat", r["title"], question, names[int(r["label"])],
                           _group(r["guid"])) for r in rows]


def nli_records(rows: Iterable[dict], split: Split) -> list[DecisionRecord]:
    names = list(NLI_LABELS)
    question = {"type": "choice", "instructions": "전제가 참이라고 할 때, 가설과의 관계는?", "criteria": dict(NLI_LABELS)}
    return [DecisionRecord(f"nli:{r['guid']}", "klue-nli", {"전제": r["premise"], "가설": r["hypothesis"]},
                           question, names[int(r["label"])], _group(r["guid"])) for r in rows
            if int(r["label"]) in (0, 1, 2)]


def nsmc_records(text: str, split: Split) -> list[DecisionRecord]:
    question = {"type": "noul", "instructions": "이 영화 리뷰는 긍정적인가?",
                "criteria": {"true": "영화를 좋게 평가함", "false": "영화를 나쁘게 평가함"}}
    out = []
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    for r in reader:
        doc = (r.get("document") or "").strip()
        if not doc or r.get("label") not in ("0", "1"):
            continue
        out.append(DecisionRecord(f"nsmc:{r['id']}", "nsmc", doc, question,
                                  "yes" if r["label"] == "1" else "no", _group(r["id"])))
    return out


def openjev_records(rows: Iterable[dict], split: Split) -> list[DecisionRecord]:
    """Business-decision rows with an exact (one-hot) target; games and soft targets are skipped."""
    import json

    out = []
    for r in rows:
        if not str(r["source"]).startswith(OPENJEV_BUSINESS_PREFIXES):
            continue
        target = list(r["target"])
        best = max(target)
        if best < 0.999 or target.count(best) != 1:
            continue
        idx = target.index(best)
        options = list(r["options"])
        state = json.loads(r["state_json"]) if r.get("state_json") else ""
        kind = r["kind"]
        if kind == "noul":
            question = {"type": "noul", "instructions": r["question"]}
            expected = "yes" if options[idx].lower() == "yes" else "no"
        elif kind == "choice":
            criteria: dict[str, str] = {}
            for opt in options:
                key, _, desc = opt.partition(": ")
                criteria[key.strip()] = desc.strip() or key.strip()
            if len(criteria) != len(options):
                continue
            question = {"type": "choice", "instructions": r["question"], "criteria": criteria}
            expected = list(criteria)[idx]
        elif kind == "score":
            question = {"type": "score", "instructions": r["question"], "criteria": options}
            expected = str(idx)
        else:
            continue
        out.append(DecisionRecord(f"openjev:{r['id']}", "openjev-business", state, question, expected,
                                  _group(str(r["id"]))))
    return out


def massive_records(locale: str, split: Split, questions: list[str], *, cache_dir=None, downloader=None,
                    source: str = "massive") -> list[DecisionRecord]:
    from omj.train.data import load_criteria, load_massive_rows, question_dict

    criteria = load_criteria(locale)
    rows = load_massive_rows(locale, split, cache_dir=cache_dir, downloader=downloader)
    out = []
    for row in rows:
        for qid in questions:
            out.append(DecisionRecord(f"{source}:{row['id']}/{qid}", source, row["utt"],
                                      question_dict(qid, criteria, locale), row[qid],
                                      int(row["id"]) * 7 + zlib.crc32(qid.encode())))
    return out


JSONL_DEV_PER_MILLE = 100  # 10% of a user's items (by id) are held out as dev


def jsonl_records(path: str | Path, split: Split) -> list[DecisionRecord]:
    """Your own decisions in the bundled-suite format, one record per labelled question.

    Rows are validated exactly like a bench suite. The train/dev split is by item id, so all
    questions of one item land on the same side; questions without an expected answer are skipped.
    """
    from omj.bench.suites.local import load_local

    source = f"jsonl:{Path(path).stem}"
    out = []
    for item in load_local(Path(path).expanduser(), Path(path).stem):
        group = _group(f"{source}:{item.id}")
        if ((group % 1000) < JSONL_DEV_PER_MILLE) != (split == "dev"):
            continue
        for qid, question in item.questions.items():
            label = (item.expected or {}).get(qid)
            if label is not None:
                out.append(DecisionRecord(f"{source}:{item.id}:{qid}", source, item.state, question, str(label),
                                          _group(f"{source}:{item.id}:{qid}")))
    return out


def load_source(name: str, split: Split, *, locale: str = "ko-KR", questions: list[str] | None = None,
                cache_dir=None, fetch: Fetch | None = None, path: str = "") -> list[DecisionRecord]:
    """All records of one source and split (downloads once)."""
    if name == "jsonl":
        return jsonl_records(path, split)
    if name == "massive":
        return massive_records(locale, split, questions or ["scenario", "intent"], downloader=fetch)
    if name == "massive-en":
        return massive_records("en-US", split, questions or ["scenario", "intent"], downloader=fetch,
                               source="massive-en")
    if name not in URLS:
        raise OmjError(ErrorCode.E_CONFIG, f"unknown training source {name!r} (expected one of {', '.join(SOURCE_NAMES)})")
    url_split = "train" if name == "banking77" else split
    blob = fetch_cached(URLS[name][url_split], f"{name}-{url_split}.bin", cache_dir=cache_dir, fetch=fetch)
    if name == "nsmc":
        return nsmc_records(blob.decode("utf-8"), split)
    rows = _parquet_rows(blob)
    return {"banking77": banking77_records, "klue-ynat": ynat_records, "klue-nli": nli_records,
            "openjev-business": openjev_records}[name](rows, split)


def balanced_sample(records: list[DecisionRecord], n: int, seed: int) -> list[DecisionRecord]:
    """Deterministic sample of ``n`` records, balanced across question types when possible."""
    by_kind: dict[str, list[DecisionRecord]] = {}
    for rec in sorted(records, key=lambda r: r.id):
        by_kind.setdefault(rec.question["type"], []).append(rec)
    rng = random.Random(seed)
    kinds = sorted(by_kind)
    quota = {k: n // len(kinds) for k in kinds}
    for k in kinds[: n - sum(quota.values())]:
        quota[k] += 1
    picked: list[DecisionRecord] = []
    spare: list[DecisionRecord] = []
    for k in kinds:
        pool = by_kind[k][:]
        rng.shuffle(pool)
        picked.extend(pool[: quota[k]])
        spare.extend(pool[quota[k]:])
    if len(picked) < n:
        rng.shuffle(spare)
        picked.extend(spare[: n - len(picked)])
    return picked[:n]


# ---------------------------------------------------------------- surface augmentation

_LEAD_INS = ("", "Based on the state above, answer: ", "Decide: ", "Question: ", "다음 질문에 답하시오: ")


def _render_lines(state: dict) -> str:
    return "\n".join(f"{k}:{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}" for k, v in state.items())


def augment_surface(rec: DecisionRecord, rng: random.Random) -> DecisionRecord:
    """A copy with the same meaning but different surface form (the expected answer is remapped).

    - dict state: keys reordered, and half the time rendered as "key: value" lines;
    - instructions: an optional neutral lead-in;
    - choice: half the time option keys are renamed to neutral ids so the answer must come
      from the descriptions (an empty description inherits the old key text).
    """
    state = rec.state
    if isinstance(state, dict) and state:
        items = list(state.items())
        rng.shuffle(items)
        state = dict(items)
        if rng.random() < 0.5:
            state = _render_lines(state)
    question = dict(rec.question)
    question["instructions"] = rng.choice(_LEAD_INS) + str(question["instructions"])
    expected = rec.expected
    if question["type"] == "choice" and rng.random() < 0.5:
        old = list(question["criteria"].items())
        new_keys = [f"opt_{chr(ord('a') + i)}" if i < 26 else f"opt_{i}" for i in range(len(old))]
        question["criteria"] = {nk: (desc if desc else ok) for nk, (ok, desc) in zip(new_keys, old)}
        expected = new_keys[[k for k, _ in old].index(rec.expected)]
    return DecisionRecord(rec.id + ":aug", rec.source, state, question, expected, _group(rec.id + ":aug"))


def add_surface_augmentations(records: list[DecisionRecord], fraction: float, seed: int) -> list[DecisionRecord]:
    """Deterministically pick ``fraction`` of records and append their surface-perturbed copies."""
    if fraction <= 0:
        return list(records)
    picked = [r for r in records if (_group(f"aug:{seed}:{r.id}") % 10_000) < fraction * 10_000]
    return list(records) + [augment_surface(r, random.Random(_group(f"augrng:{seed}:{r.id}"))) for r in picked]
