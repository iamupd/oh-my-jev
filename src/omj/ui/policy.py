"""Structured decision policies evaluated over System One answers (REQ-U07).

Rules are data, never code. The browser mirrors this module in
``static/index.html`` (``evaluatePolicy``); tests pin both to the same cases.

Rule forms (one condition key per rule, all rules must hold):
  {"q": id, "choice": key}                 choice answer equals key
  {"q": id, "prob_of": key, "gte": x}      P(key) >= x   (choice / score)
  {"q": id, "top_prob_gte": x}             max probability >= x (choice / score)
  {"q": id, "score_gte": x}                expected score >= x
  {"q": id, "score_lte": x}                expected score <= x
  {"q": id, "noul_gte": x}                 P(yes) >= x
  {"q": id, "noul_lte": x}                 P(yes) <= x
"""

from __future__ import annotations

from typing import Any

RULE_KEYS = ("choice", "prob_of", "top_prob_gte", "score_gte", "score_lte", "noul_gte", "noul_lte")


class PolicyError(ValueError):
    pass


def validate_policy(policy: dict, question_ids: set[str]) -> None:
    rules = policy.get("all")
    if not isinstance(rules, list) or not rules:
        raise PolicyError("policy.all must be a non-empty list of rules")
    for index, rule in enumerate(rules):
        if rule.get("q") not in question_ids:
            raise PolicyError(f"rule {index}: unknown question {rule.get('q')!r}")
        present = [k for k in RULE_KEYS if k in rule]
        if len(present) != 1:
            raise PolicyError(f"rule {index}: exactly one of {RULE_KEYS} is required")
        if present[0] == "prob_of" and not isinstance(rule.get("gte"), (int, float)):
            raise PolicyError(f"rule {index}: prob_of needs a numeric 'gte'")


def _check(rule: dict, answer: dict[str, Any] | None) -> tuple[bool, str]:
    if answer is None:
        return False, "no answer"
    probs = answer.get("probabilities") or {}
    if "choice" in rule:
        got = answer.get("choice")
        return got == rule["choice"], f"choice={got}"
    if "prob_of" in rule:
        p = float(probs.get(rule["prob_of"], 0.0))
        return p >= rule["gte"], f"P({rule['prob_of']})={p:.3f}"
    if "top_prob_gte" in rule:
        p = max((float(v) for v in probs.values()), default=0.0)
        return p >= rule["top_prob_gte"], f"top P={p:.3f}"
    if "score_gte" in rule or "score_lte" in rule:
        score = answer.get("score")
        if score is None:
            return False, "not a score answer"
        ok = score >= rule["score_gte"] if "score_gte" in rule else score <= rule["score_lte"]
        return ok, f"score={float(score):.2f}"
    if "noul_gte" in rule or "noul_lte" in rule:
        p = answer.get("noul")
        if p is None:
            return False, "not a noul answer"
        ok = p >= rule["noul_gte"] if "noul_gte" in rule else p <= rule["noul_lte"]
        return ok, f"P(yes)={float(p):.3f}"
    return False, "unknown rule"


def evaluate_policy(policy: dict, answers: dict[str, dict]) -> dict[str, Any]:
    """Return {"passed", "label", "rules": [{"rule", "ok", "detail"}]}."""
    results = []
    for rule in policy.get("all", []):
        ok, detail = _check(rule, answers.get(rule.get("q")))
        results.append({"rule": rule, "ok": ok, "detail": detail})
    passed = bool(results) and all(r["ok"] for r in results)
    label = policy.get("pass_label", "pass") if passed else policy.get("fail_label", "fail")
    return {"passed": passed, "label": label, "rules": results}
