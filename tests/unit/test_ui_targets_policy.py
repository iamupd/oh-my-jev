"""omj ui: target parsing/config derivation, env-file loading, and policy evaluation.

# REQ-U03
# REQ-U04
# REQ-U07
# REQ-U11
# REQ-U12
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from omj.config import BackendSection, Config
from omj.errors import OmjError
from omj.ui import policy as pol
from omj.ui.app import PRESETS_DIR, STATIC_DIR, load_presets
from omj.ui.targets import TargetSpec, check_unique, default_targets, load_env_file, parse_target, target_config


def test_parse_target_keeps_windows_adapter_path() -> None:
    assert parse_target("mine=semif:C:\\adapters\\best") == TargetSpec("mine", "semif", "C:\\adapters\\best")
    assert parse_target("jev=typesafe") == TargetSpec("jev", "typesafe", "")


@pytest.mark.parametrize("bad", ["noequals", "=semif", "Bad Name=semif", "x="])
def test_parse_target_rejects_malformed(bad: str) -> None:
    with pytest.raises(OmjError):
        parse_target(bad)


def test_default_targets_adds_jev_only_with_key() -> None:
    cfg = Config(backend=BackendSection(name="semif", model="Qwen/Qwen3.5-2B", adapter="/a/best"))
    assert default_targets(cfg, {}) == [TargetSpec("local", "semif", "/a/best")]
    assert default_targets(cfg, {"JEV_KEY": "x"})[1] == TargetSpec("jev", "typesafe")
    only_jev = Config(backend=BackendSection(name="typesafe"))
    assert default_targets(only_jev, {"JEV_KEY": "x"}) == [TargetSpec("jev", "typesafe", "")]


def test_duplicate_target_names_rejected() -> None:
    with pytest.raises(OmjError):
        check_unique([TargetSpec("a", "mock"), TargetSpec("a", "typesafe")])


def test_target_config_clears_bearer_and_foreign_model() -> None:
    cfg = Config(backend=BackendSection(name="semif", model="Qwen/Qwen3.5-2B", revision="abc"))
    cfg = cfg.model_copy(update={"serve": cfg.serve.model_copy(update={"api_key": "secret-bearer"})})
    local = target_config(cfg, TargetSpec("local", "semif", "/a"))
    assert local.backend.model == "Qwen/Qwen3.5-2B" and local.backend.adapter == "/a"
    assert local.serve.api_key == ""
    jev = target_config(cfg, TargetSpec("jev", "typesafe"))
    assert jev.backend.name == "typesafe" and jev.backend.model == "" and jev.backend.revision == ""


def test_semif_target_needs_configured_model() -> None:
    with pytest.raises(OmjError):
        target_config(Config(backend=BackendSection(name="mock")), TargetSpec("local", "semif"))


def test_env_file_loads_names_without_overriding(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text('# comment\nJEV_KEY="abc123"\nexport OTHER=1\nKEEP=new\n', encoding="utf-8")
    env = {"KEEP": "old"}
    loaded = load_env_file(f, env)
    assert loaded == ["JEV_KEY", "OTHER"]
    assert env == {"KEEP": "old", "JEV_KEY": "abc123", "OTHER": "1"}


# ---------------------------------------------------------------- policy

ANSWERS = {
    "action": {"type": "choice", "choice": "promote", "probabilities": {"promote": 0.84, "hold": 0.1, "discard": 0.06}},
    "evidence": {"type": "score", "score": 2.7, "probabilities": {"0": 0.05, "1": 0.1, "2": 0.1, "3": 0.5, "4": 0.25}},
    "correction": {"type": "noul", "noul": 0.96},
}
CASES = [
    ({"q": "action", "choice": "promote"}, True),
    ({"q": "action", "choice": "hold"}, False),
    ({"q": "action", "prob_of": "promote", "gte": 0.8}, True),
    ({"q": "action", "prob_of": "promote", "gte": 0.9}, False),
    ({"q": "action", "top_prob_gte": 0.84}, True),
    ({"q": "evidence", "score_gte": 2.5}, True),
    ({"q": "evidence", "score_lte": 2.5}, False),
    ({"q": "correction", "noul_gte": 0.8}, True),
    ({"q": "correction", "noul_lte": 0.5}, False),
    ({"q": "missing", "noul_gte": 0.1}, False),
    ({"q": "action", "noul_gte": 0.1}, False),
]


@pytest.mark.parametrize("rule, expected", CASES)
def test_policy_rules(rule: dict, expected: bool) -> None:
    out = pol.evaluate_policy({"all": [rule], "pass_label": "P", "fail_label": "F"}, ANSWERS)
    assert out["passed"] is expected
    assert out["label"] == ("P" if expected else "F")


def test_policy_requires_all_rules() -> None:
    policy = {"all": [c[0] for c in CASES[:3]]}
    assert pol.evaluate_policy(policy, ANSWERS)["passed"] is False  # rule 2 fails
    assert pol.evaluate_policy({"all": []}, ANSWERS)["passed"] is False


def test_validate_policy_rejects_bad_rules() -> None:
    with pytest.raises(pol.PolicyError):
        pol.validate_policy({"all": [{"q": "x", "choice": "a"}]}, {"y"})
    with pytest.raises(pol.PolicyError):
        pol.validate_policy({"all": [{"q": "x", "choice": "a", "noul_gte": 0.1}]}, {"x"})
    with pytest.raises(pol.PolicyError):
        pol.validate_policy({"all": [{"q": "x", "prob_of": "a"}]}, {"x"})


def test_browser_policy_mirrors_python() -> None:
    """The JS evaluatePolicy in index.html must agree with policy.py on every case."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    start = html.index("function checkRule")
    end = html.index("function ruleText")
    script = html[start:end] + (
        "const cases = " + json.dumps([c[0] for c in CASES]) + ";\n"
        "const answers = " + json.dumps(ANSWERS, ensure_ascii=False) + ";\n"
        "console.log(JSON.stringify(cases.map(r => evaluatePolicy({all:[r]}, answers).passed)));\n"
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [c[1] for c in CASES]


def test_bundled_presets_are_valid_and_bilingual() -> None:
    presets = load_presets()
    assert len(presets) >= 4
    for p in presets:
        assert set(p["locales"]) == {"en", "ko"}, p["id"]
        for loc in p["locales"].values():
            assert loc["act"]["pass"] and loc["act"]["fail"] and loc["labels"]["pass"]


def test_presets_carry_no_private_network_details() -> None:
    text = " ".join(path.read_text(encoding="utf-8") for path in PRESETS_DIR.glob("*.json"))
    assert not re.search(r"\b(10|192\.168|172\.(1[6-9]|2\d|3[01]))\.\d+\.\d+", text)


def test_preset_locales_must_share_question_shape(tmp_path: Path) -> None:
    good = json.loads((PRESETS_DIR / "04-support-routing.json").read_text(encoding="utf-8"))
    good["locales"]["ko"]["questions"]["intent"]["criteria"] = {"key_a": "x", "key_b": "y"}
    (tmp_path / "bad.json").write_text(json.dumps(good, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_presets(tmp_path)


def test_preset_needs_english_locale(tmp_path: Path) -> None:
    good = json.loads((PRESETS_DIR / "04-support-routing.json").read_text(encoding="utf-8"))
    del good["locales"]["en"]
    (tmp_path / "bad.json").write_text(json.dumps(good, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="'en'"):
        load_presets(tmp_path)


def test_page_has_no_external_assets() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r'(src|href)\s*=\s*"https?://', html)


def test_page_defaults_to_english_with_korean_available() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert '<html lang="en">' in html
    assert "ko: {" in html and 'return "en";' in html


def test_resolve_access_rules() -> None:
    from omj.cli.ui_cmd import resolve_access

    assert resolve_access("127.0.0.1", False, None) is None
    assert resolve_access("localhost", False, "abc") == "abc"
    with pytest.raises(OmjError):
        resolve_access("0.0.0.0", False, None)
    generated = resolve_access("0.0.0.0", True, None)
    assert generated and len(generated) >= 24
    assert resolve_access("203.0.113.5", True, "given") == "given"


def test_parse_target_at_profile_takes_backend_from_that_config(tmp_path: Path) -> None:
    from omj.config import save_config

    profile = tmp_path / "qwen-0.8b.toml"
    cfg = Config()
    cfg.backend.name, cfg.backend.model, cfg.backend.quant = "semif", "Qwen/Qwen3.5-0.8B", "bf16"
    save_config(cfg, profile)

    spec = parse_target(f"small=@{profile}")
    assert (spec.name, spec.backend, spec.profile) == ("small", "semif", str(profile))

    main = Config()
    main.backend.name, main.backend.model = "semif", "Qwen/Qwen3.5-4B"
    gateway_bearer = "-".join(["not", "a", "real", "bearer"])
    main = main.model_copy(update={"serve": main.serve.model_copy(update={"api_key": gateway_bearer})})
    derived = target_config(main, spec)
    assert derived.backend.model == "Qwen/Qwen3.5-0.8B"
    assert derived.serve.api_key == ""


def test_parse_target_at_profile_must_exist(tmp_path: Path) -> None:
    with pytest.raises(OmjError):
        parse_target(f"x=@{tmp_path / 'missing.toml'}")


def test_adapter_target_follows_the_adapter_base_model(tmp_path: Path) -> None:
    import json

    adapter = tmp_path / "best"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}), encoding="utf-8")
    main = Config()
    main.backend.name, main.backend.model, main.backend.revision = "semif", "Qwen/Qwen3.5-2B", "15852e8c"

    derived = target_config(main, TargetSpec("tuned", "semif", str(adapter)))
    assert (derived.backend.model, derived.backend.revision, derived.backend.adapter) == ("Qwen/Qwen3.5-4B", "", str(adapter))


def test_parse_target_hf_id_runs_that_model_on_semif() -> None:
    spec = parse_target("small=Qwen/Qwen3.5-0.8B")
    assert (spec.backend, spec.model, spec.adapter) == ("semif", "Qwen/Qwen3.5-0.8B", "")

    main = Config()
    main.backend.name, main.backend.model, main.backend.adapter = "mock", "", ""
    main.hardware.vram_gb = 8.0
    derived = target_config(main, parse_target("big=Qwen/Qwen3.5-4B"))
    assert (derived.backend.name, derived.backend.model, derived.backend.quant) == ("semif", "Qwen/Qwen3.5-4B", "nf4")


def test_targets_without_a_name_get_one_from_what_they_run() -> None:
    from omj.ui.targets import parse_targets

    specs = parse_targets([
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "semif",
        "semif:/runs/adapters/example-intent-en/20260929T010203Z/best",
        "mock",
        "mock",
    ])
    assert [s.name for s in specs] == ["qwen3.5-0.8b", "qwen3.5-2b", "semif", "example-intent-en", "mock", "mock-2"]
    assert specs[0].model == "Qwen/Qwen3.5-0.8B" and specs[3].backend == "semif"


def test_explicit_names_still_work_and_must_be_unique() -> None:
    from omj.ui.targets import parse_targets

    assert [s.name for s in parse_targets(["small=Qwen/Qwen3.5-0.8B", "jev=typesafe"])] == ["small", "jev"]
    with pytest.raises(OmjError):
        parse_targets(["a=mock", "a=typesafe"])


@pytest.mark.parametrize("bad", ["notabackend", "semif-typo:x", "a/b/c"])
def test_unnamed_targets_must_be_a_model_backend_or_config(bad: str) -> None:
    with pytest.raises(OmjError):
        parse_target(bad)
