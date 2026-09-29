"""`omj init` end-to-end behavior: detect -> select -> smoke -> config -> snippets.

REQ-004, REQ-005, REQ-006, REQ-007, REQ-010
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from omj.backends.base import Capabilities, Health, RawAnswer
from omj.cli import init_cmd
from omj.cli.init_cmd import InitOptions, run_init
from omj.cli.main import app
from omj.config import config_path, load_config
from omj.errors import OmjError
from omj.hw.detect import Probe

runner = CliRunner()


class _CpuNoKeyProbe(Probe):
    """No GPU, modest RAM: forces the matrix's "cpu, no key" branch unless overridden."""

    def nvidia_smi(self) -> str | None:
        return None

    def platform_name(self) -> str:
        return "linux"

    def total_ram_bytes(self) -> int | None:
        return 16 * 1024**3

    def python_version(self) -> str:
        return "3.11.9"


class _FaultyBackend:
    """Backend that returns a raw probability vector summing to 1.2 for noul.

    Simulates a broken kev/typesafe-style backend that hands back
    probabilities directly (no logits) but violates the Sigma p = 1
    contract, which `_run_smoke_question` must catch before
    `assemble_answer` would otherwise silently renormalize it away.
    """

    name = "faulty"
    model_id = "faulty-v1"
    capabilities = Capabilities(
        max_options=255,
        max_state_tokens=65536,
        supports_batch=True,
        calibrated=True,
        device="cpu",
    )

    def load(self, cfg) -> None:
        return None

    def decide(self, state, questions) -> dict[str, RawAnswer]:
        answers: dict[str, RawAnswer] = {}
        for qid, question in questions.items():
            kind = question["type"]
            if kind == "noul":
                answers[qid] = RawAnswer(qid=qid, kind="noul", keys=["yes", "no"], probs=[0.7, 0.5])
            elif kind == "choice":
                keys = list(question["criteria"].keys())
                probs = [1.0 / len(keys)] * len(keys)
                answers[qid] = RawAnswer(qid=qid, kind="choice", keys=keys, probs=probs)
            else:
                keys = [str(i) for i in range(len(question["criteria"]))]
                probs = [1.0 / len(keys)] * len(keys)
                answers[qid] = RawAnswer(qid=qid, kind="score", keys=keys, probs=probs)
        return answers

    def health(self) -> Health:
        return Health(ok=True)

    def count_tokens(self, state, questions) -> int:
        return 1


def _combined_output(result) -> str:
    output = result.stdout
    try:
        output += result.stderr
    except ValueError:
        pass  # stdout/stderr were mixed into a single stream
    return output


def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEV_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)


def test_run_init_mock_backend_writes_config_and_returns_smoke(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # REQ-004, REQ-006
    _clear_provider_keys(monkeypatch)
    opts = InitOptions(backend="mock", yes=True, json=True)
    out = io.StringIO()

    result = run_init(opts, probe=_CpuNoKeyProbe(), out=out)

    assert result.ok is True
    assert result.backend == "mock"
    assert result.endpoint == "http://127.0.0.1:8799/v1/systemone"
    assert set(result.smoke) == {"noul", "choice", "score"}
    assert result.smoke["noul"]["type"] == "noul"
    assert 0.0 <= result.smoke["noul"]["noul"] <= 1.0
    assert result.smoke["choice"]["type"] == "choice"
    assert abs(sum(result.smoke["choice"]["probabilities"].values()) - 1.0) < 1e-6
    assert result.smoke["score"]["type"] == "score"
    assert abs(sum(result.smoke["score"]["probabilities"].values()) - 1.0) < 1e-6

    cfg = load_config(config_path())
    assert cfg.backend.name == "mock"
    assert cfg.serve.port == 8799
    assert cfg.hardware.device == "cpu"


def test_run_init_json_mode_prints_nothing(monkeypatch: pytest.MonkeyPatch, omj_home: Path) -> None:
    # REQ-007
    _clear_provider_keys(monkeypatch)
    out = io.StringIO()
    opts = InitOptions(backend="mock", yes=True, json=True)

    run_init(opts, probe=_CpuNoKeyProbe(), out=out)

    assert out.getvalue() == ""


def test_run_init_schema_violation_raises_e_schema(monkeypatch: pytest.MonkeyPatch, omj_home: Path) -> None:
    # REQ-005
    _clear_provider_keys(monkeypatch)
    opts = InitOptions(backend="mock", yes=True)

    with pytest.raises(OmjError) as excinfo:
        run_init(
            opts,
            probe=_CpuNoKeyProbe(),
            backend_factory=lambda name: _FaultyBackend(),
            out=io.StringIO(),
        )

    assert excinfo.value.code.value == "E_SCHEMA"
    assert "smoke noul" in excinfo.value.message


def test_cli_init_mock_backend_succeeds(monkeypatch: pytest.MonkeyPatch, omj_home: Path) -> None:
    # REQ-004, REQ-006, REQ-010
    _clear_provider_keys(monkeypatch)
    monkeypatch.setattr(init_cmd, "SystemProbe", _CpuNoKeyProbe)

    result = runner.invoke(app, ["init", "--backend", "mock", "--yes"])

    assert result.exit_code == 0, _combined_output(result)
    assert "http://127.0.0.1:8799/v1/systemone" in result.stdout
    assert "typesafe_sdk" in result.stdout
    assert "TypeSafeClient" in result.stdout
    assert "@ai-sdk/gateway" in result.stdout
    assert "createGateway" in result.stdout

    cfg = load_config(config_path())
    assert cfg.backend.name == "mock"
    assert cfg.serve.port == 8799


def test_cli_init_json_mode_single_object(monkeypatch: pytest.MonkeyPatch, omj_home: Path) -> None:
    # REQ-007
    _clear_provider_keys(monkeypatch)
    monkeypatch.setattr(init_cmd, "SystemProbe", _CpuNoKeyProbe)

    result = runner.invoke(app, ["init", "--backend", "mock", "--yes", "--json"])

    assert result.exit_code == 0, _combined_output(result)
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["backend"] == "mock"
    assert payload["endpoint"] == "http://127.0.0.1:8799/v1/systemone"
    assert set(payload["smoke"]) == {"noul", "choice", "score"}
    assert payload["errors"] == []


def test_cli_init_schema_violation_exits_one(monkeypatch: pytest.MonkeyPatch, omj_home: Path) -> None:
    # REQ-005
    _clear_provider_keys(monkeypatch)
    monkeypatch.setattr(init_cmd, "SystemProbe", _CpuNoKeyProbe)
    monkeypatch.setattr(init_cmd, "create_backend", lambda name: _FaultyBackend())

    result = runner.invoke(app, ["init", "--backend", "mock", "--yes"])

    assert result.exit_code == 1
    assert "E_SCHEMA" in _combined_output(result)


def test_cli_init_schema_violation_json_mode_ok_false(monkeypatch: pytest.MonkeyPatch, omj_home: Path) -> None:
    # REQ-005, REQ-007
    _clear_provider_keys(monkeypatch)
    monkeypatch.setattr(init_cmd, "SystemProbe", _CpuNoKeyProbe)
    monkeypatch.setattr(init_cmd, "create_backend", lambda name: _FaultyBackend())

    result = runner.invoke(app, ["init", "--backend", "mock", "--yes", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("E_SCHEMA" in err for err in payload["errors"])


def test_cli_init_port_option_reflected_in_config_and_endpoint(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # REQ-006
    _clear_provider_keys(monkeypatch)
    monkeypatch.setattr(init_cmd, "SystemProbe", _CpuNoKeyProbe)

    result = runner.invoke(app, ["init", "--backend", "mock", "--yes", "--port", "9001"])

    assert result.exit_code == 0, _combined_output(result)
    assert "http://127.0.0.1:9001/v1/systemone" in result.stdout

    cfg = load_config(config_path())
    assert cfg.serve.port == 9001


def test_cli_init_model_override_reflected_in_config(monkeypatch: pytest.MonkeyPatch, omj_home: Path) -> None:
    # REQ-006
    _clear_provider_keys(monkeypatch)
    monkeypatch.setattr(init_cmd, "SystemProbe", _CpuNoKeyProbe)

    result = runner.invoke(
        app, ["init", "--backend", "mock", "--model", "custom-mock-model", "--yes"]
    )

    assert result.exit_code == 0, _combined_output(result)
    cfg = load_config(config_path())
    assert cfg.backend.model == "custom-mock-model"


def test_cli_init_no_gpu_no_key_without_backend_raises_e_no_gpu(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # REQ-002 (matrix precondition exercised through the init CLI surface)
    _clear_provider_keys(monkeypatch)
    monkeypatch.setattr(init_cmd, "SystemProbe", _CpuNoKeyProbe)

    result = runner.invoke(app, ["init", "--yes"])

    assert result.exit_code == 1
    assert "E_NO_GPU" in _combined_output(result)


class _SmallGpuProbe(Probe):
    """8 GB CUDA card: the matrix's semif Qwen3.5-2B branch."""

    def nvidia_smi(self) -> str | None:
        return "NVIDIA GeForce RTX 4060 Laptop GPU, 8188"

    def platform_name(self) -> str:
        return "win32"

    def total_ram_bytes(self) -> int | None:
        return 32 * 1024**3

    def python_version(self) -> str:
        return "3.11.9"


def _mock_factory(name: str):
    from omj.backends.mock import MockBackend

    return MockBackend()


CUSTOM_JEV_ENV = "MY_JEV" + "_KEY"
CUSTOM_OR_ENV = "MY_OR" + "_KEY"
FAKE_PROVIDER_VALUE = "sk-live-" + "abcdefgh1234"


def test_run_init_custom_api_key_env_is_treated_as_a_typesafe_key(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # REQ-005: --api-key-env names the variable that actually holds the key.
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv(CUSTOM_JEV_ENV, FAKE_PROVIDER_VALUE)
    opts = InitOptions(yes=True, json=True)
    opts.api_key_env = CUSTOM_JEV_ENV

    result = run_init(
        opts, probe=_CpuNoKeyProbe(), backend_factory=_mock_factory, out=io.StringIO()
    )

    assert result.backend == "typesafe"
    cfg = load_config(config_path())
    assert cfg.backend.name == "typesafe"
    assert cfg.backend.api_key_env == CUSTOM_JEV_ENV
    assert cfg.backend.provider == "typesafe"


def test_run_init_provider_openrouter_with_custom_key_env(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # REQ-005
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv(CUSTOM_OR_ENV, FAKE_PROVIDER_VALUE)
    opts = InitOptions(provider="openrouter", yes=True, json=True)
    opts.api_key_env = CUSTOM_OR_ENV

    result = run_init(
        opts, probe=_CpuNoKeyProbe(), backend_factory=_mock_factory, out=io.StringIO()
    )

    assert result.backend == "typesafe"
    cfg = load_config(config_path())
    assert cfg.backend.provider == "openrouter"
    assert cfg.backend.api_key_env == CUSTOM_OR_ENV


def test_run_init_provider_openrouter_defaults_key_env_to_openrouter_key(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # REQ-005
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENROUTER_KEY", FAKE_PROVIDER_VALUE)
    opts = InitOptions(provider="openrouter", yes=True, json=True)

    run_init(opts, probe=_CpuNoKeyProbe(), backend_factory=_mock_factory, out=io.StringIO())

    cfg = load_config(config_path())
    assert cfg.backend.provider == "openrouter"
    assert cfg.backend.api_key_env == "OPENROUTER_KEY"


def test_run_init_unknown_provider_raises_e_config(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # REQ-005
    _clear_provider_keys(monkeypatch)
    opts = InitOptions(backend="mock", provider="anthropic", yes=True, json=True)

    with pytest.raises(OmjError) as excinfo:
        run_init(opts, probe=_CpuNoKeyProbe(), backend_factory=_mock_factory, out=io.StringIO())

    assert excinfo.value.code.value == "E_CONFIG"
    assert "--provider" in excinfo.value.message


def test_run_init_no_download_records_offline_and_skips_the_downloader(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # REQ-005: --no-download must also stop the backend's own implicit download.
    _clear_provider_keys(monkeypatch)
    calls: list[tuple] = []

    def fake_downloader(repo_id: str, revision: str, local_dir: str) -> str:
        calls.append((repo_id, revision, local_dir))
        return local_dir

    opts = InitOptions(backend="semif", no_download=True, yes=True, json=True)
    run_init(
        opts,
        probe=_SmallGpuProbe(),
        downloader=fake_downloader,
        backend_factory=_mock_factory,
        out=io.StringIO(),
        local_available=lambda: True,
    )

    assert calls == []
    cfg = load_config(config_path())
    assert cfg.backend.name == "semif"
    assert cfg.backend.offline is True


def test_run_init_download_path_leaves_offline_false(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # REQ-005, REQ-009
    _clear_provider_keys(monkeypatch)
    monkeypatch.delenv("OMJ_NO_NETWORK", raising=False)
    calls: list[tuple] = []

    def fake_downloader(repo_id: str, revision: str, local_dir: str) -> str:
        calls.append((repo_id, revision, local_dir))
        return local_dir

    opts = InitOptions(backend="semif", yes=True, json=True)
    run_init(
        opts,
        probe=_SmallGpuProbe(),
        downloader=fake_downloader,
        backend_factory=_mock_factory,
        out=io.StringIO(),
        local_available=lambda: True,
    )

    assert len(calls) == 1
    assert calls[0][0] == "Qwen/Qwen3.5-2B"
    cfg = load_config(config_path())
    assert cfg.backend.offline is False


def _no_download(repo_id: str, revision: str, local_dir: str) -> str:
    raise AssertionError("must not download when the semif extra is missing")


def test_run_init_without_semif_extra_falls_back_to_mock_and_skips_download(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    # An 8 GB GPU auto-selects semif; without torch/transformers that would
    # download GB of weights and then fail at import time.
    _clear_provider_keys(monkeypatch)
    out = io.StringIO()
    result = run_init(
        InitOptions(yes=True),
        probe=_SmallGpuProbe(),
        downloader=_no_download,
        backend_factory=_mock_factory,
        out=out,
        local_available=lambda: False,
    )

    assert result.backend == "mock"
    assert load_config(config_path()).backend.name == "mock"
    assert "uv sync --extra semif" in out.getvalue()


def test_run_init_without_semif_extra_prefers_remote_backend_when_a_key_exists(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENROUTER_KEY", FAKE_PROVIDER_VALUE)
    result = run_init(
        InitOptions(yes=True, json=True),
        probe=_SmallGpuProbe(),
        downloader=_no_download,
        backend_factory=_mock_factory,
        out=io.StringIO(),
        local_available=lambda: False,
    )

    assert result.backend == "typesafe"
    assert load_config(config_path()).backend.provider == "openrouter"
    # Audit A6: the key variable must follow the auto-picked provider.
    assert load_config(config_path()).backend.api_key_env == "OPENROUTER_KEY"


def test_run_init_explicit_semif_without_extra_fails_before_download(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    _clear_provider_keys(monkeypatch)
    with pytest.raises(OmjError) as err:
        run_init(
            InitOptions(backend="semif", yes=True, json=True),
            probe=_SmallGpuProbe(),
            downloader=_no_download,
            backend_factory=_mock_factory,
            out=io.StringIO(),
            local_available=lambda: False,
        )
    assert "uv sync --extra semif" in str(err.value)


def test_run_init_cpu_with_only_an_openrouter_key_stores_that_key_variable(
    monkeypatch: pytest.MonkeyPatch, omj_home: Path
) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENROUTER_KEY", FAKE_PROVIDER_VALUE)
    result = run_init(
        InitOptions(yes=True, json=True), probe=_CpuNoKeyProbe(), backend_factory=_mock_factory, out=io.StringIO()
    )

    cfg = load_config(config_path())
    assert result.backend == "typesafe"
    assert (cfg.backend.provider, cfg.backend.api_key_env) == ("openrouter", "OPENROUTER_KEY")
