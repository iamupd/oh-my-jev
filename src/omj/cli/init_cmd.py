"""`omj init`: detect hardware, pick a backend, download it, and write config.toml.

Flow (REQ-004, REQ-005, REQ-006, REQ-007, REQ-010):
  detect -> read JEV_KEY/OPENROUTER_KEY presence -> select_backend(overrides)
  -> download the model when the backend needs one -> backend.load(...)
  -> run 3 fixed smoke requests (noul, choice, score), validating schema and
     probability mass -> save config.toml -> print endpoint + SDK snippets.

`run_init` never prints in --json mode and always raises OmjError on any
failure so both the CliRunner surface (via main.py's OmjGroup, non-json) and
direct unit-style calls (pytest.raises) see the same behavior. The `init`
command itself only translates that into the single-JSON-object contract
required by REQ-007 when --json is passed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, replace
from typing import Callable, TextIO

import typer

from omj.backends.base import Backend
from omj.backends.registry import create_backend
from omj.cli.snippets import python_snippet, typescript_snippet
from omj.config import (
    BackendSection,
    Config,
    HardwareSection,
    LogSection,
    ServeSection,
    UsageSection,
    save_config,
)
from omj.errors import ErrorCode, OmjError
from omj.gateway.assemble import assemble_answer, softmax
from omj.gateway.schema import Question, answer_keys
from omj.hw.detect import Probe, SystemProbe, detect
from omj.hw.matrix import Overrides, Selection, select_backend
from omj.models.download import Downloader, download_model, hf_snapshot_download

DEFAULT_PORT = 8799
JEV_KEY_ENV = "JEV_KEY"
OPENROUTER_KEY_ENV = "OPENROUTER_KEY"
_VALID_QUANTS = ("bf16", "4bit-prequant")
_VALID_PROVIDERS = ("typesafe", "openrouter")

# Fixed smoke payloads (REQ-004): one state/instructions/criteria triple per
# question kind, in the order they are asked. criteria is None for noul
# (yes/no needs none), a dict for choice, a list for score.
_SMOKE_QUESTIONS: dict[str, tuple[str, str, dict[str, str] | list[str] | None]] = {
    "noul": (
        "2 + 2 = 4",
        "Is the statement true?",
        None,
    ),
    "choice": (
        "A customer writes: 'My invoice shows a duplicate charge from last month, please help.'",
        "Which team should this support ticket be routed to?",
        {
            "billing": "Payment, invoice, and refund issues",
            "technical": "Product bugs and technical errors",
            "sales": "Pricing and new purchase questions",
            "general": "Anything that does not fit the other categories",
        },
    ),
    "score": (
        "The production database is completely down and no customers can check out.",
        "Rate the severity of this incident.",
        ["Low", "Medium", "High", "Critical"],
    ),
}


@dataclass
class InitOptions:
    backend: str | None = None
    model: str | None = None
    quant: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    provider: str | None = None
    port: int = DEFAULT_PORT
    no_download: bool = False
    yes: bool = False
    json: bool = False


@dataclass
class InitResult:
    ok: bool
    backend: str
    model: str
    endpoint: str
    smoke: dict[str, dict]


def _mask(value: str) -> str:
    # Never let a secret reach stdout/stderr/exceptions in full (repo rule).
    if len(value) <= 4:
        return "****"
    return f"{value[:4]}****"


# Backends that run a model in this process and need the optional ``semif`` extra.
_LOCAL_BACKENDS = ("semif", "kev")
_LOCAL_EXTRA_HINT = "uv sync --extra semif"


def _local_extra_installed() -> bool:
    return all(importlib.util.find_spec(mod) is not None for mod in ("torch", "transformers"))


def _fallback_without_local_extra(
    selection: Selection, has_typesafe_key: bool, has_openrouter_key: bool
) -> Selection:
    """Replace an auto-selected local backend when the ``semif`` extra is missing.

    Downloading several GB of weights only to fail at import time is the worst
    first-run experience, so fall back to the remote backend when a key exists,
    otherwise to the mock backend, and tell the user how to enable local models.
    """
    why = f"{selection.backend} needs the semif extra ('{_LOCAL_EXTRA_HINT}')"
    if has_typesafe_key or has_openrouter_key:
        provider = "typesafe" if has_typesafe_key else "openrouter"
        return replace(
            selection, backend="typesafe", model="jev-latest", revision="", quant="bf16",
            reason=f"{why}; using {provider} backend",
        )
    return replace(
        selection, backend="mock", model="", revision="", quant="bf16",
        reason=f"{why}; no API key, using mock backend",
    )


def run_init(
    opts: InitOptions,
    *,
    probe: Probe | None = None,
    downloader: Downloader | None = None,
    backend_factory: Callable[[str], Backend] | None = None,
    out: TextIO | None = None,
    confirm: Callable[[str], bool] | None = None,
    local_available: Callable[[], bool] | None = None,
) -> InitResult:
    probe = probe if probe is not None else SystemProbe()
    downloader = downloader if downloader is not None else hf_snapshot_download
    backend_factory = backend_factory if backend_factory is not None else create_backend
    out = out if out is not None else sys.stdout
    confirm = confirm if confirm is not None else typer.confirm
    local_available = local_available if local_available is not None else _local_extra_installed

    def emit(line: str = "") -> None:
        if not opts.json:
            print(line, file=out)

    if opts.quant is not None and opts.quant not in _VALID_QUANTS:
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"--quant must be one of {_VALID_QUANTS}, got {opts.quant!r}",
        )

    if opts.provider is not None and opts.provider not in _VALID_PROVIDERS:
        raise OmjError(
            ErrorCode.E_CONFIG,
            f"--provider must be one of {_VALID_PROVIDERS}, got {opts.provider!r}",
        )

    hw = detect(probe)

    provider = opts.provider or "typesafe"
    default_key_env = OPENROUTER_KEY_ENV if provider == "openrouter" else JEV_KEY_ENV
    api_key_env_name = opts.api_key_env or default_key_env
    if opts.api_key:
        os.environ[api_key_env_name] = opts.api_key
        emit(
            f"Warning: --api-key value ({_mask(opts.api_key)}) was set on "
            f"{api_key_env_name} for this process only; it is not persisted "
            f"to config.toml. Export {api_key_env_name} yourself to reuse it."
        )

    # A key in the variable named by --api-key-env counts for the provider it
    # was given for; otherwise the matrix would report "no key configured".
    custom_key = bool(opts.api_key_env and os.environ.get(opts.api_key_env))
    has_typesafe_key = bool(os.environ.get(JEV_KEY_ENV)) or (
        custom_key and provider == "typesafe"
    )
    has_openrouter_key = bool(os.environ.get(OPENROUTER_KEY_ENV)) or (
        custom_key and provider == "openrouter"
    )

    overrides = Overrides(backend=opts.backend, model=opts.model, quant=opts.quant)
    selection = select_backend(hw, has_typesafe_key, has_openrouter_key, overrides)
    local_missing = selection.backend in _LOCAL_BACKENDS and not local_available()
    if local_missing and opts.backend is None and opts.model is None:
        auto_local = selection
        selection = _fallback_without_local_extra(selection, has_typesafe_key, has_openrouter_key)
    elif local_missing:
        # Explicit --backend/--model: fail before any download, not after it.
        raise OmjError(
            ErrorCode.E_BACKEND,
            f"backend {selection.backend!r} requires the 'semif' extra; run '{_LOCAL_EXTRA_HINT}'",
        )
    else:
        auto_local = None

    emit(
        f"Detected hardware: device={hw.device} vram_gb={hw.vram_gb} "
        f"ram_gb={hw.ram_gb} platform={hw.platform}"
    )
    emit(
        f"Selected backend={selection.backend} model={selection.model!r} "
        f"quant={selection.quant} ({selection.reason or 'override'})"
    )
    if auto_local is not None:
        emit(
            f"Note: this machine can run {auto_local.model} locally. To use it, run "
            f"'{_LOCAL_EXTRA_HINT}' and then 'uv run omj init' again."
        )

    if selection.backend == "semif" and not opts.no_download:
        # --json is for machine consumption; an interactive prompt would
        # corrupt the single-JSON-object stdout contract (REQ-007), so a
        # non-interactive run without --yes is treated as consent to
        # download rather than left to hang or fail unexpectedly.
        if not opts.yes and not opts.json:
            proceed = confirm(f"Download {selection.model}@{selection.revision or 'main'} now?")
            if not proceed:
                raise OmjError(
                    ErrorCode.E_DOWNLOAD,
                    "download declined; rerun with --yes or pass --no-download",
                )
        emit(f"Downloading {selection.model}@{selection.revision or 'main'}...")
        download_model(selection.model, selection.revision, downloader=downloader)

    if selection.backend == "typesafe" and opts.provider is None:
        provider = "typesafe" if has_typesafe_key else "openrouter"
        if opts.api_key_env is None:
            # The key variable follows the provider picked here; keeping the
            # JEV_KEY default for an OpenRouter-only setup fails with E_AUTH.
            api_key_env_name = OPENROUTER_KEY_ENV if provider == "openrouter" else JEV_KEY_ENV

    backend_section = BackendSection(
        name=selection.backend,
        model=selection.model,
        revision=selection.revision,
        quant=selection.quant,
        calibration="",
        provider=provider,
        api_key_env=api_key_env_name,
        offline=opts.no_download,
    )

    backend = backend_factory(selection.backend)
    backend.load(backend_section)

    smoke: dict[str, dict] = {}
    for kind, (state, instructions, criteria) in _SMOKE_QUESTIONS.items():
        smoke[kind] = _run_smoke_question(backend, kind, state, instructions, criteria)

    cfg = Config(
        hardware=HardwareSection(
            device=hw.device,
            vram_gb=hw.vram_gb,
            ram_gb=hw.ram_gb,
            platform=hw.platform,
            python=hw.python,
            gpu_name=hw.gpu_name,
        ),
        backend=backend_section,
        serve=ServeSection(port=opts.port),
        log=LogSection(),
        usage=UsageSection(),
    )
    save_config(cfg)

    origin = f"http://127.0.0.1:{opts.port}"
    endpoint = f"{origin}/v1/systemone"

    emit()
    emit(f"omj is ready. Endpoint: {endpoint}")
    emit()
    emit("Python:")
    emit(python_snippet(origin))
    emit("TypeScript:")
    emit(typescript_snippet(origin))

    return InitResult(
        ok=True,
        backend=selection.backend,
        model=selection.model,
        endpoint=endpoint,
        smoke=smoke,
    )


def _run_smoke_question(
    backend: Backend,
    kind: str,
    state: str,
    instructions: str,
    criteria: dict[str, str] | list[str] | None,
) -> dict:
    qid = kind
    question = Question(type=kind, instructions=instructions, criteria=criteria)
    questions_wire = {qid: question.model_dump()}

    try:
        raw_answers = backend.decide(state, questions_wire)
    except OmjError:
        raise
    except Exception as exc:
        raise OmjError(ErrorCode.E_SCHEMA, f"smoke {kind}: backend raised {exc!r}") from exc

    raw = raw_answers.get(qid)
    if raw is None:
        raise OmjError(ErrorCode.E_SCHEMA, f"smoke {kind}: backend returned no answer for {qid!r}")

    expected_keys = answer_keys(question)
    keys = list(raw.keys) if raw.keys else expected_keys
    if keys != expected_keys:
        raise OmjError(
            ErrorCode.E_SCHEMA,
            f"smoke {kind}: answer keys {keys} do not match expected {expected_keys}",
        )

    if raw.logits is not None:
        probs = softmax(raw.logits)
    else:
        # Backends that hand back probabilities directly (kev, typesafe) are
        # not temperature-scaled, so they must already be a valid
        # distribution: check before assemble_answer's own renormalization
        # would otherwise silently paper over a broken backend.
        probs = list(raw.probs or [])
        total = sum(probs)
        if abs(total - 1.0) > 1e-6:
            raise OmjError(
                ErrorCode.E_SCHEMA,
                f"smoke {kind}: backend probabilities sum to {total:.6f}, expected 1 +/- 1e-6",
            )

    answer = assemble_answer(question, keys, probs)
    _validate_smoke_answer(kind, answer, expected_keys)
    return answer


def _validate_smoke_answer(kind: str, answer: dict, expected_keys: list[str]) -> None:
    if answer.get("type") != kind:
        raise OmjError(
            ErrorCode.E_SCHEMA,
            f"smoke {kind}: answer type field is {answer.get('type')!r}, expected {kind!r}",
        )

    if kind == "noul":
        value = answer.get("noul")
        if not isinstance(value, (int, float)) or not (-1e-6 <= value <= 1.0 + 1e-6):
            raise OmjError(ErrorCode.E_SCHEMA, f"smoke noul: 'noul' value {value!r} is not in [0, 1]")
        return

    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(expected_keys):
        raise OmjError(
            ErrorCode.E_SCHEMA,
            f"smoke {kind}: 'probabilities' keys {sorted(probabilities or {})} != "
            f"expected {sorted(expected_keys)}",
        )

    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-6:
        raise OmjError(
            ErrorCode.E_SCHEMA,
            f"smoke {kind}: probabilities sum to {total:.6f}, expected 1 +/- 1e-6",
        )

    if "confidence" not in answer:
        raise OmjError(ErrorCode.E_SCHEMA, f"smoke {kind}: missing 'confidence' field")

    if kind == "choice":
        if answer.get("choice") not in expected_keys:
            raise OmjError(
                ErrorCode.E_SCHEMA,
                f"smoke choice: 'choice' value {answer.get('choice')!r} not among {expected_keys}",
            )
    elif kind == "score":
        if "score" not in answer:
            raise OmjError(ErrorCode.E_SCHEMA, "smoke score: missing 'score' field")
        legend = answer.get("legend")
        if not isinstance(legend, dict) or set(legend) != set(expected_keys):
            raise OmjError(ErrorCode.E_SCHEMA, "smoke score: 'legend' keys do not match expected keys")


def _empty_smoke() -> dict[str, None]:
    return {"noul": None, "choice": None, "score": None}


def register(app: typer.Typer) -> None:
    @app.command()
    def init(
        backend: str = typer.Option(None, "--backend", help="Force a backend (mock, semif, kev, typesafe)."),
        model: str = typer.Option(None, "--model", help="Override the selected model id."),
        quant: str = typer.Option(None, "--quant", help="Quantization: bf16 or 4bit-prequant."),
        api_key: str = typer.Option(
            None,
            "--api-key",
            help="Provider API key for this process only; never persisted to config.toml.",
        ),
        api_key_env: str = typer.Option(
            None,
            "--api-key-env",
            help="Environment variable to read the provider key from (default JEV_KEY).",
        ),
        provider: str = typer.Option(
            None,
            "--provider",
            help="Hosted provider for the typesafe backend: typesafe (default) or openrouter.",
        ),
        port: int = typer.Option(DEFAULT_PORT, "--port", help="Gateway port recorded in config.toml."),
        no_download: bool = typer.Option(False, "--no-download", help="Skip downloading model weights."),
        yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompts."),
        json_output: bool = typer.Option(False, "--json", help="Print a single machine-readable JSON object."),
    ) -> None:
        """Detect hardware, select a backend, download it, and write config.toml."""
        opts = InitOptions(
            backend=backend,
            model=model,
            quant=quant,
            api_key=api_key,
            api_key_env=api_key_env,
            provider=provider,
            port=port,
            no_download=no_download,
            yes=yes,
            json=json_output,
        )

        if not opts.json:
            run_init(opts)
            return

        try:
            result = run_init(opts)
        except OmjError as exc:
            payload = {
                "ok": False,
                "backend": backend or "",
                "model": model or "",
                "endpoint": f"http://127.0.0.1:{port}/v1/systemone",
                "smoke": _empty_smoke(),
                "errors": [f"{exc.code.value}: {exc.message}"],
            }
            typer.echo(json.dumps(payload))
            raise typer.Exit(code=1) from None

        payload = {
            "ok": True,
            "backend": result.backend,
            "model": result.model,
            "endpoint": result.endpoint,
            "smoke": result.smoke,
            "errors": [],
        }
        typer.echo(json.dumps(payload))
