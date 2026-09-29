"""Official typesafe-sdk-python against the gateway, changing only base_url.
# REQ-030
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from typesafe_sdk import Choice, Noul, Score, TypeSafeAuthenticationError, TypeSafeClient

from omj.backends.mock import MockBackend
from omj.config import Config, LogSection, ServeSection
from omj.gateway.app import create_app
from omj.gateway.decision_log import DecisionLogger

READY_TIMEOUT_S = 5.0

# Synthetic placeholders used only by these in-process tests; the mock backend
# never checks these values against anything real.
UNUSED_KEY = "omj-placeholder-unused-key"
SERVER_TOKEN = "omj-placeholder-sdk-token"
WRONG_TOKEN = "omj-placeholder-wrong-token"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_app(app, tmp_path_factory: pytest.TempPathFactory, name: str) -> Iterator[str]:
    log_dir = tmp_path_factory.mktemp(name)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    # uvicorn.Server.run() skips signal-handler setup off the main thread, so a
    # daemon thread is a safe way to host it for the lifetime of this module.
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + READY_TIMEOUT_S
    with httpx.Client() as probe:
        while time.monotonic() < deadline:
            try:
                if probe.get(f"{base_url}/health", timeout=0.2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.02)
        else:
            raise RuntimeError(f"{name}: gateway did not become healthy within {READY_TIMEOUT_S}s")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=READY_TIMEOUT_S)
    _ = log_dir  # kept alive for the fixture's lifetime; DecisionLogger writes under it


@pytest.fixture(scope="module")
def gateway_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    log_dir = tmp_path_factory.mktemp("omj-sdk-open")
    app = create_app(
        MockBackend(),
        Config(log=LogSection(dir=str(log_dir))),
        decision_logger=DecisionLogger(log_dir=log_dir),
    )
    yield from _run_app(app, tmp_path_factory, "omj-sdk-open-run")


@pytest.fixture(scope="module")
def auth_gateway_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    log_dir = tmp_path_factory.mktemp("omj-sdk-auth")
    app = create_app(
        MockBackend(),
        Config(serve=ServeSection(api_key=SERVER_TOKEN), log=LogSection(dir=str(log_dir))),
        decision_logger=DecisionLogger(log_dir=log_dir),
    )
    yield from _run_app(app, tmp_path_factory, "omj-sdk-auth-run")


@pytest.fixture(scope="module")
def sdk_client(gateway_url: str) -> Iterator[TypeSafeClient]:
    with TypeSafeClient(api_key=UNUSED_KEY, base_url=gateway_url) as client:
        yield client


def test_noul_question(sdk_client: TypeSafeClient) -> None:
    result = sdk_client.system_one(
        state="I was charged twice for the same order. Please help.",
        model="jev-latest",
        questions={"billing": Noul(instructions="Is this message about a billing problem?")},
    )

    answer = result.nouls["billing"]
    assert 0.0 <= answer.noul <= 1.0


def test_choice_question(sdk_client: TypeSafeClient) -> None:
    result = sdk_client.system_one(
        state="This is an angry and hostile message demanding a refund.",
        model="jev-latest",
        questions={
            "tone": Choice(
                instructions="What is the tone of this message?",
                criteria={
                    "angry": "an upset or hostile message",
                    "calm": "a neutral or polite message",
                },
            )
        },
    )

    answer = result.choices["tone"]
    assert answer.choice in {"angry", "calm"}
    assert 0.0 <= answer.confidence <= 1.0
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_score_question(sdk_client: TypeSafeClient) -> None:
    result = sdk_client.system_one(
        state="This needs attention today, it is very urgent.",
        model="jev-latest",
        questions={
            "urgency": Score(
                instructions="How urgent is this message?",
                criteria=["can wait", "needs attention this week", "needs attention today"],
            )
        },
    )

    answer = result.scores["urgency"]
    assert 0.0 <= answer.score <= 2.0
    assert 0.0 <= answer.confidence <= 1.0
    assert set(answer.legend.keys()) == {0, 1, 2}
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_combined_three_question_request(sdk_client: TypeSafeClient) -> None:
    result = sdk_client.system_one(
        state="I was charged twice and I am furious, this needs attention today.",
        model="jev-latest",
        questions={
            "billing": Noul(instructions="Is this message about a billing problem?"),
            "tone": Choice(
                instructions="What is the tone of this message?",
                criteria={
                    "angry": "an upset or hostile message",
                    "calm": "a neutral or polite message",
                },
            ),
            "urgency": Score(
                instructions="How urgent is this message?",
                criteria=["can wait", "needs attention this week", "needs attention today"],
            ),
        },
    )

    assert set(result.answers) == {"billing", "tone", "urgency"}
    assert 0.0 <= result.nouls["billing"].noul <= 1.0
    assert result.choices["tone"].choice in {"angry", "calm"}
    assert sum(result.choices["tone"].probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= result.scores["urgency"].score <= 2.0
    assert sum(result.scores["urgency"].probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_wrong_api_key_raises_sdk_authentication_error(auth_gateway_url: str) -> None:
    with TypeSafeClient(api_key=WRONG_TOKEN, base_url=auth_gateway_url) as client:
        with pytest.raises(TypeSafeAuthenticationError):
            client.system_one(
                state="hello there",
                model="jev-latest",
                questions={"greeting": Noul(instructions="Is this a greeting?")},
            )


def test_correct_api_key_succeeds(auth_gateway_url: str) -> None:
    with TypeSafeClient(api_key=SERVER_TOKEN, base_url=auth_gateway_url) as client:
        result = client.system_one(
            state="hello there",
            model="jev-latest",
            questions={"greeting": Noul(instructions="Is this a greeting?")},
        )

    assert 0.0 <= result.nouls["greeting"].noul <= 1.0
