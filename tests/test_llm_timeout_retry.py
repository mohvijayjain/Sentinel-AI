"""
NVIDIA LLM call is bounded: explicit timeout, SDK retries disabled, at most
NVIDIA_LLM_MAX_RETRIES classified retries, worst case <= 60s.

Defaults: 25s * (1 + 1) + 1 * 2s = 52s.

Before, OpenAI() was built with the SDK defaults (600s timeout, 2 automatic
retries); a live /chat took ~5.5 minutes. Deterministic: fake client, fake
sleep, no network, no waiting. Secrets are fake.
"""

import importlib.util
import logging
import os
import sys

import httpx
import openai
import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FAKE_KEY = "nvapi-FAKEfakeFAKEfake1234567890wxyz"
REQUEST = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO_ROOT, *relative.split("/"))
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_llm(monkeypatch):
    """Fresh config + llm modules for the given env (no .env, fake key)."""

    def _factory(**env):
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        for key in ("NVIDIA_LLM_TIMEOUT_SECONDS", "NVIDIA_LLM_MAX_RETRIES",
                    "NVIDIA_LLM_RETRY_DELAY_SECONDS"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))

        config = _load("sentinel_test_rag_config", "src/rag/config.py")
        monkeypatch.setitem(sys.modules, "src.rag.config", config)
        llm = _load("sentinel_test_rag_llm", "src/rag/llm.py")
        return config, llm

    return _factory


class FakeClient:
    """chat.completions.create() replays scripted outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome

        class Message:
            content = f"  {outcome}  "

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


def _status(cls, code, text="error"):
    return cls(
        f"Error code: {code} - {text}",
        response=httpx.Response(code, request=REQUEST),
        body=None,
    )


def _ask(llm, client, sleeps):
    return llm.generate_answer(
        "Why was retraining rejected?", "Retraining Event 2: REJECTED.",
        client=client, sleep=sleeps.append,
    )


# ============================================================
# Ceiling and configuration
# ============================================================

def test_default_worst_case_is_52s_and_under_ceiling(load_llm):

    config, _ = load_llm()

    assert (config.NVIDIA_LLM_TIMEOUT_SECONDS, config.NVIDIA_LLM_MAX_RETRIES,
            config.NVIDIA_LLM_RETRY_DELAY_SECONDS) == (25.0, 1, 2.0)
    assert config.llm_worst_case_seconds() == 25 * (1 + 1) + 1 * 2 == 52
    assert config.llm_worst_case_seconds() <= config.LLM_LATENCY_CEILING_SECONDS == 60


def test_client_timeout_is_explicit_and_sdk_retries_disabled(load_llm):

    _, llm = load_llm()

    assert llm._client.max_retries == 0
    assert llm._client.timeout.read == 25.0
    assert llm._client.timeout.connect == 5.0


def test_configured_timeout_respected(load_llm):

    config, llm = load_llm(NVIDIA_LLM_TIMEOUT_SECONDS=10, NVIDIA_LLM_MAX_RETRIES=2,
                           NVIDIA_LLM_RETRY_DELAY_SECONDS=1)

    assert llm._client.timeout.read == 10.0
    assert config.llm_worst_case_seconds() == 10 * 3 + 2 * 1 == 32


@pytest.mark.parametrize(
    "env",
    [
        {"NVIDIA_LLM_TIMEOUT_SECONDS": 40},                        # 40*2+2 = 82
        {"NVIDIA_LLM_MAX_RETRIES": 3},                             # 25*4+6 = 106
        {"NVIDIA_LLM_TIMEOUT_SECONDS": 29, "NVIDIA_LLM_RETRY_DELAY_SECONDS": 3},  # 61
    ],
)
def test_override_above_ceiling_fails_fast(load_llm, env):

    with pytest.raises(ValueError, match="exceeds the 60s ceiling"):
        load_llm(**env)


@pytest.mark.parametrize(
    "env",
    [{"NVIDIA_LLM_TIMEOUT_SECONDS": 0}, {"NVIDIA_LLM_MAX_RETRIES": -1},
     {"NVIDIA_LLM_RETRY_DELAY_SECONDS": -1}],
)
def test_invalid_values_rejected(load_llm, env):

    with pytest.raises(ValueError):
        load_llm(**env)


# ============================================================
# Retry behaviour
# ============================================================

def test_first_attempt_success(load_llm):

    _, llm = load_llm()
    client, sleeps = FakeClient(["answer"]), []

    assert _ask(llm, client, sleeps) == "answer"
    assert client.calls == 1 and sleeps == []


@pytest.mark.parametrize(
    "transient",
    [
        openai.APITimeoutError(request=REQUEST),
        openai.APIConnectionError(request=REQUEST),
        _status(openai.RateLimitError, 429),
        _status(openai.InternalServerError, 503),
        _status(openai.InternalServerError, 502),
    ],
    ids=["timeout", "connection", "429", "503", "502"],
)
def test_transient_then_success(load_llm, transient):

    _, llm = load_llm()
    client, sleeps = FakeClient([transient, "recovered"]), []

    assert _ask(llm, client, sleeps) == "recovered"
    assert client.calls == 2
    assert sleeps == [2.0]


def test_repeated_failure_stops_at_max_retries(load_llm):

    _, llm = load_llm(NVIDIA_LLM_MAX_RETRIES=2, NVIDIA_LLM_TIMEOUT_SECONDS=15)
    last = openai.APITimeoutError(request=REQUEST)
    client = FakeClient([openai.APITimeoutError(request=REQUEST),
                         openai.APITimeoutError(request=REQUEST), last, "never"])
    sleeps = []

    with pytest.raises(openai.APITimeoutError) as excinfo:
        _ask(llm, client, sleeps)

    assert excinfo.value is last             # original exception, unchanged
    assert client.calls == 3                 # 1 + max_retries, no more
    assert sleeps == [2.0, 2.0]


@pytest.mark.parametrize(
    "permanent",
    [
        _status(openai.AuthenticationError, 401, "invalid api key"),
        _status(openai.PermissionDeniedError, 403),
        _status(openai.BadRequestError, 400, "malformed request"),
        _status(openai.NotFoundError, 404, "model not found"),
        _status(openai.InternalServerError, 501),
    ],
    ids=["401-auth", "403", "400-malformed", "404-invalid-model", "501"],
)
def test_permanent_errors_do_not_retry(load_llm, permanent):

    _, llm = load_llm()
    client, sleeps = FakeClient([permanent, "never"]), []

    with pytest.raises(type(permanent)) as excinfo:
        _ask(llm, client, sleeps)

    assert excinfo.value is permanent
    assert client.calls == 1
    assert sleeps == []


def test_no_fake_answer_on_exhaustion(load_llm):

    _, llm = load_llm()
    client = FakeClient([openai.APITimeoutError(request=REQUEST),
                         openai.APITimeoutError(request=REQUEST)])

    with pytest.raises(openai.APITimeoutError):
        _ask(llm, client, [])


# ============================================================
# Logs are redacted
# ============================================================

def test_failure_logs_never_contain_the_key(load_llm, caplog):

    _, llm = load_llm()
    leaky = _status(
        openai.AuthenticationError, 401,
        f"Incorrect API key provided: {FAKE_KEY}; Authorization: Bearer {FAKE_KEY}",
    )
    client = FakeClient([_status(openai.InternalServerError, 503,
                                 f"upstream rejected api_key={FAKE_KEY}"), leaky])

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(openai.AuthenticationError):
            _ask(llm, client, [])

    assert FAKE_KEY not in caplog.text
    assert "retrying in 2.0s" in caplog.text
    assert "not retryable" in caplog.text
    assert "AuthenticationError" in caplog.text


# ============================================================
# Existing "information unavailable" instruction preserved
# ============================================================

def test_unavailable_instruction_still_sent(load_llm):

    _, llm = load_llm()
    captured = {}

    class Capturing(FakeClient):
        def create(self, **kwargs):
            captured.update(kwargs)
            return super().create(**kwargs)

    _ask(llm, Capturing(["ok"]), [])

    system = captured["messages"][0]["content"]
    assert "say that the information is unavailable" in system
    assert captured["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
