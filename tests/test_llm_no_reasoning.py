"""
/chat answers must not wait on the model's hidden reasoning.

nvidia/nemotron-3.5-lightning-30b-a3b reasons before answering by default.
On "Why was the challenger rejected?" that took 1,200-1,800 completion
tokens and 39-46s, past the 25s per-attempt timeout: both attempts raised
APITimeoutError and /chat returned 500. Every request now sends
chat_template_kwargs.enable_thinking = False (measured: ~1-2s, same
grounded answer). Model, prompt, timeout and retry policy are unchanged.
"""

from test_llm_timeout_retry import FakeClient, _ask, load_llm  # noqa: F401


class Capturing(FakeClient):

    def __init__(self, outcomes):
        super().__init__(outcomes)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return super().create(**kwargs)


def test_reasoning_disabled_on_every_request(load_llm):

    _, llm = load_llm()
    client = Capturing(["answer"])

    assert _ask(llm, client, []) == "answer"

    [request] = client.requests
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_reasoning_disabled_on_retries_too(load_llm):

    import httpx
    from openai import APITimeoutError

    _, llm = load_llm()
    timeout = APITimeoutError(request=httpx.Request("POST", "https://x.test"))
    client = Capturing([timeout, "answer"])

    assert _ask(llm, client, []) == "answer"

    assert len(client.requests) == 2
    assert all(r["extra_body"] == llm.NO_REASONING for r in client.requests)


def test_request_otherwise_unchanged(load_llm):

    _, llm = load_llm()
    client = Capturing(["answer"])

    _ask(llm, client, [])

    [request] = client.requests
    assert set(request) == {"model", "messages", "temperature", "extra_body"}
    assert request["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert request["temperature"] == 0.2
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
