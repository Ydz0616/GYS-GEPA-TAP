import httpx
import pytest

from gys.config import ModelConfig
from gys.llm import EmptyCompletionError, LLMClient, MockLLM, MockResponse, Usage


def _cfg(model="m", **kw):
    return ModelConfig(model=model, max_retries=2, **kw)


def test_mock_queue_and_recording():
    backend = MockLLM(["a", "b"])
    client = LLMClient(backend=backend, sleep=lambda _s: None)
    assert client.chat([{"role": "user", "content": "x"}], _cfg()) == "a"
    assert client.chat([{"role": "user", "content": "y"}], _cfg()) == "b"
    assert [r["messages"][0]["content"] for r in backend.requests] == ["x", "y"]
    assert client.usage.calls == 2


def test_mock_by_model():
    backend = MockLLM(by_model={"m": ["from-m"]}, default="fallback")
    client = LLMClient(backend=backend, sleep=lambda _s: None)
    assert client.chat([], _cfg(model="m")) == "from-m"
    assert client.chat([], _cfg(model="other")) == "fallback"


def test_empty_content_retries_then_raises():
    backend = MockLLM([MockResponse(content="", finish_reason="length")] * 3)
    client = LLMClient(backend=backend, sleep=lambda _s: None)
    with pytest.raises(EmptyCompletionError) as exc:
        client.chat([], _cfg())
    assert "finish_reason='length'" in str(exc.value)
    assert len(backend.requests) == 3  # 1 try + 2 retries


def test_empty_then_success():
    backend = MockLLM([MockResponse(content=""), MockResponse(content="ok")])
    client = LLMClient(backend=backend, sleep=lambda _s: None)
    assert client.chat([], _cfg()) == "ok"


def test_usage_accounting():
    u = Usage()
    u.add(10, 5)
    u.add(1, 2)
    assert u.total_tokens == 18
    assert u.to_dict()["calls"] == 2


def test_http_errors_retry(monkeypatch):
    class Flaky:
        def __init__(self):
            self.n = 0

        def complete(self, messages, cfg):
            self.n += 1
            if self.n < 2:
                raise httpx.ConnectError("boom")
            return "ok", 0, 0, "stop"

    client = LLMClient(backend=Flaky(), sleep=lambda _s: None)
    assert client.chat([], _cfg()) == "ok"
