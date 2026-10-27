"""A single OpenAI-compatible chat-completions client plus a mock backend.

Every role in the harness (attacker, target, evaluator, judge, reflector) talks
to a model through :class:`LLMClient`. Real calls go through :class:`HTTPBackend`
(any OpenAI-compatible ``POST {base_url}/chat/completions`` endpoint, e.g. Gemini's
compatibility layer); tests use :class:`MockLLM`, which never touches the network.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from gys.config import ModelConfig

Message = dict[str, str]


class EmptyCompletionError(RuntimeError):
    """Raised when the model returns no usable content.

    Gemini 3.x "flash" models can spend the whole ``max_tokens`` budget on hidden
    reasoning and return an empty ``content``. We treat that as retryable and, if
    it never resolves, raise this with the finish reason for a clear diagnostic.
    """


@dataclass
class Usage:
    """Cumulative token accounting across every call made through a client."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


class ChatBackend(Protocol):
    """A minimal chat-completions backend."""

    def complete(
        self, messages: list[Message], cfg: ModelConfig
    ) -> tuple[str, int, int, str]:
        """Return ``(content, prompt_tokens, completion_tokens, finish_reason)``."""
        ...


class HTTPBackend:
    """Calls an OpenAI-compatible ``/chat/completions`` endpoint over HTTP."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client()
        self._owns_client = client is None

    def complete(
        self, messages: list[Message], cfg: ModelConfig
    ) -> tuple[str, int, int, str]:
        api_key = os.environ.get(cfg.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"environment variable {cfg.api_key_env} is not set; "
                "export the API key or run with the mock backend"
            )
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        resp = self._client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=cfg.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content") or ""
        finish = choice.get("finish_reason") or ""
        usage = data.get("usage") or {}
        return (
            content,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            finish,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


@dataclass
class MockResponse:
    """A canned backend reply for tests."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


class MockLLM:
    """Scripted backend. Responses can be queued globally or keyed by role model.

    Every request is recorded in :attr:`requests` so tests can assert on the exact
    messages each role sent.
    """

    def __init__(
        self,
        responses: list[MockResponse | str] | None = None,
        *,
        by_model: dict[str, list[MockResponse | str]] | None = None,
        default: MockResponse | str | None = None,
    ) -> None:
        self._queue = [self._coerce(r) for r in (responses or [])]
        self._by_model = {
            m: [self._coerce(r) for r in rs] for m, rs in (by_model or {}).items()
        }
        self._default = self._coerce(default) if default is not None else None
        self.requests: list[dict[str, object]] = []

    @staticmethod
    def _coerce(r: MockResponse | str) -> MockResponse:
        return r if isinstance(r, MockResponse) else MockResponse(content=r)

    def complete(
        self, messages: list[Message], cfg: ModelConfig
    ) -> tuple[str, int, int, str]:
        self.requests.append({"model": cfg.model, "messages": messages})
        queue = self._by_model.get(cfg.model)
        if queue:
            r = queue.pop(0)
        elif self._queue:
            r = self._queue.pop(0)
        elif self._default is not None:
            r = self._default
        else:
            raise AssertionError(
                f"MockLLM exhausted: no scripted response for model {cfg.model!r}"
            )
        return r.content, r.prompt_tokens, r.completion_tokens, r.finish_reason


@dataclass
class LLMClient:
    """Wraps a backend with retries, empty-content handling, and usage accounting."""

    backend: ChatBackend
    usage: Usage = field(default_factory=Usage)
    sleep: object = time.sleep  # injectable so tests don't actually wait

    def chat(self, messages: list[Message], cfg: ModelConfig) -> str:
        last_error: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                content, p_tok, c_tok, finish = self.backend.complete(messages, cfg)
                self.usage.add(p_tok, c_tok)
                if content.strip():
                    return content
                last_error = EmptyCompletionError(
                    f"model {cfg.model!r} returned empty content "
                    f"(finish_reason={finish!r})"
                )
            except (httpx.HTTPError, EmptyCompletionError) as exc:
                last_error = exc
            if attempt < cfg.max_retries:
                self.sleep(min(2.0**attempt, 8.0))
        assert last_error is not None
        raise last_error


def build_client(mock: bool, backend: ChatBackend | None = None) -> LLMClient:
    """Convenience constructor used by the CLI and tests."""
    if backend is not None:
        return LLMClient(backend=backend)
    if mock:
        return LLMClient(backend=MockLLM(default="Rating: [[1]]"))
    return LLMClient(backend=HTTPBackend())
