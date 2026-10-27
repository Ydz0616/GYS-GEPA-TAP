"""Reflector wiring for GEPA.

GEPA proposes a new attacker system prompt by feeding the current prompt and the
judge feedback into a reflection LM, using a reflection prompt template. The
template is versioned under ``prompts/reflector/`` and uses the ``<curr_param>``
and ``<side_info>`` placeholders that GEPA's instruction proposer fills. This
module loads that template and adapts our OpenAI-compatible client into the
``callable(prompt) -> str`` interface GEPA expects for ``reflection_lm``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gys.config import ModelConfig
from gys.llm import LLMClient

REFLECTOR_PROMPT_PATH = Path("prompts") / "reflector" / "reflector_v000.md"


def reflection_prompt_template(path: Path = REFLECTOR_PROMPT_PATH) -> str:
    """Load the versioned reflector template passed to ``gepa.optimize``."""
    return path.read_text(encoding="utf-8")


def make_reflection_lm(client: LLMClient, model: ModelConfig):
    """Adapt our chat client to GEPA's reflection-LM calling convention."""

    def _call(prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        return client.chat(messages, model)

    return _call
