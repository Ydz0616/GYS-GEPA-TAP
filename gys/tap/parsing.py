"""Parsers for attacker, judge, and on-topic model outputs.

Behavior mirrors RICommunity/TAP: malformed attacker JSON yields ``None``; a
missing judge rating defaults to ``1``; a missing on-topic verdict defaults to
on-topic (``1``). Keeping these fallbacks identical preserves the loop's
pruning semantics.
"""

from __future__ import annotations

import ast
import re

_JUDGE_RE = re.compile(r"\[\[(\d+)\]\]")
_ON_TOPIC_RE = re.compile(r"\[\[(yes|no)\]\]")


def extract_attacker_json(raw: str) -> dict | None:
    """Extract ``{"improvement": ..., "prompt": ...}`` from attacker output.

    Returns ``None`` when no valid object with both keys can be parsed.
    """
    start = raw.find("{")
    end = raw.find("}") + 1
    if start == -1 or end == 0:
        return None
    candidate = raw[start:end].replace("\n", "")
    try:
        parsed = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, dict):
        return None
    if not all(key in parsed for key in ("improvement", "prompt")):
        return None
    return parsed


def parse_judge_score(raw: str) -> int:
    """Parse a 1-10 rating from ``Rating: [[n]]``; default 1 if absent."""
    match = _JUDGE_RE.search(raw)
    if match is None:
        return 1
    return int(match.group(1))


def parse_on_topic(raw: str) -> int:
    """Parse ``Response: [[YES|NO]]`` to 1/0; default 1 (on-topic) if absent."""
    match = _ON_TOPIC_RE.search(raw.lower())
    if match is None:
        return 1
    return int(match.group(1) == "yes")
