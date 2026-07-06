"""Parse the self-critique model's verdict into a normalized dict.

Kept **dependency-free** (no LangGraph / DB imports) so it is unit-testable
offline — the parsing is where small local models bite.

Small models served via Ollama (e.g. llama3) frequently:
  * wrap the JSON verdict in ```json ... ``` fences or add a prose preamble,
  * emit ``"supported": null`` when unsure, or a stringized ``"true"``/``"false"``.

The old parser used a bare ``json.loads`` and ``bool(data.get("supported"))``,
which mis-handled all of the above: fenced/prose output fell through to a silent
default, and ``"supported": null`` became ``bool(None) == False`` — a false
"unsupported". This module is tolerant of those shapes and, when the verdict is
genuinely unreadable, biases toward *supported* so a flaky critic never silently
suppresses an otherwise-good answer. Explicit ``false`` is still honored, so real
fabrications are still caught.
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Tokens that mean "not supported" when the model returns a string instead of a
# JSON boolean. Anything else (including the empty/unknown case) reads as True.
_FALSEY = {"false", "no", "0", "unsupported", "not supported", "untrue", "fail"}


def _coerce_supported(value: object) -> bool:
    """Interpret the model's ``supported`` field.

    Ambiguous / missing / null -> True: we only declare an answer unsupported on
    an explicit negative, so a confused critic can't block a good answer.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower().strip(".!,")
        return token not in _FALSEY
    return True


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a possibly fenced / prose-wrapped string."""
    if not isinstance(raw, str):
        raise ValueError("critique output is not text")
    text = raw.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in critique output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("critique JSON is not an object")
    return data


def parse_critique(raw: str) -> dict:
    """Normalize a raw critique completion to ``{supported, issues, revised_answer}``.

    On any parse failure, default to ``supported=True`` (with ``parse_error`` set)
    so a malformed critic response never blocks an otherwise-good answer.
    """
    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"supported": True, "issues": [], "revised_answer": None, "parse_error": True}

    return {
        "supported": _coerce_supported(data.get("supported", True)),
        "issues": list(data.get("issues", []) or []),
        "revised_answer": data.get("revised_answer"),
    }
