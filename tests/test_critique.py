"""Offline tests for the self-critique verdict parser.

These lock in tolerance for the messy shapes small local models (llama3 via
Ollama) actually emit, and the regression that a paraphrased/combined but
correct answer must not be reported as unsupported.
"""
from __future__ import annotations

from app.graph.critique import _coerce_supported, parse_critique

_FENCE = "```"


def test_clean_valid_supported():
    v = parse_critique('{"supported": true, "issues": [], "revised_answer": null}')
    assert v["supported"] is True
    assert v["issues"] == []
    assert v["revised_answer"] is None
    assert "parse_error" not in v


def test_fenced_json_is_read_not_masked():
    # ```json ... ``` wrapper — old parser threw and silently defaulted.
    raw = _FENCE + 'json\n{"supported": true, "issues": []}\n' + _FENCE
    v = parse_critique(raw)
    assert v["supported"] is True
    assert not v.get("parse_error"), "fenced JSON should be parsed, not defaulted"


def test_prose_preamble_is_tolerated():
    raw = 'Here is my assessment:\n{"supported": true, "issues": [], "revised_answer": null}'
    v = parse_critique(raw)
    assert v["supported"] is True
    assert not v.get("parse_error")


def test_supported_null_is_not_false():
    # Regression: bool(None) == False previously produced a false "unsupported".
    v = parse_critique('{"supported": null, "issues": []}')
    assert v["supported"] is True


def test_string_boolean_true():
    assert parse_critique('{"supported": "true"}')["supported"] is True


def test_real_fabrication_is_caught():
    raw = (
        '{"supported": false, "issues": ["draft claims 30 days; context says 20"], '
        '"revised_answer": "Full-time employees get 20 days of vacation."}'
    )
    v = parse_critique(raw)
    assert v["supported"] is False
    assert v["revised_answer"].startswith("Full-time employees get 20")


def test_fenced_fabrication_still_caught():
    # The old parser would have thrown on the fence and defaulted to supported,
    # masking a real fabrication. The hardened parser must catch it.
    raw = _FENCE + 'json\n{"supported": false, "issues": ["made up"], "revised_answer": "x"}\n' + _FENCE
    assert parse_critique(raw)["supported"] is False


def test_garbage_defaults_supported_with_flag():
    v = parse_critique("the model rambled without any json at all")
    assert v["supported"] is True
    assert v["parse_error"] is True


def test_coerce_supported_variants():
    assert _coerce_supported(True) is True
    assert _coerce_supported(False) is False
    assert _coerce_supported(None) is True          # ambiguous -> supported
    assert _coerce_supported("false") is False
    assert _coerce_supported("no") is False
    assert _coerce_supported("yes") is True
    assert _coerce_supported("supported") is True
