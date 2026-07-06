"""Heuristic prompt-injection / jailbreak detector.

A lightweight, dependency-free first line of defence run on every incoming user
message before it reaches the graph. It flags common injection patterns
(instruction overrides, system-prompt exfiltration, role hijacking, obfuscation)
and returns a score + matched signals.

IMPORTANT — this is deliberately simple. A production system should back this
with a dedicated guardrail model (e.g. Llama Guard / Prompt Guard, Azure/AWS
content safety, or an LLM-as-judge classifier) and treat this regex layer as
cheap pre-filtering only. The `detect_injection` return shape is designed to be
a drop-in replacement point for such a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (compiled_pattern, weight, label)
_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?)", re.I), 0.9, "instruction-override"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|above|the)\s+\w+", re.I), 0.8, "instruction-override"),
    (re.compile(r"forget\s+(everything|all|your)\s+\w+", re.I), 0.6, "instruction-override"),
    (re.compile(r"(reveal|show|print|repeat|display)\s+(me\s+)?(your\s+)?(the\s+)?(system\s+)?(prompt|instructions|rules)", re.I), 0.9, "prompt-exfiltration"),
    (re.compile(r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions)", re.I), 0.7, "prompt-exfiltration"),
    (re.compile(r"you\s+are\s+now\s+", re.I), 0.6, "role-hijack"),
    (re.compile(r"(act|behave|pretend|roleplay)\s+as\s+(if\s+)?", re.I), 0.5, "role-hijack"),
    (re.compile(r"(developer|god|admin|dan)\s+mode", re.I), 0.7, "jailbreak"),
    (re.compile(r"\bjailbreak\b", re.I), 0.7, "jailbreak"),
    (re.compile(r"without\s+any\s+(restrictions|filters|rules|guardrails)", re.I), 0.7, "jailbreak"),
    (re.compile(r"(exfiltrate|leak|send)\s+.*(data|secrets?|keys?|credentials?)", re.I), 0.8, "exfiltration"),
    (re.compile(r"</?(system|assistant|user)>", re.I), 0.6, "delimiter-injection"),
    (re.compile(r"\bBEGIN\s+SYSTEM\b|\bEND\s+SYSTEM\b", re.I), 0.6, "delimiter-injection"),
]

# Long base64-looking blobs are a common obfuscation vector.
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

DEFAULT_THRESHOLD = 0.7


@dataclass
class InjectionVerdict:
    is_suspicious: bool
    score: float
    matched: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "is_suspicious": self.is_suspicious,
            "score": round(self.score, 3),
            "matched": self.matched,
            "reason": self.reason,
        }


def detect_injection(text: str, threshold: float = DEFAULT_THRESHOLD) -> InjectionVerdict:
    if not text or not text.strip():
        return InjectionVerdict(is_suspicious=False, score=0.0, reason="empty input")

    matched: list[str] = []
    score = 0.0
    for pattern, weight, label in _PATTERNS:
        if pattern.search(text):
            matched.append(label)
            score = max(score, weight)  # take strongest signal, then bump for stacking

    # Stacking distinct signals raises confidence.
    distinct = len(set(matched))
    if distinct > 1:
        score = min(1.0, score + 0.1 * (distinct - 1))

    if _BASE64_RE.search(text):
        matched.append("obfuscation-base64")
        score = min(1.0, score + 0.2)

    is_suspicious = score >= threshold
    reason = (
        f"matched {sorted(set(matched))}" if matched else "no injection patterns matched"
    )
    return InjectionVerdict(is_suspicious=is_suspicious, score=score, matched=sorted(set(matched)), reason=reason)
