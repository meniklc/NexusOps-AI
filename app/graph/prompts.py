"""System prompts for the graph's LLM nodes.

Each begins with the `LLM-TASK:` tag so the offline Fake provider can dispatch
deterministically (real models treat it as an ordinary instruction line).
"""
from __future__ import annotations

from typing import Sequence

SYNTHESIS_SYSTEM = """LLM-TASK: synthesize
You are a precise enterprise assistant. Answer the user's question using ONLY
the numbered context snippets provided. Cite every claim with its snippet
number in square brackets, e.g. [1], [2]. If the context does not contain the
answer, say so plainly rather than guessing. Be concise.
"""

CRITIQUE_SYSTEM = """LLM-TASK: critique
You check whether a DRAFT answer is grounded in the provided CONTEXT. Judge
MEANING, not wording — an answer is SUPPORTED whenever the context states or
clearly implies it, even if the words differ.

These are all SUPPORTED (do NOT flag them):
- Paraphrasing or rewording a fact from the context.
- Combining two or more facts from the context into a single sentence.
- Summarising or restating a number/limit that matches the context.
- Naming the company or document subject the context is obviously about.

Mark the answer UNSUPPORTED only for a real fabrication: a claim the context
does NOT state and does NOT imply, or that CONTRADICTS the context (e.g. wrong
numbers). Do not flag an answer just because it is phrased differently.

Respond with ONLY a JSON object and nothing else — no prose, no code fences:
{"supported": true or false, "issues": ["..."], "revised_answer": "..." or null}
Use "supported": false with a corrected "revised_answer" ONLY when there is a
real fabrication. Otherwise use "supported": true, "issues": [], "revised_answer": null.
"""


def format_context(retrieved: Sequence[dict]) -> str:
    """Render retrieved chunks as numbered, citeable snippets."""
    lines = []
    for i, chunk in enumerate(retrieved, start=1):
        source = chunk.get("source", "unknown")
        content = " ".join(str(chunk.get("content", "")).split())
        lines.append(f"[{i}] (source: {source}) {content}")
    return "\n".join(lines)


def synthesis_user_prompt(question: str, retrieved: Sequence[dict]) -> str:
    context = format_context(retrieved) if retrieved else "(no context retrieved)"
    return f"Question: {question}\n\nContext:\n{context}"


def critique_user_prompt(draft: str, retrieved: Sequence[dict]) -> str:
    context = format_context(retrieved) if retrieved else "(no context retrieved)"
    return f"DRAFT:\n{draft}\n\nCONTEXT:\n{context}"
