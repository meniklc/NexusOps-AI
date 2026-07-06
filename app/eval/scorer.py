"""Lightweight, dependency-free RAG scorers.

We implement compact analogues of the standard RAGAS metrics so the harness
runs offline (and for free with the Fake provider):

- context_recall     : did retrieval surface the facts needed to answer?
- faithfulness       : is the answer grounded in the retrieved context
                       (i.e. not hallucinated)?
- answer_relevance   : does the answer actually address the question / contain
                       the expected key facts?
- citation_rate      : does the answer cite its sources ([n])?

Each is in [0, 1]. To use the real RAGAS library instead, see the note in
run_eval.py — the item/score shapes line up with RAGAS inputs.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "it", "this", "that", "these",
    "those", "from", "per", "up", "may", "must", "can", "will", "your", "you",
    "what", "how", "many", "much", "do", "does", "which", "who", "when", "where",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CITATION_RE = re.compile(r"\[\d+\]")


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1}


@dataclass
class Scores:
    context_recall: float
    faithfulness: float
    answer_relevance: float
    citation_rate: float

    def mean(self) -> float:
        vals = [self.context_recall, self.faithfulness, self.answer_relevance, self.citation_rate]
        return sum(vals) / len(vals)

    def as_dict(self) -> dict:
        return asdict(self)


def _ratio(found: int, total: int) -> float:
    return 1.0 if total == 0 else found / total


def context_recall(retrieved_texts: list[str], expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present anywhere in the retrieved context."""
    if not expected_keywords:
        return 1.0
    haystack = " ".join(retrieved_texts).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in haystack)
    return _ratio(hits, len(expected_keywords))


def faithfulness(answer: str, retrieved_texts: list[str]) -> float:
    """Share of the answer's content tokens that are supported by the context.

    A proxy for "not hallucinated": tokens in the answer that never appear in
    any retrieved chunk are treated as potentially unsupported.
    """
    answer_toks = tokens(answer)
    if not answer_toks:
        return 0.0
    context_toks = set()
    for t in retrieved_texts:
        context_toks |= tokens(t)
    supported = sum(1 for t in answer_toks if t in context_toks)
    return supported / len(answer_toks)


def answer_relevance(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected key facts that appear in the answer."""
    if not expected_keywords:
        return 1.0
    ans = (answer or "").lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans)
    return _ratio(hits, len(expected_keywords))


def citation_rate(answer: str) -> float:
    return 1.0 if _CITATION_RE.search(answer or "") else 0.0


def score_item(
    answer: str, retrieved_texts: list[str], expected_keywords: list[str]
) -> Scores:
    return Scores(
        context_recall=context_recall(retrieved_texts, expected_keywords),
        faithfulness=faithfulness(answer, retrieved_texts),
        answer_relevance=answer_relevance(answer, expected_keywords),
        citation_rate=citation_rate(answer),
    )
