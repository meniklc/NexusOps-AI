"""Evaluation dataset: ~10 Q&A pairs grounded in sample_docs/.

`expected_keywords` are the salient facts a correct answer / retrieval should
contain; they drive context_recall and answer_relevance.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalItem:
    question: str
    expected_keywords: list[str]
    reference_answer: str = ""
    intent: str = "qa"
    tags: list[str] = field(default_factory=list)


DATASET: list[EvalItem] = [
    EvalItem(
        question="How many paid vacation days do full-time employees get per year?",
        expected_keywords=["20", "vacation"],
        reference_answer="Full-time employees accrue 20 days of paid vacation per year.",
    ),
    EvalItem(
        question="How many days can be carried over to the next year?",
        expected_keywords=["5", "carried"],
        reference_answer="Up to 5 unused vacation days may be carried over.",
    ),
    EvalItem(
        question="What is the office attendance expectation under the hybrid policy?",
        expected_keywords=["two days", "office"],
        reference_answer="Employees are expected in the office at least two days per week.",
    ),
    EvalItem(
        question="Within how many days must business expenses be submitted?",
        expected_keywords=["30 days", "expense"],
        reference_answer="Business expenses must be submitted within 30 days.",
    ),
    EvalItem(
        question="How much paid parental leave does the primary caregiver receive?",
        expected_keywords=["16 weeks", "parental"],
        reference_answer="The primary caregiver receives 16 weeks of paid parental leave.",
    ),
    EvalItem(
        question="What is the return window for hardware products?",
        expected_keywords=["30 days", "return"],
        reference_answer="Hardware may be returned within 30 days of delivery for a full refund.",
    ),
    EvalItem(
        question="How much does an Analytics Seat cost per year?",
        expected_keywords=["149", "seat"],
        reference_answer="An Analytics Seat costs $149.00 per seat per year.",
    ),
    EvalItem(
        question="What response SLA does Support Gold provide for critical issues?",
        expected_keywords=["4-hour", "sla"],
        reference_answer="Support Gold provides a 4-hour response SLA for critical issues.",
    ),
    EvalItem(
        question="How long are APAC and LATAM orders expected to take to ship?",
        expected_keywords=["5 business days", "apac"],
        reference_answer="APAC and LATAM orders ship within 5 business days.",
    ),
    EvalItem(
        question="How long does Acme retain order data for compliance?",
        expected_keywords=["7 years", "retain"],
        reference_answer="Acme retains order data for 7 years for compliance.",
    ),
]
