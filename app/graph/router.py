"""Intent classification + routing.

`classify_intent` is a deterministic rule-based classifier — cheap, fast, and a
legitimate production pattern (route with rules/regex, reserve the LLM for the
hard work). It is dependency-free and unit-tested offline. Swapping in an
LLM-based classifier later is a one-function change.

Routing targets:
  * ``action`` — create/track a ticket (human-in-the-loop).
  * ``sql``    — questions about the **business data** (the ``orders`` table:
                 orders, customers, revenue, regions, order statuses, …).
  * ``qa``     — everything else, answered from the ingested **document corpus**
                 (company handbook + product FAQ: HR policies, warranties,
                 returns, support tiers, …). This is the safe default.

Design note: aggregation words like "how many"/"count"/"total"/"average" are
NOT enough to mean SQL on their own — "how many vacation days do I get?" is a
handbook question. So SQL requires either an unambiguous data-domain signal, or
an aggregation word *paired with* a data-domain noun. Document/policy/HR/FAQ
topics are pinned to ``qa`` even when phrased with aggregation words.
"""
from __future__ import annotations

import re

from app.graph.state import Intent

# --- action (ticket creation) ---
_ACTION_RE = re.compile(
    r"\b(create|open|file|raise|submit|log)\b.*\b(ticket|issue|case|incident|request)\b",
    re.I,
)
_ACTION_HINT_RE = re.compile(r"\b(ticket|incident)\b", re.I)

# --- structured data / SQL -------------------------------------------------
# Strong, unambiguous signals that the question is about the orders/business
# dataset. Any of these alone routes to SQL.
_SQL_STRONG_RE = re.compile(
    r"\b(revenue|sales|sold|orders?|customers?|transactions?|"
    r"delivered|shipped|refunded|cancelled|canceled|pending|"
    r"top \d+|(by|per)\s+(region|product|status|category|customer|month))\b",
    re.I,
)

# Aggregation phrasing. Ambiguous on its own (documents ask "how many …" too),
# so it only implies SQL when a data-domain noun is also present.
_AGG_RE = re.compile(
    r"\b(how many|number of|count|total|sum|average|avg|"
    r"most|least|highest|lowest)\b",
    re.I,
)
_DATA_NOUN_RE = re.compile(
    r"\b(orders?|revenue|sales|customers?|products?|regions?|"
    r"categories|category|units?|quantity|seats?)\b",
    re.I,
)

# Document / policy / HR / FAQ topics -> always answered from the corpus (qa),
# even when phrased with aggregation words ("how many vacation days …").
_DOC_TOPIC_RE = re.compile(
    r"\b(policy|policies|handbook|guidelines?|warrant(y|ies)|"
    r"returns?|refunds?|vacation|leave|pto|holidays?|sick|"
    r"remote work|work from home|wfh|expenses?|reimburse\w*|"
    r"onboarding|benefits?|parental|maternity|paternity|caregiver|"
    r"eligib\w*|sla|support\s+(tier|gold)|faq|privacy|shipping)\b",
    re.I,
)


def classify_intent(query: str) -> Intent:
    q = (query or "").strip()
    if not q:
        return "qa"

    if _ACTION_RE.search(q) or (_ACTION_HINT_RE.search(q) and re.search(r"\b(create|open|file|raise|need|want)\b", q, re.I)):
        return "action"

    strong_sql = bool(_SQL_STRONG_RE.search(q))

    # Policy/HR/FAQ questions belong to the document corpus. Pinning them to qa
    # before the aggregation check is what stops "how many vacation days …" from
    # being misrouted to the SQL agent (which only knows the orders table).
    if _DOC_TOPIC_RE.search(q) and not strong_sql:
        return "qa"

    if strong_sql:
        return "sql"

    # Aggregation counts as SQL only when it's about the business-data domain.
    if _AGG_RE.search(q) and _DATA_NOUN_RE.search(q):
        return "sql"

    return "qa"


def route_by_intent(state: dict) -> str:
    """Conditional-edge selector after classification."""
    return state.get("intent", "qa")
