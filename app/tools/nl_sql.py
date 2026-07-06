"""Deterministic natural-language -> SQL heuristic for the sample `orders` table.

This powers two things:

1. The **offline** path: the Fake LLM provider delegates SQL generation here so
   the NL->SQL agent works with no API key.
2. A **safety net** for the OpenAI path (not used by default) if you ever want a
   non-LLM fallback.

It is intentionally scoped to the demo schema and only ever emits read-only
`SELECT` statements. It is *not* a general text-to-SQL engine — a real system
would use a schema-aware LLM plus strict allow-listing (which is what the
OpenAI path does), but this keeps the demo runnable and deterministic.

Schema (see db/init.sql):
    orders(id, customer_name, product, category, quantity, unit_price,
           status, region, order_date)
"""
from __future__ import annotations

import re

STATUSES = {"pending", "shipped", "delivered", "cancelled", "canceled", "refunded"}
REGIONS = {"na": "NA", "emea": "EMEA", "apac": "APAC", "latam": "LATAM"}
CATEGORIES = {"hardware", "software", "service"}
PRODUCTS = {
    "widget pro": "Widget Pro",
    "widget mini": "Widget Mini",
    "analytics seat": "Analytics Seat",
    "support gold": "Support Gold",
}


def _where_clauses(q: str) -> list[str]:
    clauses: list[str] = []

    for status in STATUSES:
        if re.search(rf"\b{status}\b", q):
            canonical = "cancelled" if status in {"cancelled", "canceled"} else status
            clauses.append(f"status = '{canonical}'")
            break

    for key, canonical in REGIONS.items():
        # match 'emea', 'apac', or phrases like 'north america'/'na region'
        if re.search(rf"\b{key}\b", q):
            clauses.append(f"region = '{canonical}'")
            break

    for cat in CATEGORIES:
        if re.search(rf"\b{cat}\b", q):
            clauses.append(f"category = '{cat}'")
            break

    for key, canonical in PRODUCTS.items():
        if key in q:
            clauses.append(f"product = '{canonical}'")
            break

    return clauses


def heuristic_nl_to_sql(question: str) -> str:
    """Translate a natural-language question into a safe read-only SQL query."""
    q = question.lower().strip()
    where = _where_clauses(q)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    wants_revenue = any(
        w in q for w in ("revenue", "sales", "total value", "how much", "amount", "sum")
    )
    wants_count = any(w in q for w in ("how many", "number of", "count"))
    group_by_region = "region" in q and any(
        w in q for w in ("by region", "per region", "each region", "across region")
    )
    group_by_product = "product" in q and any(
        w in q for w in ("by product", "per product", "each product")
    )
    group_by_status = "status" in q and any(
        w in q for w in ("by status", "per status", "each status")
    )

    revenue_expr = "SUM(quantity * unit_price) AS revenue"

    if group_by_region:
        metric = revenue_expr if wants_revenue else "COUNT(*) AS orders"
        return f"SELECT region, {metric} FROM orders{where_sql} GROUP BY region ORDER BY 2 DESC;"
    if group_by_product:
        metric = revenue_expr if wants_revenue else "COUNT(*) AS orders"
        return f"SELECT product, {metric} FROM orders{where_sql} GROUP BY product ORDER BY 2 DESC;"
    if group_by_status:
        metric = revenue_expr if wants_revenue else "COUNT(*) AS orders"
        return f"SELECT status, {metric} FROM orders{where_sql} GROUP BY status ORDER BY 2 DESC;"

    if wants_revenue:
        return f"SELECT {revenue_expr} FROM orders{where_sql};"
    if wants_count:
        return f"SELECT COUNT(*) AS count FROM orders{where_sql};"

    # Default: recent matching rows.
    return f"SELECT * FROM orders{where_sql} ORDER BY order_date DESC LIMIT 10;"
