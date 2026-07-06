"""Read-only SQL guardrail.

Validates that an LLM-generated query is a single, read-only SELECT before it
ever touches the database. Dependency-free so it can be unit-tested offline.

Defence in depth: the executor *also* runs inside a READ ONLY transaction with a
statement timeout (see sql_tool.py). Never rely on string checks alone in
production — this pairs a static allow-list with a hard DB-level guarantee.
"""
from __future__ import annotations

import re

FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "copy", "merge", "call", "do", "vacuum", "analyze",
    "comment", "reindex", "cluster", "lock", "set", "reset", "begin",
    "commit", "rollback", "savepoint", "listen", "notify",
}

_ALLOWED_START = ("select", "with")
_COMMENT_RE = re.compile(r"(--|/\*|\*/)")


class UnsafeSQLError(ValueError):
    """Raised when a query is not a safe, single read-only SELECT."""


def _strip(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def is_safe_select(sql: str) -> bool:
    try:
        validate_select(sql)
        return True
    except UnsafeSQLError:
        return False


def validate_select(sql: str) -> str:
    """Return a cleaned single-statement SELECT, or raise UnsafeSQLError."""
    cleaned = _strip(sql)
    if not cleaned:
        raise UnsafeSQLError("empty query")

    if _COMMENT_RE.search(cleaned):
        raise UnsafeSQLError("SQL comments are not allowed")

    # Reject multiple statements (any remaining semicolon after stripping one trailing).
    if ";" in cleaned:
        raise UnsafeSQLError("multiple statements are not allowed")

    lowered = cleaned.lower()
    if not lowered.startswith(_ALLOWED_START):
        raise UnsafeSQLError("only SELECT / WITH queries are allowed")

    # Word-boundary scan for mutating keywords.
    words = set(re.findall(r"[a-z_]+", lowered))
    bad = words & FORBIDDEN
    if bad:
        raise UnsafeSQLError(f"forbidden keyword(s): {', '.join(sorted(bad))}")

    return cleaned


def ensure_limit(sql: str, max_rows: int = 100) -> str:
    """Append a LIMIT if the (validated) query has none."""
    cleaned = _strip(sql)
    if re.search(r"\blimit\b", cleaned, re.IGNORECASE):
        return cleaned
    return f"{cleaned} LIMIT {max_rows}"
