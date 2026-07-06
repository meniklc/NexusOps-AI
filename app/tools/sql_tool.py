"""NL->SQL tool over the sample `orders` table.

Pipeline: question -> LLM generates SQL (schema-aware prompt) -> static safety
guard -> execute inside a READ ONLY transaction with a statement timeout ->
deterministic summary. With `LLM_PROVIDER=fake` the SQL comes from a heuristic
translator, so the whole path runs offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.db import get_conn
from app.llm.base import ChatMessage
from app.llm.factory import get_llm
from app.tools.sql_guard import UnsafeSQLError, ensure_limit, validate_select

# The `LLM-TASK: sql` tag lets the Fake provider dispatch to its heuristic.
SQL_SYSTEM = """LLM-TASK: sql
You are a careful analytics engineer. Translate the user's question into a
single, read-only PostgreSQL SELECT query against this schema:

  orders(
    id            BIGINT,
    customer_name TEXT,
    product       TEXT,        -- 'Widget Pro' | 'Widget Mini' | 'Analytics Seat' | 'Support Gold'
    category      TEXT,        -- 'hardware' | 'software' | 'service'
    quantity      INTEGER,
    unit_price    NUMERIC,
    status        TEXT,        -- 'pending' | 'shipped' | 'delivered' | 'cancelled' | 'refunded'
    region        TEXT,        -- 'NA' | 'EMEA' | 'APAC' | 'LATAM'
    order_date    DATE
  )

Rules:
- Output ONLY the SQL, no prose, no code fences.
- Use a single SELECT statement. Never modify data.
- Revenue = SUM(quantity * unit_price).
"""

MAX_ROWS = 100


@dataclass
class SQLResult:
    question: str
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    row_count: int = 0
    summary: str = ""
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "summary": self.summary,
            "error": self.error,
        }


def generate_sql(question: str) -> str:
    llm = get_llm()
    raw = llm.chat(
        [ChatMessage("system", SQL_SYSTEM), ChatMessage("user", f"Question: {question}")]
    )
    # Strip accidental code fences from chatty models.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned[3:]
    return cleaned.strip()


def execute_select(sql: str) -> tuple[list[str], list[list]]:
    """Run a validated SELECT inside a read-only, time-limited transaction."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Defence in depth: DB-enforced read-only + timeout.
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL statement_timeout = '5s'")
            cur.execute(sql)
            columns = [d.name for d in cur.description] if cur.description else []
            rows = [list(r) for r in cur.fetchmany(MAX_ROWS)]
        conn.rollback()  # nothing to commit; releases the transaction
    return columns, rows


def summarize(question: str, columns: list[str], rows: list[list]) -> str:
    if not rows:
        return "The query returned no matching rows."
    if len(rows) == 1 and len(columns) == 1:
        return f"Result: {columns[0]} = {rows[0][0]}."
    if len(rows) == 1:
        pairs = ", ".join(f"{c} = {v}" for c, v in zip(columns, rows[0]))
        return f"Result: {pairs}."
    # Multi-row: describe grouped output compactly.
    preview = []
    for r in rows[:5]:
        preview.append(", ".join(f"{c}={v}" for c, v in zip(columns, r)))
    more = "" if len(rows) <= 5 else f" (+{len(rows) - 5} more)"
    return f"Returned {len(rows)} rows. Top: " + " | ".join(preview) + more


def run_sql_agent(question: str) -> SQLResult:
    """Full NL->SQL->execute->summarize flow with guardrails."""
    raw_sql = ""
    try:
        raw_sql = generate_sql(question)
        safe_sql = ensure_limit(validate_select(raw_sql), MAX_ROWS)
    except UnsafeSQLError as exc:
        return SQLResult(
            question=question,
            sql=raw_sql,
            error=f"Rejected unsafe SQL: {exc}",
            summary="I could not run that query because it was not a safe read-only SELECT.",
        )

    try:
        columns, rows = execute_select(safe_sql)
    except Exception as exc:  # DB / syntax / timeout
        return SQLResult(
            question=question,
            sql=safe_sql,
            error=str(exc),
            summary=f"The query failed to execute: {exc}",
        )

    return SQLResult(
        question=question,
        sql=safe_sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        summary=summarize(question, columns, rows),
    )
