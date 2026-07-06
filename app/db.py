"""Database access helpers.

Thin wrapper around a psycopg connection pool. We use raw psycopg (v3) rather
than an ORM because the vector / full-text queries are hand-written SQL, and
pgvector integrates with psycopg via `register_vector`.
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from app.config import settings

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    """Per-connection setup: register the pgvector adapter."""
    register_vector(conn)


def get_pool() -> ConnectionPool:
    """Lazily create a process-wide connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.psycopg_dsn,
            min_size=1,
            max_size=10,
            configure=_configure,
            open=True,
        )
    return _pool


@contextlib.contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Borrow a connection from the pool as a context manager."""
    with get_pool().connection() as conn:
        yield conn


def healthcheck() -> bool:
    """Return True if the database is reachable."""
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
