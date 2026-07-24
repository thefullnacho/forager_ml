"""Database connection and migration helpers."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg

_HERE = Path(__file__).resolve().parent
DEFAULT_DSN = os.environ.get(
    "FORAGER_OBS_DSN",
    "postgresql://postgres:forager@localhost:5433/forager_obs",
)


def connect(dsn: str | None = None) -> psycopg.Connection:
    return psycopg.connect(dsn or DEFAULT_DSN)


def migrate(conn: psycopg.Connection) -> None:
    """Apply schema.sql then views.sql. Both are idempotent (IF NOT EXISTS /
    CREATE OR REPLACE), so this is safe to run repeatedly."""
    for fname in ("schema.sql", "views.sql"):
        conn.execute((_HERE / fname).read_text())
    conn.commit()
