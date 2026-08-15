"""This repo's observability database: its name, its port, its SQL.

The connection and migration mechanics live in ``forager_obs`` and are shared
with the Field Station. What stays here is only what must differ between the
two: the database name, the host port, and the location of the schema/views
that define THIS repo's grain (one row per image, plus a child row per expert).
"""

from __future__ import annotations

from pathlib import Path

import psycopg
from forager_obs import connect as _connect
from forager_obs import default_dsn
from forager_obs import migrate as _migrate

_HERE = Path(__file__).resolve().parent

DB_NAME = "forager_obs"
#: Must differ from the Field Station's (5434). They previously shared 5433, so
#: the second compose file to come up either failed on the port or pointed at
#: the first repo's database.
PORT = 5433

DEFAULT_DSN = default_dsn(DB_NAME, PORT)


def connect(dsn: str | None = None) -> psycopg.Connection:
    """FORAGER_OBS_DSN still wins when set; see forager_obs.db.connect."""
    return _connect(dsn, fallback=DEFAULT_DSN)


def migrate(conn: psycopg.Connection) -> None:
    """Apply this repo's schema.sql then views.sql. Both are idempotent."""
    _migrate(conn, _HERE)
