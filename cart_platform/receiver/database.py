"""Receiver SQLite connection, migrations and idempotent store.

Separate database file from the ML server's ``platform.db``. The two are never
merged and the ML server never opens this file directly.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("receiver.db")
SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"

_LOCAL = threading.local()
_DB_PATH: Path | None = None


def configure(path: Path | str) -> None:
    global _DB_PATH
    _DB_PATH = Path(path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def db_path() -> Path:
    if _DB_PATH is None:
        from settings import RECEIVER_DB_PATH

        configure(RECEIVER_DB_PATH)
    return _DB_PATH


def connect() -> sqlite3.Connection:
    path = db_path()
    key = f"rconn::{path}"
    conn = getattr(_LOCAL, key, None)
    if conn is None:
        conn = sqlite3.connect(str(path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        setattr(_LOCAL, key, conn)
    return conn


_ADDITIVE_COLUMNS = {
    "training_results": (("config_hash", "TEXT"), ("frozen_config_json", "TEXT"),
                         ("effective_config_json", "TEXT")),
    "calibration_results": (("config_hash", "TEXT"),),
    "backtest_results": (("config_hash", "TEXT"), ("frozen_config_json", "TEXT"),
                         ("effective_config_json", "TEXT")),
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add config-provenance columns to receiver databases created earlier."""
    for table, columns in _ADDITIVE_COLUMNS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                LOGGER.info("Receiver migration: added %s.%s", table, name)


def init_db() -> None:
    conn = connect()
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    _migrate(conn)
    conn.commit()
    LOGGER.info("Receiver database ready at %s", db_path())


def reset_connection() -> None:
    """Drop the thread-local handle (used by tests that re-point the DB)."""
    for attr in [a for a in vars(_LOCAL) if a.startswith("rconn::")]:
        try:
            getattr(_LOCAL, attr).close()
        except Exception:
            pass
        delattr(_LOCAL, attr)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    conn = connect()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur
    except Exception:
        conn.rollback()
        raise


def query(sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in connect().execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def as_json(value) -> str:
    return json.dumps(value, default=str)


def from_json(value, default=None):
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default
