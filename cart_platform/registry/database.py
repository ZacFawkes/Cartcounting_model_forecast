"""SQLite access layer for the platform registry and operational tables."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from settings import DB_PATH

LOGGER = logging.getLogger("platform.db")
SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"

_LOCAL = threading.local()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def connect(path: Path | str = None) -> sqlite3.Connection:
    """One connection per thread; SQLite objects are not thread-safe."""
    path = Path(path or DB_PATH)
    key = f"conn::{path}"
    conn = getattr(_LOCAL, key, None)
    if conn is None:
        conn = sqlite3.connect(str(path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        setattr(_LOCAL, key, conn)
    return conn


def reset_connection(path: Path | str = None) -> None:
    """Drop cached handles so the next ``connect`` reopens the file.

    A handle survives ``unlink``: SQLite keeps writing to the unlinked inode
    and the work is invisible in the new file. Anything that replaces the
    database on disk has to clear the cache first.
    """
    keys = ([f"conn::{Path(path)}"] if path
            else [a for a in vars(_LOCAL) if a.startswith("conn::")])
    for key in keys:
        conn = getattr(_LOCAL, key, None)
        if conn is None:
            continue
        try:
            conn.close()
        except sqlite3.Error:     # pragma: no cover — already broken handle
            pass
        delattr(_LOCAL, key)


def init_db(path: Path | str = None) -> None:
    conn = connect(path)
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    conn.commit()
    # Additive column migrations for tables that predate the on-demand module.
    from registry.migrations import apply_migrations

    apply_migrations(path)
    LOGGER.info("Registry initialised at %s", path or DB_PATH)


@contextmanager
def tx(path: Path | str = None):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def execute(sql: str, params: tuple | dict = (), path=None) -> sqlite3.Cursor:
    with tx(path) as conn:
        return conn.execute(sql, params)


def query(sql: str, params: tuple | dict = (), path=None) -> list[dict]:
    conn = connect(path)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple | dict = (), path=None) -> dict | None:
    rows = query(sql, params, path)
    return rows[0] if rows else None


def query_df(sql: str, params: tuple | dict = (), path=None) -> pd.DataFrame:
    return pd.DataFrame(query(sql, params, path))


def upsert(table: str, row: dict, keys: list[str], path=None) -> None:
    """INSERT ... ON CONFLICT(keys) DO UPDATE for the supplied columns."""
    cols = list(row)
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in keys)
    conflict = ", ".join(keys)
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
        if updates
        else f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    )
    execute(sql, tuple(row[c] for c in cols), path)


def insert_ignore(table: str, row: dict, path=None) -> bool:
    """Returns False when the row already existed (idempotent writes)."""
    cols = list(row)
    sql = (
        f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )
    cur = execute(sql, tuple(row[c] for c in cols), path)
    return cur.rowcount > 0


def as_json(value) -> str:
    def _d(o):
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        return str(o)

    return json.dumps(value, default=_d)


def from_json(value, default=None):
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default
