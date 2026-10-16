"""Additive column migrations for tables that already exist in platform.db.

New tables are created by ``schema.sql`` (all ``CREATE TABLE IF NOT EXISTS``).
Columns added to pre-existing tables need ALTER statements, applied here so an
existing development database is upgraded in place and never overwritten.
"""
from __future__ import annotations

import logging
import sqlite3

from registry.database import connect

LOGGER = logging.getLogger("platform.db")

# table -> {column: "TYPE [DEFAULT ...]"}
ADDITIVE_COLUMNS: dict[str, dict[str, str]] = {
    "model_registry": {
        "entity_id": "TEXT",
        "upload_id": "TEXT",
        "job_id": "TEXT",
        "strategy": "TEXT",
        "dataset_checksum": "TEXT",
        "feature_schema_hash": "TEXT",
        "calibration_status": "TEXT",
        "cart_ceiling": "REAL",
        "min_history_records": "INTEGER",
        "alert_band_id": "TEXT",
        "alert_band_status": "TEXT",
        "alert_band_json": "TEXT",
    },
    "entity_config": {
        "last_upload_id": "TEXT",
    },
    "backtest_reports": {
        "config_hash": "TEXT",
        "frozen_config_json": "TEXT",
        "effective_config_json": "TEXT",
    },
    "calibration_records": {
        "calibration_red": "REAL",
        "config_hash": "TEXT",
    },
}


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def apply_migrations(path=None) -> list[str]:
    """Idempotently add missing columns. Returns the statements applied."""
    conn = connect(path)
    applied: list[str] = []
    for table, columns in ADDITIVE_COLUMNS.items():
        present = _existing_columns(conn, table)
        if not present:
            continue  # table not created yet; schema.sql will define it fully
        for column, decl in columns.items():
            if column in present:
                continue
            stmt = f"ALTER TABLE {table} ADD COLUMN {column} {decl}"
            conn.execute(stmt)
            applied.append(stmt)
    if applied:
        conn.commit()
        LOGGER.info("Applied %d additive migration(s)", len(applied))
    return applied
