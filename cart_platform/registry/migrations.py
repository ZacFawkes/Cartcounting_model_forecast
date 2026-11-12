"""Additive column migrations for tables that already exist in platform.db.

New tables are created by ``schema.sql`` (all ``CREATE TABLE IF NOT EXISTS``).
Columns added to pre-existing tables need ALTER statements, applied here so an
existing development database is upgraded in place and never overwritten.
"""
from __future__ import annotations

import logging
import re
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
        # Disk retention (registry/retention.py): when the artifacts of this
        # retired model were deleted, and how many bytes that freed.
        "purged_at": "TEXT",
        "purged_bytes": "INTEGER",
    },
    "entity_config": {
        "last_upload_id": "TEXT",
        # D9c — identity includes the physical bay a channel watches.
        "bay_id": "TEXT",
    },
    "upload_entities": {
        "bay_id": "TEXT",
    },
    "scoring_results": {
        # Stamped by the outbox when the receiver acknowledges the decision.
        # Retention refuses to delete a decision without it.
        "delivered_at": "TEXT",
        "bay_id": "TEXT",
    },
    "replenishment_orders": {
        "bay_id": "TEXT",
    },
    "scoring_results": {
        # Stamped by the outbox when the receiver acknowledges the decision.
        # Retention refuses to delete a decision without it.
        "delivered_at": "TEXT",
        "bay_id": "TEXT",
    },
    "replenishment_orders": {
        "bay_id": "TEXT",
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


LEGACY_ENTITY_UNIQUE = "UNIQUE (kdvr, chan_no, site_key)"

# One registry row per physical bay. A channel that watches several bays has
# one entity per bay, so (site, kdvr, chan) alone is not unique any more.
ENTITY_IDENTITY_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_entity_identity "
    "ON entity_config(site_key, kdvr, chan_no, IFNULL(bay_id, ''))"
)


def _drop_legacy_entity_unique(conn: sqlite3.Connection) -> list[str]:
    """Rebuild entity_config without the pre-bay UNIQUE table constraint.

    SQLite cannot drop a table constraint in place, so the table is recreated
    from its own DDL with that one line removed and the rows copied across.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='entity_config'"
    ).fetchone()
    if row is None or LEGACY_ENTITY_UNIQUE not in (row[0] or ""):
        return []

    ddl = row[0]
    rebuilt = ddl.replace(f"    {LEGACY_ENTITY_UNIQUE},\n", "")
    if rebuilt == ddl:                      # different whitespace in an older DB
        rebuilt = re.sub(rf"\s*{re.escape(LEGACY_ENTITY_UNIQUE)}\s*,", "", ddl)
    rebuilt = rebuilt.replace("entity_config", "entity_config_migrate", 1)
    columns = ", ".join(sorted(_existing_columns(conn, "entity_config")))
    indexes = [
        r[0] for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name="
            "'entity_config' AND sql IS NOT NULL"
        )
    ]

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(rebuilt)
        conn.execute(
            f"INSERT INTO entity_config_migrate ({columns}) "
            f"SELECT {columns} FROM entity_config"
        )
        conn.execute("DROP TABLE entity_config")
        conn.execute("ALTER TABLE entity_config_migrate RENAME TO entity_config")
        for stmt in indexes:
            conn.execute(stmt.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1)
                         if "IF NOT EXISTS" not in stmt else stmt)
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    LOGGER.info("Rebuilt entity_config without the legacy identity UNIQUE constraint")
    return [f"REBUILD entity_config (dropped {LEGACY_ENTITY_UNIQUE})"]


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

    applied += _drop_legacy_entity_unique(conn)
    if _existing_columns(conn, "entity_config"):
        conn.execute(ENTITY_IDENTITY_INDEX)
        conn.commit()
    if applied:
        LOGGER.info("Applied %d additive migration(s)", len(applied))
    return applied
