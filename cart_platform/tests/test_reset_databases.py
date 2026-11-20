"""Resetting the databases without losing their structure."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

import settings as C                       # noqa: E402
from registry import database               # noqa: E402


def _structure(path: Path) -> dict[str, set[str]]:
    conn = sqlite3.connect(str(path))
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
                for t in tables}
    finally:
        conn.close()


def _indexes(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
            " AND name NOT LIKE 'sqlite_%'")}
    finally:
        conn.close()


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """A populated platform database at an isolated path."""
    db = tmp_path / "platform.db"
    monkeypatch.setattr(C, "DB_PATH", db)
    monkeypatch.setattr(database, "DB_PATH", db)
    monkeypatch.setattr(C, "RECEIVER_DB_PATH", tmp_path / "receiver.db")
    monkeypatch.setattr(C, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(C, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(C, "UPLOAD_DIR", tmp_path / "uploads")
    for attr in ("RUN_DIR", "SCORE_ARTIFACT_DIR", "TRAINING_ARTIFACT_DIR",
                 "PER_ENTITY_MODEL_DIR"):
        monkeypatch.setattr(C, attr, tmp_path / "logs" / attr.lower())

    database.reset_connection()
    database.init_db(db)
    conn = database.connect(db)
    conn.execute(
        "INSERT INTO model_registry (model_id, version, status, artifact_path,"
        " red_model, amber_model) VALUES ('m1',1,'champion','/x',15,20)")
    conn.execute(
        "INSERT INTO api_requests (request_id, entity_id, endpoint, method,"
        " status, created_at) VALUES ('r1','163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8','/score','POST','ok','x')")
    conn.commit()
    database.reset_connection()
    yield db
    database.reset_connection()


def _run(argv):
    from scripts.reset_databases import main
    return main(argv)


def test_dry_run_keeps_every_row(seeded):
    before = _structure(seeded)

    assert _run(["--db", "platform"]) == 0

    conn = sqlite3.connect(str(seeded))
    assert conn.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0] == 1
    conn.close()
    assert _structure(seeded) == before


def test_recreate_empties_the_data_and_keeps_the_structure(seeded):
    before, before_idx = _structure(seeded), _indexes(seeded)

    assert _run(["--db", "platform", "--yes", "--no-backup"]) == 0

    assert _structure(seeded) == before
    assert _indexes(seeded) == before_idx
    conn = sqlite3.connect(str(seeded))
    try:
        for table in before:
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    finally:
        conn.close()


def test_truncate_keeps_the_structure_and_restarts_ids(seeded):
    before = _structure(seeded)

    assert _run(["--db", "platform", "--mode", "truncate", "--yes",
                 "--no-backup"]) == 0

    assert _structure(seeded) == before
    conn = sqlite3.connect(str(seeded))
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM api_requests").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_sequence").fetchone()[0] == 0
        conn.execute(
            "INSERT INTO api_requests (request_id, entity_id, endpoint, method,"
            " status, created_at) VALUES ('r2','e','/s','POST','ok','x')")
        conn.commit()
        assert conn.execute("SELECT id FROM api_requests").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_backup_is_written_by_default(seeded):
    assert _run(["--db", "platform", "--yes"]) == 0

    backups = list(seeded.parent.glob("platform.*.bak.db"))
    assert len(backups) == 1
    conn = sqlite3.connect(str(backups[0]))
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM model_registry").fetchone()[0] == 1
    finally:
        conn.close()


def test_the_reset_database_is_immediately_usable(seeded):
    """A cached handle must not survive the file it points at."""
    database.connect(seeded)          # warm the thread-local cache

    assert _run(["--db", "platform", "--yes", "--no-backup"]) == 0

    conn = database.connect(seeded)
    conn.execute(
        "INSERT INTO model_registry (model_id, version, status, artifact_path,"
        " red_model, amber_model) VALUES ('m2',1,'candidate','/y',15,20)")
    conn.commit()
    database.reset_connection()

    # visible to a brand-new connection, i.e. written to the file on disk
    fresh = sqlite3.connect(str(seeded))
    try:
        assert fresh.execute(
            "SELECT model_id FROM model_registry").fetchone()[0] == "m2"
    finally:
        fresh.close()


def test_artifacts_are_left_alone_unless_asked(seeded):
    bundle = Path(C.MODEL_DIR) / "per_entity" / "163_215_0" / "bundle.pkl"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"x")

    _run(["--db", "platform", "--yes", "--no-backup"])
    assert bundle.exists()

    _run(["--db", "platform", "--yes", "--no-backup", "--with-artifacts"])
    assert not bundle.exists()
    assert Path(C.MODEL_DIR).is_dir()
