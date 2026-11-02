"""Disk retention for retired models — registry/retention.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

import settings as C                       # noqa: E402
from registry import database               # noqa: E402


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Isolated platform tree + database seeded with one of each status."""
    monkeypatch.setattr(C, "DB_PATH", tmp_path / "platform.db")
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "platform.db")
    monkeypatch.setattr(C, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(C, "PER_ENTITY_MODEL_DIR", tmp_path / "models" / "per_entity")
    monkeypatch.setattr(C, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(C, "TRAINING_ARTIFACT_DIR", tmp_path / "logs" / "training")
    monkeypatch.setattr(C, "SCORE_ARTIFACT_DIR", tmp_path / "logs" / "scores")
    monkeypatch.setattr(C, "UPLOAD_DIR", tmp_path / "uploads")
    for d in (C.PER_ENTITY_MODEL_DIR, C.TRAINING_ARTIFACT_DIR,
              C.SCORE_ARTIFACT_DIR, C.UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)

    database.init_db(C.DB_PATH)
    conn = database.connect(C.DB_PATH)

    def make(model_id, status, job_id, upload_id, retired_at):
        bundle = C.PER_ENTITY_MODEL_DIR / "163_215_0" / model_id
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "bundle.pkl").write_bytes(b"x" * 1024)
        job_dir = C.TRAINING_ARTIFACT_DIR / job_id / "163_215_0"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "training.log").write_text("log")
        up_dir = C.UPLOAD_DIR / upload_id
        up_dir.mkdir(parents=True, exist_ok=True)
        (up_dir / "data.csv").write_text("a,b\n1,2\n")
        conn.execute(
            "INSERT INTO model_registry (model_id, version, status, trained_at,"
            " artifact_path, red_model, amber_model, entity_id, upload_id, job_id)"
            " VALUES (?,1,?,?,?,15,20,'163:215:0',?,?)",
            (model_id, status, retired_at, str(bundle / "bundle.pkl"),
             upload_id, job_id),
        )
        conn.execute(
            "INSERT INTO dataset_uploads (upload_id, stored_path, file_checksum,"
            " config_fingerprint, config_json, status) VALUES (?,?,?,?,'{}','registered')",
            (upload_id, str(up_dir / "data.csv"), upload_id, upload_id),
        )
        conn.execute(
            "INSERT INTO training_jobs (job_id, upload_id, entity_ids, status)"
            " VALUES (?,?,'[\"163:215:0\"]','succeeded')", (job_id, upload_id))
        return bundle, job_dir.parent, up_dir

    old = make("mdl_old", "retired", "job_old", "upl_old", "2020-01-01T00:00:00.000+00:00")
    fresh = make("mdl_fresh", "retired", "job_fresh", "upl_fresh", "2999-01-01T00:00:00.000+00:00")
    live = make("mdl_live", "champion", "job_live", "upl_live", "2020-01-01T00:00:00.000+00:00")
    conn.commit()
    return {"conn": conn, "old": old, "fresh": fresh, "live": live, "root": tmp_path}


def test_dry_run_deletes_nothing(tree):
    from registry.retention import purge_retired

    report = purge_retired(grace_days=14, apply=False, path=C.DB_PATH)

    assert report["models"] == 1
    assert report["bytes"] > 0
    for path in tree["old"]:
        assert path.exists()


def test_apply_removes_only_cold_retired_artifacts(tree):
    from registry.retention import purge_retired

    report = purge_retired(grace_days=14, apply=True, path=C.DB_PATH)

    assert report["models"] == 1
    for path in tree["old"]:
        assert not path.exists(), f"{path} should be purged"
    for path in tree["fresh"] + tree["live"]:
        assert path.exists(), f"{path} must survive"


def test_apply_stamps_the_registry_and_audit_trail(tree):
    from registry.retention import purge_retired

    purge_retired(grace_days=14, apply=True, path=C.DB_PATH)
    conn = tree["conn"]

    row = conn.execute(
        "SELECT status, purged_at, purged_bytes FROM model_registry"
        " WHERE model_id = 'mdl_old'").fetchone()
    assert row["status"] == "retired"          # row kept for audit
    assert row["purged_at"] and row["purged_bytes"] > 0

    kinds = {r["kind"] for r in conn.execute(
        "SELECT kind FROM artifact_purges WHERE model_id = 'mdl_old'")}
    assert {"model_bundle", "training_artifacts", "upload"} <= kinds


def test_purge_is_idempotent(tree):
    from registry.retention import purge_retired

    first = purge_retired(grace_days=14, apply=True, path=C.DB_PATH)
    second = purge_retired(grace_days=14, apply=True, path=C.DB_PATH)

    assert first["models"] == 1
    assert second["models"] == 0 and second["bytes"] == 0


def test_upload_shared_with_a_live_model_is_kept(tree):
    from registry.retention import purge_retired

    conn = tree["conn"]
    conn.execute("UPDATE model_registry SET upload_id = 'upl_old',"
                 " job_id = 'job_old' WHERE model_id = 'mdl_live'")
    conn.commit()

    purge_retired(grace_days=14, apply=True, path=C.DB_PATH)

    assert (C.UPLOAD_DIR / "upl_old").exists()
    assert (C.TRAINING_ARTIFACT_DIR / "job_old").exists()
    assert not (C.PER_ENTITY_MODEL_DIR / "163_215_0" / "mdl_old").exists()


def test_paths_outside_the_platform_tree_are_refused(tree, tmp_path):
    from registry.retention import purge_retired

    outside = tmp_path.parent / "outside_bundle"
    outside.mkdir(exist_ok=True)
    (outside / "bundle.pkl").write_bytes(b"keep me")
    tree["conn"].execute(
        "UPDATE model_registry SET artifact_path = ? WHERE model_id = 'mdl_old'",
        (str(outside / "bundle.pkl"),))
    tree["conn"].commit()

    purge_retired(grace_days=14, apply=True, path=C.DB_PATH)

    assert (outside / "bundle.pkl").exists()
