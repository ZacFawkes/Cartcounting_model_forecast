"""Disk and database retention for retired models — registry/retention.py."""
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
            " VALUES (?,1,?,?,?,15,20,'163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8',?,?)",
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

        # Dependent rows: every one of these must follow a retired model and
        # must survive for a live one.
        conn.execute(
            "INSERT INTO calibration_records (calibration_id, job_id, entity_id,"
            " model_id, status, scope) VALUES (?,?,'163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8',?,'fitted','entity')",
            (f"cal_{model_id}", job_id, model_id))
        conn.execute(
            "INSERT INTO entity_model_bindings (entity_id, model_id, role,"
            " job_id, upload_id, retired_at) VALUES ('163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8',?,?,?,?,?)",
            (model_id, "retired" if status == "retired" else "active",
             job_id, upload_id,
             retired_at if status == "retired" else None))
        conn.execute(
            "INSERT INTO backtest_reports (report_id, job_id, entity_id,"
            " model_id, backtest_type, status) VALUES (?,?,'163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8',?,"
            "'forecast_accuracy','completed')",
            (f"rep_{model_id}", job_id, model_id))
        conn.execute(
            "INSERT INTO result_outbox (event_id, event_type, destination,"
            " entity_id, job_id, model_id, payload_json, status, created_at)"
            " VALUES (?,'training','http://x','163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8',?,?,'{}','sent',?)",
            (f"evt_{model_id}", job_id, model_id, retired_at))
        return bundle, job_dir.parent, up_dir

    old = make("mdl_old", "retired", "job_old", "upl_old", "2020-01-01T00:00:00.000+00:00")
    fresh = make("mdl_fresh", "retired", "job_fresh", "upl_fresh", "2999-01-01T00:00:00.000+00:00")
    live = make("mdl_live", "champion", "job_live", "upl_live", "2020-01-01T00:00:00.000+00:00")

    def score(model_id, request_id, created_at):
        """One decision: an immutable CSV plus its scoring_results row."""
        day_dir = C.SCORE_ARTIFACT_DIR / "163_215_0" / created_at[:10]
        day_dir.mkdir(parents=True, exist_ok=True)
        csv = day_dir / f"{request_id}.csv"
        csv.write_text("entity_id,shortage_risk\n163:215:0,0.4\n")
        conn.execute(
            "INSERT INTO scoring_artifacts (request_id, entity_id,"
            " decision_time, artifact_path, model_id, created_at)"
            " VALUES (?,'163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8',?,?,?,?)",
            (request_id, created_at, str(csv), model_id, created_at))
        conn.execute(
            "INSERT INTO scoring_results (request_id, entity_id, decision_time,"
            " model_id, shortage_risk, scored_at) VALUES (?,'163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8',?,?,0.4,?)",
            (request_id, created_at, model_id, created_at))
        conn.execute(
            "INSERT INTO api_requests (request_id, entity_id, endpoint, method,"
            " status, created_at) VALUES (?,'163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8','/score','POST','ok',?)",
            (request_id, created_at))
        return csv

    stale_csv = score("mdl_live", "req-stale", "2020-02-02T00:00:00.000+00:00")
    recent_csv = score("mdl_live", "req-recent", "2999-02-02T00:00:00.000+00:00")
    conn.commit()
    return {"conn": conn, "old": old, "fresh": fresh, "live": live,
            "stale_csv": stale_csv, "recent_csv": recent_csv, "root": tmp_path}


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


# ---------------------------------------------------------------------------
# database rows
# ---------------------------------------------------------------------------

def _count(conn, table, where="1=1", params=()):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}",
                        params).fetchone()[0]


def test_dependent_rows_of_a_retired_model_are_deleted(tree):
    from registry.retention import purge_retired

    conn = tree["conn"]
    purge_retired(grace_days=7, apply=True, path=C.DB_PATH)

    for table in ("calibration_records", "entity_model_bindings",
                  "backtest_reports"):
        assert _count(conn, table, "model_id = 'mdl_old'") == 0, table
    assert _count(conn, "training_jobs", "job_id = 'job_old'") == 0
    assert _count(conn, "dataset_uploads", "upload_id = 'upl_old'") == 0


def test_a_live_model_is_refused_even_when_named_directly(tree):
    from registry.retention import _delete_model_rows, ModelPurge

    conn = tree["conn"]
    purge = ModelPurge(model_id="mdl_live", entity_id="163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8",
                       job_id="job_live", upload_id="upl_live",
                       retired_at="2020-01-01T00:00:00.000+00:00")

    counts = _delete_model_rows(conn, purge, apply=True)

    assert counts == {}
    assert _count(conn, "calibration_records", "model_id = 'mdl_live'") == 1
    assert _count(conn, "training_jobs", "job_id = 'job_live'") == 1


def test_live_model_rows_and_registry_row_survive(tree):
    from registry.retention import run_retention

    conn = tree["conn"]
    run_retention(grace_days=7, apply=True, path=C.DB_PATH, do_vacuum=False)

    for table in ("calibration_records", "entity_model_bindings",
                  "backtest_reports"):
        assert _count(conn, table, "model_id = 'mdl_live'") == 1, table
    assert _count(conn, "training_jobs", "job_id = 'job_live'") == 1
    assert _count(conn, "dataset_uploads", "upload_id = 'upl_live'") == 1
    # the fresh retired model is inside its grace window
    assert _count(conn, "calibration_records", "model_id = 'mdl_fresh'") == 1
    # registry rows are never deleted — audit trail
    assert _count(conn, "model_registry") == 3
    # champion artifacts are untouched
    assert (C.PER_ENTITY_MODEL_DIR / "163_215_0" / "mdl_live").exists()


def test_delivered_outbox_events_of_any_model_age_out(tree):
    from registry.retention import purge_operational_logs

    conn = tree["conn"]
    purge_operational_logs(grace_days=7, apply=True, path=C.DB_PATH)

    # seeded 'sent' and older than the cutoff, for retired and live alike
    assert _count(conn, "result_outbox", "event_id = 'evt_mdl_old'") == 0
    assert _count(conn, "result_outbox", "event_id = 'evt_mdl_live'") == 0
    # dated in the future, so still inside its window
    assert _count(conn, "result_outbox", "event_id = 'evt_mdl_fresh'") == 1


def test_files_only_mode_keeps_every_row(tree):
    from registry.retention import purge_retired

    conn = tree["conn"]
    purge_retired(grace_days=7, apply=True, path=C.DB_PATH, delete_rows=False)

    assert _count(conn, "calibration_records", "model_id = 'mdl_old'") == 1
    assert not (C.PER_ENTITY_MODEL_DIR / "163_215_0" / "mdl_old").exists()


def test_stale_score_csv_goes_but_the_decision_row_stays(tree):
    from registry.retention import purge_operational_logs

    conn = tree["conn"]
    report = purge_operational_logs(grace_days=7, apply=True, path=C.DB_PATH)

    assert not tree["stale_csv"].exists()
    assert tree["recent_csv"].exists()
    assert report["score_files"] == 1
    # scoring_results is the source of truth and is never deleted
    assert _count(conn, "scoring_results", "request_id = 'req-stale'") == 1
    assert _count(conn, "scoring_artifacts", "request_id = 'req-stale'") == 0
    assert _count(conn, "scoring_artifacts", "request_id = 'req-recent'") == 1
    assert _count(conn, "api_requests", "request_id = 'req-stale'") == 0
    assert _count(conn, "api_requests", "request_id = 'req-recent'") == 1


def test_undelivered_outbox_work_is_never_deleted(tree):
    from registry.retention import purge_operational_logs

    conn = tree["conn"]
    for state in ("pending", "sending"):
        conn.execute(
            "INSERT INTO result_outbox (event_id, event_type, destination,"
            " entity_id, model_id, payload_json, status, created_at)"
            " VALUES (?,'score','http://x','163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8','mdl_live','{}',?,"
            "'2020-01-01T00:00:00.000+00:00')", (f"evt_{state}", state))
    conn.commit()

    purge_operational_logs(grace_days=7, apply=True, path=C.DB_PATH)

    assert _count(conn, "result_outbox",
                  "status IN ('pending','sending')") == 2


def test_dry_run_changes_no_row(tree):
    from registry.retention import run_retention

    conn = tree["conn"]
    before = {t: _count(conn, t) for t in
              ("calibration_records", "result_outbox", "api_requests",
               "scoring_artifacts", "training_jobs", "model_registry")}

    report = run_retention(grace_days=7, apply=False, path=C.DB_PATH)

    assert report["rows_total"] > 0
    assert {t: _count(conn, t) for t in before} == before
    assert tree["stale_csv"].exists()


def test_full_sweep_is_idempotent(tree):
    from registry.retention import run_retention

    first = run_retention(grace_days=7, apply=True, path=C.DB_PATH)
    second = run_retention(grace_days=7, apply=True, path=C.DB_PATH)

    assert first["rows_total"] > 0 and first["models"] == 1
    assert second["rows_total"] == 0 and second["models"] == 0


def test_orphan_rows_are_dropped(tree):
    from registry.retention import drop_orphan_rows

    conn = tree["conn"]
    conn.execute(
        "INSERT INTO calibration_records (calibration_id, entity_id, model_id,"
        " status, scope) VALUES ('cal_ghost','163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8','mdl_ghost','fitted','entity')")
    conn.commit()

    report = drop_orphan_rows(apply=True, path=C.DB_PATH)

    assert report["rows"]["calibration_records"] == 1
    assert _count(conn, "calibration_records", "model_id = 'mdl_ghost'") == 0
    assert _count(conn, "calibration_records", "model_id = 'mdl_live'") == 1


def test_orphan_score_files_in_an_old_day_directory_are_swept(tree):
    from registry.retention import purge_operational_logs

    old_day = C.SCORE_ARTIFACT_DIR / "163_215_0" / "2020-03-03"
    old_day.mkdir(parents=True, exist_ok=True)
    orphan = old_day / "req-ghost.csv"
    orphan.write_text("entity_id\n163:215:0\n")

    recent_day = C.SCORE_ARTIFACT_DIR / "163_215_0" / "2999-03-03"
    recent_day.mkdir(parents=True, exist_ok=True)
    recent_orphan = recent_day / "req-ghost2.csv"
    recent_orphan.write_text("entity_id\n163:215:0\n")

    purge_operational_logs(grace_days=7, apply=True, path=C.DB_PATH)

    assert not orphan.exists()
    assert recent_orphan.exists()


# ---------------------------------------------------------------------------
# decisions and replenishment orders
# ---------------------------------------------------------------------------

def _decision(conn, entity_id, decision_time, model_id="mdl_live",
              delivered=None, outbox_status=None, order_status="COMPLETED"):
    conn.execute(
        "INSERT INTO scoring_results (request_id, entity_id, decision_time,"
        " model_id, shortage_risk, scored_at, delivered_at)"
        " VALUES (?,?,?,?,0.4,?,?)",
        (f"req-{decision_time}", entity_id, decision_time, model_id,
         decision_time, delivered))
    conn.execute(
        "INSERT INTO replenishment_orders (order_id, entity_id, decision_time,"
        " quantity, expected_completion_time, status, source)"
        " VALUES (?,?,?,5,?,?,'score')",
        (f"ord-{decision_time}", entity_id, decision_time.replace("T", " "),
         decision_time, order_status))
    if outbox_status:
        conn.execute(
            "INSERT INTO result_outbox (event_id, event_type, destination,"
            " entity_id, model_id, payload_json, status, sent_at, created_at)"
            " VALUES (?,'scoring.recorded','http://x',?,?,'{}',?,?,?)",
            (f"score:{entity_id}:{decision_time}:{model_id}:v1", entity_id,
             model_id, outbox_status, decision_time, decision_time))
    conn.commit()


def test_acknowledged_decisions_and_orders_are_deleted(tree):
    from registry.retention import purge_delivered_decisions

    conn = tree["conn"]
    _decision(conn, "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8", "2020-05-05T10:00:00",
              delivered="2020-05-05T10:00:01.000+00:00")

    report = purge_delivered_decisions(grace_days=7, apply=True, path=C.DB_PATH)

    assert _count(conn, "scoring_results",
                  "decision_time = '2020-05-05T10:00:00'") == 0
    assert _count(conn, "replenishment_orders",
                  "order_id = 'ord-2020-05-05T10:00:00'") == 0
    assert report["rows"]["scoring_results"] == 1
    assert report["rows"]["replenishment_orders"] == 1


def test_an_undelivered_decision_is_never_deleted(tree):
    from registry.retention import purge_delivered_decisions

    conn = tree["conn"]
    for i, state in enumerate(("pending", "sending", "dead_letter")):
        _decision(conn, f"163:215:{state}", f"2020-06-06T10:00:0{i}",
                  outbox_status=state)

    report = purge_delivered_decisions(grace_days=7, apply=True, path=C.DB_PATH)

    for state in ("pending", "sending", "dead_letter"):
        assert _count(conn, "scoring_results",
                      "entity_id = ?", (f"163:215:{state}",)) == 1, state
        # the order is pinned by its undelivered decision too
        assert _count(conn, "replenishment_orders",
                      "entity_id = ?", (f"163:215:{state}",)) == 1, state
    assert report["rows_total"] == 0


def test_delivery_proof_is_backfilled_from_a_sent_outbox_row(tree):
    from registry.retention import purge_delivered_decisions

    conn = tree["conn"]
    _decision(conn, "163:215:9", "2020-07-07T10:00:00", outbox_status="sent")

    purge_delivered_decisions(grace_days=7, apply=True, path=C.DB_PATH)

    assert _count(conn, "scoring_results", "entity_id = '163:215:9'") == 0


def test_an_open_order_survives_at_any_age(tree):
    from registry.retention import purge_delivered_decisions

    conn = tree["conn"]
    _decision(conn, "163:215:8", "2020-08-08T10:00:00",
              delivered="2020-08-08T10:00:01.000+00:00",
              order_status="PENDING")

    purge_delivered_decisions(grace_days=7, apply=True, path=C.DB_PATH)

    assert _count(conn, "replenishment_orders", "status = 'PENDING'") == 1
    assert _count(conn, "scoring_results", "entity_id = '163:215:8'") == 0


def test_a_recent_delivered_decision_is_inside_its_window(tree):
    from registry.retention import purge_delivered_decisions

    conn = tree["conn"]
    _decision(conn, "163:215:7", "2999-09-09T10:00:00",
              delivered="2999-09-09T10:00:01.000+00:00")

    purge_delivered_decisions(grace_days=7, apply=True, path=C.DB_PATH)

    assert _count(conn, "scoring_results", "entity_id = '163:215:7'") == 1
    assert _count(conn, "replenishment_orders",
                  "entity_id = '163:215:7'") == 1


def test_force_undelivered_overrides_the_delivery_gate(tree):
    from registry.retention import purge_delivered_decisions

    conn = tree["conn"]
    _decision(conn, "163:215:6", "2020-10-10T10:00:00",
              outbox_status="pending")

    purge_delivered_decisions(grace_days=7, apply=True, path=C.DB_PATH,
                              force_undelivered=True)

    assert _count(conn, "scoring_results", "entity_id = '163:215:6'") == 0


def test_marking_an_event_sent_stamps_its_decision_as_delivered(tree):
    from integrations import outbox
    from registry.retention import purge_delivered_decisions

    conn = tree["conn"]
    _decision(conn, "163:215:5", "2020-11-11T10:00:00",
              outbox_status="sending")
    row_id = conn.execute(
        "SELECT id FROM result_outbox WHERE entity_id = '163:215:5'"
    ).fetchone()[0]

    outbox.mark_sent(row_id, 200)

    delivered = conn.execute(
        "SELECT delivered_at FROM scoring_results WHERE entity_id = '163:215:5'"
    ).fetchone()[0]
    assert delivered, "delivery must be recorded on the decision itself"

    # and the decision is now eligible, even after its outbox row is gone
    conn.execute("DELETE FROM result_outbox WHERE id = ?", (row_id,))
    conn.commit()
    purge_delivered_decisions(grace_days=7, apply=True, path=C.DB_PATH)
    assert _count(conn, "scoring_results", "entity_id = '163:215:5'") == 0
