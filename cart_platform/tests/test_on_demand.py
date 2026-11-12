"""Acceptance tests for on-demand upload, scoped training and entity scoring.

Covers CART_PLATFORM_ON_DEMAND_ENTITY_MODULE_SPEC sections 9 and 10.4.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

import settings as C  # noqa: E402

UPLOAD_CONFIG = {
    "name": "acceptance_batch",
    "timezone": "Asia/Ho_Chi_Minh",
    "entity_defaults": {"max_sequence_gap_min": 15,
                        "cooldown_min": 30, "lead_time_min": 10},
    "entity_overrides": {
        "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8": {"bay_capacity": 60, "red_threshold": 15, "amber_threshold": 20,
                  "target_stock": 28},
        "163:215:1:1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0": {"bay_capacity": 90, "red_threshold": 20, "amber_threshold": 28,
                  "target_stock": 40},
        # 300:2 deliberately omitted: no bay_capacity => must stay untrainable
    },
}
TRAINED, UNTRAINED, NO_CONFIG = "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8", "163:215:1:1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0", "177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001"


@pytest.fixture(scope="module")
def client(isolated_db):
    from fastapi.testclient import TestClient

    from server.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def telemetry(sample_csv) -> pd.DataFrame:
    return pd.read_csv(sample_csv, parse_dates=["DVRDate", "next_DVRDate"])


def _records(df: pd.DataFrame) -> list[dict]:
    return [
        {"DVRDate": r.DVRDate.isoformat(), "next_DVRDate": r.next_DVRDate.isoformat(),
         "Count": float(r.Count), "Count_IN": float(r.Count_IN),
         "Count_OUT": float(r.Count_OUT)}
        for r in df.itertuples()
    ]


def _window(df: pd.DataFrame, kdvr: int, chan: int, n: int = 20,
            offset: int = 60) -> list[dict]:
    s = df[(df.KDVR == kdvr) & (df.ChanNo == chan)].sort_values("DVRDate")
    return _records(s.tail(offset).head(n))


@pytest.fixture(scope="module")
def upload(client, sample_csv) -> dict:
    with open(sample_csv, "rb") as fh:
        r = client.post("/v1/datasets/upload",
                        files={"file": ("sample_multi_entity.csv", fh, "text/csv")},
                        data={"config": json.dumps(UPLOAD_CONFIG)})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def trained_job(client, upload) -> dict:
    """Train ONLY 215:0 and wait for the job to finish."""
    r = client.post("/v1/training/jobs",
                    json={"upload_id": upload["upload_id"], "entity_ids": [TRAINED]})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + 900
    while time.time() < deadline:
        job = client.get(f"/v1/training/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "partial", "failed"):
            return job
        time.sleep(1.0)
    pytest.fail(f"training job {job_id} did not finish in time")


# ==========================================================================
# Spec §9.1 / §9.12 — discovery is scoped to the uploaded file
# ==========================================================================
def test_upload_discovers_exactly_the_pairs_in_this_file(upload):
    found = {e["entity_id"] for e in upload["entities"]}
    assert found == {TRAINED, UNTRAINED, NO_CONFIG}
    assert upload["n_entities"] == 3
    for e in upload["entities"]:
        assert e["score_url"] == (
            f"/v1/sites/{e['site_key']}/entities/{e['kdvr']}/{e['chan_no']}"
            f"/bays/{e['bay_id']}/score")
        assert e["model_status"] == "not_ready"
    assert "THIS upload only" in upload["note"]


def test_discovery_ignores_unrelated_registered_entities(client, isolated_db, upload):
    """Registering an extra entity must not make it appear in this upload."""
    from registry import entities as reg

    reg.upsert_site(999, display_name="unrelated")
    reg.upsert_entity("999:9", kdvr=999, chan_no=9, site_key=999, bay_capacity=50)
    listed = client.get(f"/v1/datasets/{upload['upload_id']}/entities").json()
    assert {e["entity_id"] for e in listed["entities"]} == {TRAINED, UNTRAINED, NO_CONFIG}
    assert "999:9" not in {e["entity_id"] for e in listed["entities"]}


def test_missing_physical_config_blocks_training_without_inventing_it(upload):
    by_id = {e["entity_id"]: e for e in upload["entities"]}
    assert by_id[NO_CONFIG]["trainable"] is False
    assert any("bay_capacity" in r for r in by_id[NO_CONFIG]["blocking_reasons"])
    assert by_id[TRAINED]["trainable"] is True


def test_upload_is_idempotent_by_checksum_and_config(client, sample_csv, upload):
    with open(sample_csv, "rb") as fh:
        r = client.post("/v1/datasets/upload",
                        files={"file": ("sample_multi_entity.csv", fh, "text/csv")},
                        data={"config": json.dumps(UPLOAD_CONFIG)})
    assert r.status_code == 200
    assert r.json()["duplicate_of"] == upload["upload_id"]
    assert r.json()["upload_id"] == upload["upload_id"]


def test_upload_rejects_invalid_schema(client, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with open(bad, "rb") as fh:
        r = client.post("/v1/datasets/upload", files={"file": ("bad.csv", fh, "text/csv")},
                        data={"config": "{}"})
    assert r.status_code == 422
    assert "missing required column" in json.dumps(r.json()["detail"])


def test_upload_quality_report_counts_dropped_rows(client, tmp_path, telemetry):
    sub = telemetry[(telemetry.KDVR == 215) & (telemetry.ChanNo == 0)].head(300).copy()
    sub.loc[sub.index[:5], "Count"] = -1                      # impossible values
    sub.loc[sub.index[5:8], "next_DVRDate"] = sub.loc[sub.index[5:8], "DVRDate"]
    path = tmp_path / "dirty.csv"
    sub.to_csv(path, index=False)
    with open(path, "rb") as fh:
        r = client.post("/v1/datasets/upload",
                        files={"file": ("dirty.csv", fh, "text/csv")},
                        data={"config": json.dumps({"name": "dirty"})})
    assert r.status_code == 200
    q = r.json()["quality"]
    assert q["n_negative_values"] >= 5
    assert q["n_incomplete_intervals"] >= 3
    assert q["n_rows_usable"] < q["n_rows_raw"]


# ==========================================================================
# Spec §9.3 — the route exists but is not ready until promotion
# ==========================================================================
def test_score_route_returns_409_until_a_model_is_promoted(client, telemetry, upload):
    r = client.post(f"/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score",
                    json={"records": _window(telemetry, 215, 0)})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "MODEL_NOT_READY"
    assert detail["readiness"]["model_status"] in ("not_ready", "candidate_not_promoted")


def test_unregistered_pair_returns_404(client, telemetry):
    r = client.post("/v1/sites/163/entities/777/5/bays/UNKNOWN_BAY/score",
                    json={"records": _window(telemetry, 215, 0)})
    assert r.status_code == 404


# ==========================================================================
# Spec §9.2 / §9.5 — training is scoped and versioned
# ==========================================================================
def test_training_job_trains_only_the_selected_entity(client, trained_job):
    assert trained_job["status"] == "succeeded", trained_job.get("error")
    assert trained_job["entity_ids"] == [TRAINED]
    assert set(trained_job["results"]) == {TRAINED}
    assert trained_job["results"][TRAINED]["status"] == "candidate_ready"


def test_other_entities_are_untouched_by_that_job(client, trained_job):
    for entity_id in (UNTRAINED, NO_CONFIG):
        site, kdvr, chan, bay = entity_id.split(":")
        readiness = client.get(
            f"/v1/sites/{site}/entities/{kdvr}/{chan}/bays/{bay}/model").json()
        assert readiness["model_status"] == "not_ready"
        assert readiness["active"] is None
        assert readiness["candidates"] == []


def test_training_rejects_entities_outside_the_upload(client, upload):
    r = client.post("/v1/training/jobs",
                    json={"upload_id": upload["upload_id"], "entity_ids": ["163:999:9:bay_x"]})
    assert r.status_code == 422
    assert "163:999:9:bay_x" in r.json()["detail"]["unknown_for_upload"]


def test_pooled_strategy_is_not_silently_applied(client, upload):
    r = client.post("/v1/training/jobs",
                    json={"upload_id": upload["upload_id"], "strategy": "pooled"})
    assert r.status_code == 501
    assert "never silently pooled" in r.json()["detail"]


def test_candidate_is_not_auto_promoted(client, trained_job):
    result = trained_job["results"][TRAINED]
    assert result["promoted"] is False
    readiness = client.get("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/model").json()
    assert readiness["model_status"] == "candidate_not_promoted"
    assert readiness["active"] is None


def test_model_artifacts_are_versioned_and_immutable(trained_job):
    result = trained_job["results"][TRAINED]
    bundle = Path(result["artifact_path"])
    meta = Path(result["metadata_path"])
    assert bundle.exists() and meta.exists()
    assert result["model_id"] in str(bundle)
    assert bundle.stat().st_mode & 0o222 == 0  # read-only
    metadata = json.loads(meta.read_text())
    assert metadata["entity_id"] == TRAINED
    assert metadata["feature_schema_hash"]
    assert metadata["frozen_thresholds"]["red_model"] == 15


# ==========================================================================
# Spec §10.2 — calibration is automatic
# ==========================================================================
def test_calibration_is_created_automatically(client, trained_job):
    cal = trained_job["results"][TRAINED]["calibration"]
    assert cal["status"] == "fitted"
    assert cal["scope"] == "entity"
    assert cal["metrics"]["n_samples"] > 0
    assert cal["metrics"]["n_positive"] > 0

    endpoint = client.get(
        f"/v1/training/jobs/{trained_job['job_id']}/calibration").json()
    assert endpoint["status"] == "fitted"
    assert endpoint["entity_id"] == TRAINED
    assert endpoint["metrics"]["brier"] is not None
    assert endpoint["metrics"]["ece"] is not None
    assert Path(endpoint["reliability_path"]).exists()


def test_calibration_window_is_disjoint_from_training_and_holdout(trained_job):
    """No leakage: train < calibration < holdout, strictly ordered."""
    windows = trained_job["results"][TRAINED]["split_windows"]
    assert windows["train"]["end"] < windows["calibration"]["start"]
    assert windows["calibration"]["end"] < windows["holdout"]["start"]


def test_low_event_entity_does_not_receive_fabricated_calibration(isolated_db,
                                                                  telemetry):
    """A stream with no shortage events must not get invented probabilities."""
    from training.calibration import INSUFFICIENT, SCOPE_NONE, auto_calibrate

    frame = pd.DataFrame({
        "decision_time": pd.date_range("2026-01-01", periods=30, freq="7min"),
        "pred_y_min_6": [40.0] * 30,
        "y_min_6": [40.0] * 30,   # never below RED: one class only
    })
    out = auto_calibrate(
        entity_id="test:0", model_id="m1", job_id="j1", calibration_frame=frame,
        pred_col="pred_y_min_6", target_col="y_min_6", red_threshold=15.0,
        artifact_dir=Path(isolated_db) / "cal_test", allow_global_fallback=False,
    )
    assert out["status"] == INSUFFICIENT
    assert out["scope"] == SCOPE_NONE
    assert out["artifact_path"] is None
    assert any("both classes present" in w for w in out["warnings"])


# ==========================================================================
# Spec §10.1 / §10.4 — exactly three backtests
# ==========================================================================
def test_exactly_three_backtests_are_produced(client, trained_job):
    job_id = trained_job["job_id"]
    body = client.get(f"/v1/training/jobs/{job_id}/backtests").json()
    assert body["backtest_types"] == list(C.BACKTEST_TYPES)
    reports = body["entities"][TRAINED]
    assert set(reports) == set(C.BACKTEST_TYPES)
    for t in C.BACKTEST_TYPES:
        assert reports[t]["status"] == "completed", reports[t]
        assert reports[t]["report_id"]
    ids = {reports[t]["report_id"] for t in C.BACKTEST_TYPES}
    assert len(ids) == 3, "each backtest must have a distinct report id"


def test_each_backtest_is_separately_queryable(client, trained_job):
    job_id = trained_job["job_id"]
    for t in C.BACKTEST_TYPES:
        r = client.get(f"/v1/training/jobs/{job_id}/backtests/{t}")
        assert r.status_code == 200
        body = r.json()
        assert body["backtest_type"] == t
        assert body["entity_id"] == TRAINED
        assert body["model_id"] == trained_job["results"][TRAINED]["model_id"]
        assert body["caveat"]
        for a in body["artifacts"]:
            assert Path(a["uri"]).exists(), f"{t} artifact missing: {a}"
    assert client.get(f"/v1/training/jobs/{job_id}/backtests/nope").status_code == 422


def test_forecast_report_is_measured_not_a_benefit_claim(client, trained_job):
    body = client.get(
        f"/v1/training/jobs/{trained_job['job_id']}/backtests/forecast_accuracy").json()
    m = body["metrics"]
    assert m["n_scored"] > 0
    per = m["per_horizon"]
    assert "min_within_40min" in per and "cart_at_20min" in per
    for horizon in per.values():
        assert horizon["MAE"] >= 0
        assert "baseline_persistence_MAE" in horizon
    assert "NOT a replenishment benefit claim" in body["caveat"]
    assert "PR_AUC" in m["shortage_classification"]


def test_policy_replay_is_labelled_as_a_simulation(client, trained_job):
    body = client.get(
        f"/v1/training/jobs/{trained_job['job_id']}/backtests/policy_replay").json()
    m = body["metrics"]
    assert m["baseline_events"] >= m["policy_events"]
    assert m["ledger_conserved"] is True
    assert body["assumptions"]["mode"] == "replay"
    assert "simulation estimates" in body["caveat"]


def test_robustness_report_shows_the_conservative_scenario(client, trained_job):
    body = client.get(
        f"/v1/training/jobs/{trained_job['job_id']}"
        f"/backtests/closed_loop_robustness").json()
    m = body["metrics"]
    names = {s["scenario"] for s in m["scenarios"]}
    assert "conservative" in names
    assert "closed_loop_audited_default" in names or "audited_default" in names
    assert m["severity_reduction_spread_pct"] >= 0
    assert "no single percentage" in body["caveat"]


def test_artifact_layout_matches_the_spec(trained_job):
    job_id = trained_job["job_id"]
    base = Path(C.TRAINING_ARTIFACT_DIR) / job_id / TRAINED.replace(":", "_")
    expected = [
        "training_metrics.json",
        "calibration/metadata.json",
        "calibration/reliability.csv",
        "backtests/forecast_accuracy/report.json",
        "backtests/forecast_accuracy/predictions.csv",
        "backtests/policy_replay/report.json",
        "backtests/policy_replay/decisions.csv",
        "backtests/closed_loop_robustness/report.json",
        "backtests/closed_loop_robustness/scenarios.csv",
    ]
    for rel in expected:
        assert (base / rel).exists(), f"missing artifact {rel}"


# ==========================================================================
# Promotion
# ==========================================================================
@pytest.fixture(scope="module")
def promoted(client, trained_job) -> str:
    model_id = trained_job["results"][TRAINED]["model_id"]
    r = client.post(f"/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/models/{model_id}/promote",
                    json={"promoted_by": "pytest"})
    assert r.status_code == 200, r.text
    assert r.json()["model_status"] == "ready"
    return model_id


def test_promotion_activates_only_that_entity(client, promoted):
    assert client.get("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/model").json()["model_status"] == "ready"
    assert client.get("/v1/sites/163/entities/215/1/bays/1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0/model").json()["model_status"] == "not_ready"


def test_a_model_cannot_be_promoted_for_a_different_entity(client, promoted):
    r = client.post(f"/v1/sites/163/entities/215/1/bays/1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0/models/{promoted}/promote",
                    json={"promoted_by": "pytest"})
    assert r.status_code == 404


def test_failed_backtest_blocks_promotion(isolated_db):
    """A candidate whose reports did not pass cannot be silently promotable."""
    from training.jobs import _promotion_gates

    ok_cal = {"status": "fitted"}
    good = {t: {"status": "completed"} for t in C.BACKTEST_TYPES}
    assert _promotion_gates(ok_cal, good) == (True, [])

    broken = dict(good)
    broken["policy_replay"] = {"status": "failed", "reason": "simulation error"}
    promotable, reasons = _promotion_gates(ok_cal, broken)
    assert promotable is False
    assert any("policy_replay" in r for r in reasons)

    promotable, reasons = _promotion_gates({"status": "insufficient_data"}, good)
    assert promotable is False
    assert any("calibration" in r for r in reasons)


def test_promotion_gate_rejects_an_unready_candidate(client, isolated_db):
    from registry import bindings as bind
    from registry.database import execute, utc_now

    execute(
        """INSERT OR REPLACE INTO entity_model_bindings
           (entity_id, model_id, calibration_id, role, promotable, blocking_reasons,
            created_at) VALUES (?,?,?,?,?,?,?)""",
        (UNTRAINED, "mdl_broken", None, "candidate", 0,
         '["backtest \'policy_replay\' status is \'failed\'"]', utc_now()),
    )
    r = client.post("/v1/sites/163/entities/215/1/bays/1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0/models/mdl_broken/promote",
                    json={"promoted_by": "pytest"})
    assert r.status_code == 409
    assert r.json()["detail"]["blocking_reasons"]
    assert bind.active_binding(UNTRAINED) is None


# ==========================================================================
# Spec §9.4 / §9.6 / §9.7 / §9.8 — scoring
# ==========================================================================
def test_score_uses_the_entity_specific_model(client, telemetry, promoted):
    r = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score",
                    json={"records": _window(telemetry, 215, 0)})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["entity_id"] == TRAINED
    assert b["model_id"] == promoted
    assert b["calibration_scope"] == "entity"
    assert 0.0 <= b["shortage_risk"] <= 1.0
    assert b["forecast"]["lower_bound_min_within_40min"] <= \
           b["forecast"]["min_within_40min"]


def test_other_entity_cannot_borrow_that_model(client, telemetry, promoted):
    r = client.post("/v1/sites/163/entities/215/1/bays/1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0/score",
                    json={"records": _window(telemetry, 215, 1)})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "MODEL_NOT_READY"


def test_path_and_body_identity_mismatch_is_rejected(client, telemetry, promoted):
    r = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score",
                    json={"records": _window(telemetry, 215, 0), "kdvr": 300})
    assert r.status_code == 422
    assert "does not match path" in r.json()["detail"]


def test_insufficient_history_is_rejected_clearly(client, telemetry, promoted):
    r = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score",
                    json={"records": _window(telemetry, 215, 0, n=5)})
    assert r.status_code == 422
    assert "at least" in r.json()["detail"]


def test_twenty_records_succeed(client, telemetry, promoted):
    r = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score",
                    json={"records": _window(telemetry, 215, 0, n=20, offset=80)})
    assert r.status_code == 200
    assert r.json()["decision_time"]


def test_each_new_decision_writes_one_csv_artifact(client, telemetry, promoted):
    client.post("/v1/sessions/reset", json={"entity_id": TRAINED})
    recs = _window(telemetry, 215, 0, n=20, offset=100)
    first = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score", json={"records": recs}).json()
    assert first["idempotent_replay"] is False
    path = Path(first["artifact_path"])
    assert path.exists() and path.suffix == ".csv"
    assert TRAINED.replace(":", "_") in str(path)
    rows = pd.read_csv(path)
    assert len(rows) == 1
    for col in ("entity_id", "decision_time", "model_id", "calibration_id",
                "shortage_risk", "resolved_config_json", "warnings_json"):
        assert col in rows.columns
    assert rows.iloc[0]["entity_id"] == TRAINED


def test_retry_reuses_the_artifact_and_creates_no_duplicate_order(client, telemetry,
                                                                  promoted):
    client.post("/v1/sessions/reset", json={"entity_id": TRAINED})
    recs = _window(telemetry, 215, 0, n=20, offset=120)
    a = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score", json={"records": recs}).json()
    b = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score", json={"records": recs}).json()
    assert b["idempotent_replay"] is True
    assert b["artifact_path"] == a["artifact_path"]
    assert b["action"]["should_replenish"] is False
    files = list(Path(a["artifact_path"]).parent.glob("*.csv"))
    assert len([f for f in files if f.name.endswith(".csv")]) >= 1
    orders = client.get("/v1/orders", params={"entity_id": TRAINED}).json()["orders"]
    assert len({o["decision_time"] for o in orders}) == len(orders)


def test_artifact_retrieval_endpoint(client, telemetry, promoted):
    recs = _window(telemetry, 215, 0, n=20, offset=140)
    b = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score", json={"records": recs}).json()
    rid = b["artifact_request_id"]
    r = client.get(f"/v1/scoring/artifacts/{rid}")
    assert r.status_code == 200
    assert r.json()["entity_id"] == TRAINED
    assert len(r.json()["rows"]) == 1
    assert client.get("/v1/scoring/artifacts/nope").status_code == 404


def test_interleaved_streams_do_not_contaminate_each_other(sample_csv):
    """Features for one camera stream never read another stream's rows."""
    from data_loader import prepare
    from feature_engineering import build_features

    df = pd.read_csv(sample_csv, parse_dates=["DVRDate", "next_DVRDate"])
    prepared = prepare(df, include_site_in_entity=True, include_bay_in_entity=True)
    assert prepared.groupby("block_id")["entity_id"].nunique().max() == 1

    feats = build_features(prepared)
    for eid in (TRAINED, UNTRAINED, NO_CONFIG):
        _site, kdvr, chan, bay = eid.split(":")
        sub = feats[feats["entity_id"] == eid]
        solo = build_features(prepare(
            df[(df.KDVR == int(kdvr)) & (df.ChanNo == int(chan))
               & (df.BayID == bay)],
            include_site_in_entity=True, include_bay_in_entity=True,
        ))
        probe = 500
        for col in ("cart_mean_9", "cart_lag_9", "cart_momentum_9", "traffic_in_sum_6"):
            assert abs(float(sub.iloc[probe][col]) - float(solo.iloc[probe][col])) < 1e-9


# ==========================================================================
# Spec §9.9 — restart durability
# ==========================================================================
def test_bindings_sessions_and_results_survive_a_restart(client, telemetry, promoted,
                                                         isolated_db):
    from fastapi.testclient import TestClient

    from server.app import app

    before = client.get("/v1/logs/scores",
                        params={"entity_id": TRAINED, "limit": 100}).json()["entries"]
    with TestClient(app) as fresh:
        readiness = fresh.get("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/model").json()
        assert readiness["model_status"] == "ready"
        assert readiness["active"]["model_id"] == promoted
        after = fresh.get("/v1/logs/scores",
                          params={"entity_id": TRAINED, "limit": 100}).json()["entries"]
        assert len(after) >= len(before)
        r = fresh.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score",
                       json={"records": _window(telemetry, 215, 0, n=20, offset=160)})
        assert r.status_code == 200


def test_orphaned_job_is_not_left_marked_success(isolated_db, upload):
    from registry.database import query_one
    from training.jobs import QUEUED, create_job, recover_orphaned_jobs

    job_id = create_job(upload["upload_id"], [TRAINED], "per_entity", {}, "pytest")
    assert query_one("SELECT status FROM training_jobs WHERE job_id=?",
                     (job_id,))["status"] == QUEUED
    recover_orphaned_jobs()
    row = query_one("SELECT * FROM training_jobs WHERE job_id=?", (job_id,))
    assert row["status"] == "failed"
    assert "restart" in row["error"]


# ==========================================================================
# Backward compatibility (spec §9.10)
# ==========================================================================
def test_legacy_score_route_still_works(client, telemetry, promoted):
    r = client.post("/v1/score", json={"entity_id": TRAINED,
                                       "records": _window(telemetry, 215, 0, n=20,
                                                          offset=180)})
    assert r.status_code == 200
    assert r.json()["entity_id"] == TRAINED


def test_existing_registry_routes_still_work(client):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/entities").status_code == 200
    assert client.get("/v1/sites").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/sessions").status_code == 200
