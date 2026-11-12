"""Regression tests for per-entity configuration isolation.

Root cause these cover: the trainer previously read ``RED_THRESHOLD`` /
``AMBER_THRESHOLD`` from the pipeline's module-level constants, and
``build_features`` encoded those same globals into 17 features — so every
entity was trained and frozen at 15/20 regardless of its overrides.

Tests A-I map one-to-one onto the task specification.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

import settings as C  # noqa: E402

# Three entities, three genuinely different threshold sets.
OVERRIDES = {
    "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8": {"bay_capacity": 60, "red_threshold": 15, "amber_threshold": 20,
              "target_stock": 28},
    "163:215:1:1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0": {"bay_capacity": 90, "red_threshold": 20, "amber_threshold": 28,
              "target_stock": 40},
    "177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001": {"bay_capacity": 120, "red_threshold": 25, "amber_threshold": 35,
              "target_stock": 50},
}
EXPECTED = {"163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8": (15, 20), "163:215:1:1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0": (20, 28), "177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001": (25, 35)}

UPLOAD_CONFIG = {
    "name": "entity_config_regression",
    "entity_defaults": {"cooldown_min": 30,
                        "lead_time_min": 10, "max_sequence_gap_min": 15},
    "entity_overrides": OVERRIDES,
}


@pytest.fixture(scope="module")
def client(isolated_db):
    from fastapi.testclient import TestClient

    from server.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def telemetry(sample_csv) -> pd.DataFrame:
    return pd.read_csv(sample_csv, parse_dates=["DVRDate", "next_DVRDate"])


def _records(df):
    return [{"DVRDate": r.DVRDate.isoformat(),
             "next_DVRDate": r.next_DVRDate.isoformat(),
             "Count": float(r.Count), "Count_IN": float(r.Count_IN),
             "Count_OUT": float(r.Count_OUT)} for r in df.itertuples()]


def _window(df, kdvr, chan, n=20, offset=60):
    s = df[(df.KDVR == kdvr) & (df.ChanNo == chan)].sort_values("DVRDate")
    return _records(s.tail(offset).head(n))


def _wait(client, job_id, timeout=1200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/training/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "partial", "failed"):
            return job
        time.sleep(1.0)
    pytest.fail(f"job {job_id} did not finish")


@pytest.fixture(scope="module")
def upload(client, sample_csv) -> dict:
    with open(sample_csv, "rb") as fh:
        r = client.post("/v1/datasets/upload",
                        files={"file": ("sample.csv", fh, "text/csv")},
                        data={"config": json.dumps(UPLOAD_CONFIG)})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def trained_all(client, upload) -> dict:
    """Train all three entities in one job — the scenario that was broken."""
    r = client.post("/v1/training/jobs",
                    json={"upload_id": upload["upload_id"],
                          "entity_ids": list(EXPECTED)})
    assert r.status_code == 200, r.text
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job.get("results")
    return job


# ==========================================================================
# TEST A — each entity freezes its OWN thresholds
# ==========================================================================
def test_a_each_entity_freezes_its_own_thresholds(trained_all):
    for entity_id, (red, amber) in EXPECTED.items():
        result = trained_all["results"][entity_id]
        assert result["status"] == "candidate_ready", result.get("error")
        frozen = result["frozen_thresholds"]
        assert float(frozen["red_model"]) == red, (
            f"{entity_id} froze red_model={frozen['red_model']}, expected {red}")
        assert float(frozen["amber_model"]) == amber, (
            f"{entity_id} froze amber_model={frozen['amber_model']}, expected {amber}")


def test_a_frozen_values_are_distinct_across_entities(trained_all):
    """The original bug produced 15/20 for all three."""
    seen = {e: (float(trained_all["results"][e]["frozen_thresholds"]["red_model"]),
                float(trained_all["results"][e]["frozen_thresholds"]["amber_model"]))
            for e in EXPECTED}
    assert len(set(seen.values())) == 3, f"thresholds collapsed to globals: {seen}"


def test_a_metadata_and_bundle_agree_with_the_registry(trained_all):
    for entity_id, (red, amber) in EXPECTED.items():
        result = trained_all["results"][entity_id]
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert meta["frozen_thresholds"] == {"red_model": red, "amber_model": amber}
        assert meta["effective_config"]["target_stock"] == \
               OVERRIDES[entity_id]["target_stock"]
        assert meta["effective_config"]["bay_capacity"] == \
               OVERRIDES[entity_id]["bay_capacity"]

        with open(result["artifact_path"], "rb") as fh:
            bundle = pickle.load(fh)
        assert bundle["red_model"] == red
        assert bundle["amber_model"] == amber
        assert bundle["model_config"]["red_model"] == red
        assert bundle["config_hash"]


def test_a_config_snapshot_is_persisted_per_entity(trained_all):
    from training.model_config import get_snapshot

    hashes = set()
    for entity_id, (red, amber) in EXPECTED.items():
        snap = get_snapshot(trained_all["job_id"], entity_id)
        assert snap is not None, f"no snapshot for {entity_id}"
        assert snap["entity_id"] == entity_id
        assert snap["frozen_thresholds"] == {"red_model": red, "amber_model": amber}
        assert snap["effective_config"]["target_stock"] == \
               OVERRIDES[entity_id]["target_stock"]
        assert snap["sources"]["red_model"].startswith("derived_from:")
        assert snap["model_id"] == trained_all["results"][entity_id]["model_id"]
        hashes.add(snap["config_hash"])
    assert len(hashes) == 3, "snapshot hashes must differ per entity"


# ==========================================================================
# TEST B — feature engineering honours the per-entity thresholds
# ==========================================================================
def test_b_build_features_uses_the_supplied_thresholds(telemetry):
    from data_loader import prepare
    from feature_engineering import build_features

    sub = telemetry[(telemetry.KDVR == 215) & (telemetry.ChanNo == 0)].head(400)
    prepared = prepare(sub)

    f15 = build_features(prepared, red_threshold=15, amber_threshold=20)
    f25 = build_features(prepared, red_threshold=25, amber_threshold=35)

    assert (f15["below_red"] == (f15["cart_current"] < 15).astype(int)).all()
    assert (f25["below_red"] == (f25["cart_current"] < 25).astype(int)).all()
    assert (f25["distance_to_red"] == f25["cart_current"] - 25).all()
    assert (f25["distance_to_amber"] == f25["cart_current"] - 35).all()
    assert not f15["below_red"].equals(f25["below_red"]), \
        "threshold features must change with the threshold"
    assert f25.attrs["red_threshold"] == 25


def test_b_default_call_is_unchanged(telemetry):
    """Backward compatibility: no thresholds supplied => pipeline constants."""
    from config import AMBER_THRESHOLD, RED_THRESHOLD
    from data_loader import prepare
    from feature_engineering import build_features

    prepared = prepare(telemetry[(telemetry.KDVR == 215) &
                                 (telemetry.ChanNo == 0)].head(300))
    default = build_features(prepared)
    explicit = build_features(prepared, red_threshold=RED_THRESHOLD,
                              amber_threshold=AMBER_THRESHOLD)
    cols = ["below_red", "below_amber", "ttb_red_3", "frac_below_amber_6"]
    for c in cols:
        assert default[c].equals(explicit[c])


def test_b_trained_features_match_the_frozen_thresholds(trained_all, telemetry):
    """Rebuild one entity's features and confirm they encode its own RED."""
    from data_loader import prepare
    from feature_engineering import build_features

    entity_id = "177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001"
    red, amber = EXPECTED[entity_id]
    sub = telemetry[(telemetry.KDVR == 300) & (telemetry.ChanNo == 2)].head(400)
    feats = build_features(prepare(sub), red_threshold=red, amber_threshold=amber)
    assert (feats["below_red"] == (feats["cart_current"] < red).astype(int)).all()

    wrong = build_features(prepare(sub), red_threshold=15, amber_threshold=20)
    assert not feats["below_red"].equals(wrong["below_red"])


def test_b_shortage_targets_use_the_entity_red(telemetry):
    from data_loader import prepare
    from feature_engineering import build_features
    from target_builder import build_targets

    prepared = prepare(telemetry[(telemetry.KDVR == 300) &
                                 (telemetry.ChanNo == 2)].head(400))
    t25 = build_targets(build_features(prepared, red_threshold=25,
                                       amber_threshold=35), red=25)
    t15 = build_targets(build_features(prepared), red=15)
    m = t25["y_min_6"].notna()
    assert (t25.loc[m, "shortage_next_6"] ==
            (t25.loc[m, "y_min_6"] < 25).astype(float)).all()
    assert t25["shortage_next_6"].sum() >= t15["shortage_next_6"].sum()


# ==========================================================================
# TEST C — all three backtests record the configuration they used
# ==========================================================================
def test_c_backtests_record_frozen_and_effective_config(client, trained_all):
    job_id = trained_all["job_id"]
    for entity_id, (red, amber) in EXPECTED.items():
        for bt in C.BACKTEST_TYPES:
            r = client.get(f"/v1/training/jobs/{job_id}/backtests/{bt}",
                           params={"entity_id": entity_id})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["entity_id"] == entity_id
            assert body["status"] == "completed", body.get("reason")
            frozen = json.loads(body["frozen_config_json"])
            effective = json.loads(body["effective_config_json"])
            assert frozen["red_model"] == red
            assert frozen["amber_model"] == amber
            assert effective["target_stock"] == OVERRIDES[entity_id]["target_stock"]
            assert effective["bay_capacity"] == OVERRIDES[entity_id]["bay_capacity"]
            assert body["config_hash"]


def test_c_forecast_report_uses_the_frozen_red(client, trained_all):
    job_id = trained_all["job_id"]
    for entity_id, (red, _amber) in EXPECTED.items():
        body = client.get(f"/v1/training/jobs/{job_id}/backtests/forecast_accuracy",
                          params={"entity_id": entity_id}).json()
        a = body["assumptions"]
        assert a["red_model_frozen"] == red
        assert a["red_used_for_metrics"] == red
        assert "frozen model RED" in a["note"]


def test_c_policy_replay_uses_the_entity_operational_config(client, trained_all):
    job_id = trained_all["job_id"]
    for entity_id, (red, amber) in EXPECTED.items():
        body = client.get(f"/v1/training/jobs/{job_id}/backtests/policy_replay",
                          params={"entity_id": entity_id}).json()
        pol = body["assumptions"]["policy"]
        inv = body["assumptions"]["inventory"]
        over = OVERRIDES[entity_id]
        assert pol["red_threshold"] == red
        assert pol["amber_threshold"] == amber
        assert pol["target_stock"] == over["target_stock"]
        assert pol["lead_time_min"] == 10
        # risk_threshold is the clustered WARNING band, not a configured value:
        # the replay must run at the operating point the model will serve.
        assert 0.0 < pol["risk_threshold"] < 1.0
        assert inv["bay_capacity"] == over["bay_capacity"]
        assert body["assumptions"]["frozen_model_thresholds"]["red_model"] == red


def test_c_robustness_records_scenario_overrides(client, trained_all):
    job_id = trained_all["job_id"]
    for entity_id, (red, _a) in EXPECTED.items():
        body = client.get(
            f"/v1/training/jobs/{job_id}/backtests/closed_loop_robustness",
            params={"entity_id": entity_id}).json()
        a = body["assumptions"]
        assert a["base_policy"]["red_threshold"] == red
        assert a["frozen_model_thresholds"]["red_model"] == red
        applied = a["scenario_overrides_applied"]
        assert "conservative" in applied
        assert applied["conservative"]["consumption_model"] == "turnover"
        names = {s["scenario"] for s in body["metrics"]["scenarios"]}
        assert "conservative" in names


def test_c_calibration_records_the_red_it_was_fitted_against(client, trained_all):
    for entity_id, (red, _a) in EXPECTED.items():
        body = client.get(f"/v1/training/jobs/{trained_all['job_id']}/calibration",
                          params={"entity_id": entity_id}).json()
        assert body["status"] == "fitted"
        assert float(body["calibration_red"]) == red
        assert body["config_hash"]


# ==========================================================================
# TEST D — changing config after training
# ==========================================================================
def test_d_config_change_does_not_mutate_the_existing_model(client, trained_all,
                                                            upload, telemetry):
    entity_id = "177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001"
    old_model = trained_all["results"][entity_id]["model_id"]
    old_report = client.get(
        f"/v1/training/jobs/{trained_all['job_id']}/backtests/policy_replay",
        params={"entity_id": entity_id}).json()

    # Promote first so the operational path is live, then change RED 25 -> 30.
    client.post(f"/v1/sites/177/entities/300/2/bays/2C2D3E4F_5061_7283_94A5_B6C7D8E9F001/models/{old_model}/promote",
                json={"promoted_by": "pytest"})
    r = client.put("/v1/entities", json={"entity_id": entity_id,
                                         "red_threshold": 30,
                                         "threshold_source": "manual_override"})
    assert r.status_code == 200
    assert r.json()["entity"]["red_threshold"] == 30

    # 1. the existing model stays frozen at 25
    readiness = client.get("/v1/sites/177/entities/300/2/bays/2C2D3E4F_5061_7283_94A5_B6C7D8E9F001/model").json()
    assert readiness["active"]["model"]["red_model"] == 25

    # 2. the historical backtest is unchanged
    again = client.get(
        f"/v1/training/jobs/{trained_all['job_id']}/backtests/policy_replay",
        params={"entity_id": entity_id}).json()
    assert again["assumptions"]["policy"]["red_threshold"] == 25
    assert again["metrics"] == old_report["metrics"]
    assert json.loads(again["frozen_config_json"])["red_model"] == 25

    # 3. operational scoring uses 30 and says so
    body = client.post("/v1/sites/177/entities/300/2/bays/2C2D3E4F_5061_7283_94A5_B6C7D8E9F001/score",
                       json={"records": _window(telemetry, 300, 2)}).json()
    cfg = body["resolved_config"]
    assert cfg["red_threshold"]["value"] == 30
    assert cfg["red_threshold"]["source"] == "entity_config"
    assert cfg["red_model"]["value"] == 25
    assert body["frozen_model_config"]["red_model"] == 25
    assert any("red_threshold=30" in w and "red_model=25" in w
               for w in body["warnings"]), body["warnings"]


def test_d_a_new_training_job_freezes_the_new_value(client, upload, trained_all):
    """A new candidate must pick up RED=30, not inherit the old model's 25."""
    r = client.post("/v1/training/jobs",
                    json={"upload_id": upload["upload_id"],
                          "entity_ids": ["177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001"]})
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job.get("results")
    result = job["results"]["177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001"]
    assert float(result["frozen_thresholds"]["red_model"]) == 30, \
        "new training must freeze the CURRENT entity config"
    assert float(result["frozen_thresholds"]["amber_model"]) == 35

    # The previously promoted model is still the active one and still at 25.
    readiness = client.get("/v1/sites/177/entities/300/2/bays/2C2D3E4F_5061_7283_94A5_B6C7D8E9F001/model").json()
    assert readiness["active"]["model"]["red_model"] == 25
    assert readiness["model_status"] == "ready"


# ==========================================================================
# TEST E — promotion isolation and scoring correctness
# ==========================================================================
def test_e_promotion_affects_only_that_entity(client, trained_all):
    before = {e: client.get(
        "/v1/sites/{}/entities/{}/{}/bays/{}/model".format(*e.split(":"))).json()
        for e in ("163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8", "163:215:1:1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0")}
    model_215_0 = trained_all["results"]["163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8"]["model_id"]
    r = client.post(f"/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/models/{model_215_0}/promote",
                    json={"promoted_by": "pytest"})
    assert r.status_code == 200
    assert client.get("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/model").json()["model_status"] == "ready"
    after_215_1 = client.get("/v1/sites/163/entities/215/1/bays/1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0/model").json()
    assert after_215_1["model_status"] == before["163:215:1:1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0"]["model_status"]
    assert after_215_1["active"] is None


def test_e_scoring_uses_the_right_frozen_config_per_entity(client, telemetry,
                                                           trained_all):
    """215:0 is promoted at 15/20; 300:2 at 25/35. No cross-contamination."""
    b0 = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score",
                     json={"records": _window(telemetry, 215, 0)}).json()
    assert b0["frozen_model_config"]["red_model"] == 15
    assert b0["resolved_config"]["red_model"]["value"] == 15
    assert b0["model_id"] == trained_all["results"]["163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8"]["model_id"]

    b2 = client.post("/v1/sites/177/entities/300/2/bays/2C2D3E4F_5061_7283_94A5_B6C7D8E9F001/score",
                     json={"records": _window(telemetry, 300, 2)}).json()
    assert b2["frozen_model_config"]["red_model"] == 25
    assert b2["model_id"] == trained_all["results"]["177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001"]["model_id"]
    assert b0["model_id"] != b2["model_id"]


def test_e_operational_recommendation_uses_effective_business_config(client,
                                                                     telemetry):
    """Replenishment sizing must use target_stock=50 for 300:2, not 28."""
    client.post("/v1/sessions/reset", json={"entity_id": "177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001"})
    body = client.post("/v1/sites/177/entities/300/2/bays/2C2D3E4F_5061_7283_94A5_B6C7D8E9F001/score",
                       json={"records": _window(telemetry, 300, 2, offset=200),
                             "params": {"risk_threshold": 0.0}}).json()
    cfg = body["resolved_config"]
    assert cfg["target_stock"]["value"] == 50
    assert cfg["bay_capacity"]["value"] == 120
    if body["action"]["should_replenish"]:
        assert body["action"]["quantity"] > 0


def test_e_unpromoted_entity_is_not_scoreable(client, telemetry):
    r = client.post("/v1/sites/163/entities/215/1/bays/1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0/score",
                    json={"records": _window(telemetry, 215, 1)})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "MODEL_NOT_READY"


# ==========================================================================
# TEST F — concurrency isolation
# ==========================================================================
def test_f_concurrent_resolution_is_isolated(isolated_db):
    """Resolving three entities in parallel must not share mutable state."""
    from concurrent.futures import ThreadPoolExecutor

    from training.model_config import resolve_for_training

    def _resolve(entity_id):
        snap = resolve_for_training(entity_id, job_id=f"job_{entity_id}")
        return entity_id, snap.frozen.red_model, snap.frozen.amber_model, \
            snap.operational["target_stock"], snap.config_hash

    from registry.entities import get_entity

    # TEST D may already have changed 300:2; expectations come from the registry
    # so this test asserts isolation, not a fixed snapshot of history.
    current = {e: (float(get_entity(e)["red_threshold"]),
                   float(get_entity(e)["amber_threshold"])) for e in EXPECTED}

    ids = list(EXPECTED) * 6
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_resolve, ids))

    for entity_id, red, amber, target, _h in results:
        assert (red, amber) == current[entity_id], \
            f"{entity_id} resolved {red}/{amber} under concurrency"
        assert target == OVERRIDES[entity_id]["target_stock"]
    hashes = {e: h for e, _r, _a, _t, h in results}
    assert len(set(hashes.values())) == 3


def test_f_snapshots_are_immutable_value_objects(isolated_db):
    from training.model_config import resolve_for_training

    a = resolve_for_training("163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8", "jobA")
    b = resolve_for_training("177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001", "jobB")
    assert a.frozen.red_model == 15
    assert b.frozen.red_model != a.frozen.red_model
    with pytest.raises(Exception):
        a.frozen.red_model = 99          # frozen dataclass
    # Mutating one snapshot's operational dict cannot touch the other.
    a.operational["target_stock"] = 999
    c = resolve_for_training("163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8", "jobC")
    assert c.operational["target_stock"] == 28


# ==========================================================================
# TEST G — legacy artifacts remain loadable
# ==========================================================================
def test_g_legacy_bundle_without_model_config_still_loads(isolated_db, trained_all):
    """Pre-fix bundles have no 'model_config' key; they must not crash."""
    from registry.database import execute, utc_now
    from registry.models import clear_cache, load_model
    from training.model_config import FrozenModelConfig

    src = Path(trained_all["results"]["163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8"]["artifact_path"])
    with open(src, "rb") as fh:
        bundle = pickle.load(fh)
    for key in ("model_config", "config_hash", "effective_config_at_training",
                "red_model", "amber_model"):
        bundle.pop(key, None)

    legacy_dir = Path(isolated_db) / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / "legacy_bundle.pkl"
    with open(legacy_path, "wb") as fh:
        pickle.dump(bundle, fh)

    execute(
        """INSERT OR REPLACE INTO model_registry
           (model_id, version, status, trained_at, artifact_path, feature_count,
            selected_models, entities_trained_on, sites_trained_on, train_window,
            red_model, amber_model, metrics, loso_metrics, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("mdl_legacy", 9999, "candidate", utc_now(), str(legacy_path),
         len(bundle["feature_cols"]), "{}", "[]", "[]", "{}", 15, 20, "{}", "{}",
         "legacy artifact without model_config"),
    )
    clear_cache()
    loaded = load_model("mdl_legacy")
    assert loaded.red_model == 15
    assert loaded.amber_model == 20
    assert loaded.model_config["red_model"] == 15
    assert isinstance(FrozenModelConfig.from_mapping(None, 15, 20), FrozenModelConfig)
    clear_cache()


def test_g_from_mapping_tolerates_partial_metadata():
    from training.model_config import FrozenModelConfig

    cfg = FrozenModelConfig.from_mapping({"red_model": 25}, 15, 20)
    assert cfg.red_model == 25
    assert cfg.amber_model == 20           # fell back
    assert cfg.feature_windows == (3, 6, 9)


# ==========================================================================
# TEST H — receiver metadata without breaking the contract
# ==========================================================================
def test_h_published_events_carry_configuration_metadata(trained_all):
    from integrations import outbox

    required = {"event_id", "event_type", "schema_version", "occurred_at",
                "entity_id", "payload"}
    found = {"training.completed": 0, "calibration.completed": 0,
             "backtest.completed": 0}

    for row in outbox.list_events(limit=500):
        event = json.loads(row["payload_json"])
        assert required.issubset(event), f"event contract broken: {set(event)}"
        payload = event["payload"]
        etype = event["event_type"]
        if etype not in found:
            continue
        found[etype] += 1
        entity_id = event["entity_id"]
        if entity_id not in EXPECTED:
            continue
        red, amber = EXPECTED[entity_id]
        if etype == "training.completed" and payload.get("config_snapshot"):
            # 300:2 is retrained at RED=30 by TEST D, so accept either freeze.
            frozen_red = payload["config_snapshot"]["frozen_thresholds"]["red_model"]
            assert frozen_red in (red, 30.0), (entity_id, frozen_red)
            assert payload["config_hash"]
        if etype == "calibration.completed" and payload.get("calibration_red"):
            assert float(payload["calibration_red"]) in (red, 30.0)
        if etype == "backtest.completed" and payload.get("frozen_model_config"):
            assert payload["frozen_model_config"]["red_model"] in (red, 30.0)
            assert payload["effective_operational_config"]["bay_capacity"] == \
                OVERRIDES[entity_id]["bay_capacity"]

    assert all(v > 0 for v in found.values()), found


def test_h_receiver_accepts_the_enriched_events(receiver_client, trained_all,
                                                monkeypatch):
    import integrations.publisher as pub
    from tests.conftest import _InProcessSession

    monkeypatch.setattr(C, "RESULT_RECEIVER_ENABLED", True)
    monkeypatch.setattr(pub, "RESULT_RECEIVER_ENABLED", True)
    session = _InProcessSession(receiver_client)
    for _ in range(8):
        if pub.flush_once(session=session)["sent"] == 0:
            break

    rows = receiver_client.get("/v1/results/backtest",
                               params={"job_id": trained_all["job_id"],
                                       "entity_id": "177:300:2:2C2D3E4F_5061_7283_94A5_B6C7D8E9F001"}).json()
    assert rows["total"] >= 1
    stored = receiver_client.get(
        f"/v1/results/events/{rows['items'][0]['event_id']}").json()
    assert stored["payload"]["frozen_model_config"]["red_model"] == 25
    assert stored["payload"]["backtest_type"] in C.BACKTEST_TYPES


# ==========================================================================
# TEST I — invalid configuration fails explicitly
# ==========================================================================
def test_i_missing_bay_capacity_fails_the_entity_explicitly(isolated_db):
    from registry import entities as reg
    from training.model_config import InvalidEntityConfig, resolve_for_training

    reg.upsert_site(901, display_name="invalid-cfg")
    reg.upsert_entity("901:0", kdvr=901, chan_no=0, site_key=901,
                      red_threshold=15, amber_threshold=20, target_stock=28)
    with pytest.raises(InvalidEntityConfig) as exc:
        resolve_for_training("901:0", "job_invalid")
    assert "bay_capacity" in str(exc.value)
    assert exc.value.entity_id == "901:0"


def test_i_inverted_thresholds_fail_rather_than_fall_back(isolated_db):
    from registry import entities as reg
    from training.model_config import InvalidEntityConfig, resolve_for_training

    reg.upsert_site(902, display_name="inverted")
    reg.upsert_entity("902:0", kdvr=902, chan_no=0, site_key=902, bay_capacity=100,
                      red_threshold=40, amber_threshold=20, target_stock=50)
    with pytest.raises(InvalidEntityConfig) as exc:
        resolve_for_training("902:0", "job_inverted")
    problems = " ".join(exc.value.problems)
    assert "amber_threshold" in problems and "above red_threshold" in problems


def test_i_target_below_amber_fails(isolated_db):
    from registry import entities as reg
    from training.model_config import InvalidEntityConfig, resolve_for_training

    reg.upsert_site(903, display_name="bad-target")
    reg.upsert_entity("903:0", kdvr=903, chan_no=0, site_key=903, bay_capacity=100,
                      red_threshold=15, amber_threshold=20, target_stock=18)
    with pytest.raises(InvalidEntityConfig) as exc:
        resolve_for_training("903:0", "job_bad_target")
    assert "target_stock" in str(exc.value)


def test_i_capacity_below_target_fails(isolated_db):
    from registry import entities as reg
    from training.model_config import InvalidEntityConfig, resolve_for_training

    reg.upsert_site(904, display_name="small-bay")
    reg.upsert_entity("904:0", kdvr=904, chan_no=0, site_key=904, bay_capacity=20,
                      red_threshold=15, amber_threshold=20, target_stock=50)
    with pytest.raises(InvalidEntityConfig) as exc:
        resolve_for_training("904:0", "job_small_bay")
    assert "bay_capacity" in str(exc.value)


def test_i_training_job_reports_invalid_config_without_falling_back(client, upload,
                                                                   isolated_db):
    """An invalid entity fails explicitly; it never trains on someone else's values."""
    from registry import entities as reg
    from registry.database import execute, utc_now

    entity_id = "163:215:1:1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0"
    reg.upsert_entity(entity_id, red_threshold=50, amber_threshold=20)
    r = client.post("/v1/training/jobs",
                    json={"upload_id": upload["upload_id"], "entity_ids": [entity_id]})
    job = _wait(client, r.json()["job_id"])
    result = job["results"][entity_id]
    assert result["status"] == "invalid_config"
    assert "INVALID_ENTITY_CONFIG" in result["error"]
    assert result.get("model_id") is None
    # restore for any later test in the module
    execute("UPDATE entity_config SET red_threshold=?, amber_threshold=? "
            "WHERE entity_id=?", (20, 28, entity_id))
