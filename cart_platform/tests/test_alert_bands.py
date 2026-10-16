"""Alert bands clustered from the entity's own risk distribution.

``risk_threshold`` and ``critical_risk_threshold`` are no longer configuration.
K-Means (K=3) on the calibrated risk of the calibration split decides where the
bands sit, the result is frozen into the model, and scoring reads it back.
The WARNING band also gates replenishment, so alerting and ordering share one
measured operating point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

import settings as C  # noqa: E402
from training import alert_bands as alb  # noqa: E402

SITE, KDVR, CHAN = 163, 215, 0
ENTITY = f"{SITE}:{KDVR}:{CHAN}"
SCORE_URL = f"/v1/sites/{SITE}/entities/{KDVR}/{CHAN}/score"

UPLOAD_CONFIG = {
    "name": "band_batch",
    "timezone": "Asia/Ho_Chi_Minh",
    "entity_defaults": {"max_sequence_gap_min": 15, "cooldown_min": 30,
                        "lead_time_min": 10},
    "entity_overrides": {
        ENTITY: {"bay_capacity": 60, "red_threshold": 15, "amber_threshold": 20,
                 "target_stock": 28},
    },
}


def _separable(seed=0, n_low=400, n_mid=60, n_high=40):
    """Three groups a clustering should find."""
    rng = np.random.default_rng(seed)
    return np.r_[rng.uniform(0.00, 0.05, n_low),
                 rng.uniform(0.30, 0.40, n_mid),
                 rng.uniform(0.70, 0.90, n_high)]


# --------------------------------------------------------------------------
# A. the clustering itself
# --------------------------------------------------------------------------
def test_three_separable_groups_become_three_bands():
    out = alb.fit_bands(_separable())
    assert out["status"] == alb.FITTED
    assert out["source"] == "model_kmeans"
    w, c = out["bands"]["warning"], out["bands"]["critical"]
    assert 0.05 < w < 0.30, "WARNING must fall in the gap between the groups"
    assert 0.40 < c < 0.70, "CRITICAL must fall in the upper gap"


def test_bands_are_ordered_and_labelled_by_centroid():
    out = alb.fit_bands(_separable())
    names = [cl["label"] for cl in out["clusters"]]
    assert names == ["NORMAL", "WARNING", "CRITICAL"]
    centres = [cl["centre"] for cl in out["clusters"]]
    assert centres == sorted(centres), "clusters must be ordered by centroid"
    assert out["bands"]["warning"] < out["bands"]["critical"]


def test_result_is_deterministic_across_runs():
    """A band is a published number; it cannot wobble between identical runs."""
    risk = _separable()
    first = alb.fit_bands(risk)["bands"]
    for _ in range(4):
        assert alb.fit_bands(risk)["bands"] == first


def test_boundary_sits_between_clusters_not_at_a_centroid():
    out = alb.fit_bands(_separable())
    lo, mid = out["clusters"][0], out["clusters"][1]
    w = out["bands"]["warning"]
    assert lo["risk_max"] <= w <= mid["risk_min"]


# --------------------------------------------------------------------------
# B. the guards — every one of these was observed on real data
# --------------------------------------------------------------------------
def test_flat_distribution_does_not_crash():
    """All-zero risk used to raise 'zero-size array' inside K-Means."""
    out = alb.fit_bands(np.zeros(500))
    assert out["status"] == alb.INSUFFICIENT
    assert "distinct" in out["reason"]
    assert out["source"] == "system_default"


def test_two_distinct_values_cannot_make_three_bands():
    out = alb.fit_bands(np.r_[np.zeros(300), np.ones(200)])
    assert out["status"] == alb.INSUFFICIENT


def test_too_few_samples_is_refused():
    out = alb.fit_bands(_separable(n_low=30, n_mid=10, n_high=5))
    assert out["status"] == alb.INSUFFICIENT
    assert str(C.ALERT_BAND_MIN_SAMPLES) in out["reason"]


def test_no_high_risk_records_is_refused():
    """Measured: a window with no events clusters at 0.006 / 0.042, which would
    mark almost every record elevated."""
    rng = np.random.default_rng(1)
    out = alb.fit_bands(rng.uniform(0.0, 0.05, 500))
    assert out["status"] == alb.INSUFFICIENT
    assert "risk >=" in out["reason"]
    assert out["bands"]["warning"] == pytest.approx(C.DEFAULT_BAND_WARNING)


def test_a_collapsed_warning_band_is_rejected():
    rng = np.random.default_rng(2)
    risk = np.r_[np.zeros(450), rng.uniform(0.001, 0.004, 30),
                 rng.uniform(0.15, 0.3, 25)]
    out = alb.fit_bands(risk)
    if out["status"] == alb.FITTED:
        assert out["bands"]["warning"] >= C.ALERT_BAND_MIN_WARNING
    else:
        assert out["status"] == alb.INSUFFICIENT


def test_a_refusal_never_invents_a_band():
    out = alb.fit_bands(np.zeros(500))
    assert out["bands"] == {"warning": C.DEFAULT_BAND_WARNING,
                            "critical": C.DEFAULT_BAND_CRITICAL}
    assert out["source"] == "system_default"


def test_a_thin_critical_cluster_is_reported_not_hidden():
    rng = np.random.default_rng(3)
    risk = np.r_[rng.uniform(0, 0.05, 600), rng.uniform(0.3, 0.4, 40),
                 rng.uniform(0.8, 0.9, 11)]
    out = alb.fit_bands(risk)
    if out["status"] == alb.FITTED and out["clusters"][2]["share"] < 0.005:
        assert any("CRITICAL cluster" in w for w in out["warnings"])


def test_unfitted_bands_are_not_served():
    assert alb.bands_from_meta(None) == {}
    assert alb.bands_from_meta({"alert_band_status": "insufficient_data",
                                "alert_band_json": '{"bands": {"warning": 0.9}}'}) == {}
    assert alb.bands_from_meta({"alert_band_status": "failed",
                                "alert_band_json": '{"bands": {"warning": 0.9}}'}) == {}


# --------------------------------------------------------------------------
# C. end to end
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(isolated_db):
    from fastapi.testclient import TestClient

    from server.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def promoted(client, sample_csv):
    import time

    with open(sample_csv, "rb") as fh:
        up = client.post("/v1/datasets/upload",
                         files={"file": ("sample.csv", fh, "text/csv")},
                         data={"config": json.dumps(UPLOAD_CONFIG)})
    assert up.status_code in (200, 201), up.text
    job = client.post("/v1/training/jobs",
                      json={"upload_id": up.json()["upload_id"],
                            "entity_ids": [ENTITY], "strategy": "per_entity"})
    assert job.status_code in (200, 201, 202), job.text
    job_id = job.json()["job_id"]
    for _ in range(240):
        st = client.get(f"/v1/training/jobs/{job_id}").json()
        if st["status"] in ("succeeded", "completed", "failed"):
            break
        time.sleep(0.5)
    assert st["status"] in ("succeeded", "completed"), st
    result = st["results"][ENTITY]
    client.post(f"/v1/sites/{SITE}/entities/{KDVR}/{CHAN}"
                f"/models/{result['model_id']}/promote",
                json={"promoted_by": "pytest"})
    return result


def test_training_reports_the_clustered_bands(promoted):
    assert "alert_bands" in promoted
    ab = promoted["alert_bands"]
    assert ab["status"] in (alb.FITTED, alb.INSUFFICIENT)
    assert 0.0 <= ab["bands"]["warning"] < ab["bands"]["critical"] <= 1.0
    if ab["status"] == alb.FITTED:
        assert ab["source"] == "model_kmeans"
        assert len(ab["clusters"]) == 3
        assert sum(c["n"] for c in ab["clusters"]) == ab["diagnostics"]["n_samples"]


def test_bands_are_frozen_on_the_model_row(promoted):
    from registry import models as mreg

    meta = mreg.get_model_meta(promoted["model_id"])
    assert meta["alert_band_status"] in (alb.FITTED, alb.INSUFFICIENT)
    record = json.loads(meta["alert_band_json"])
    assert record["method"] == "kmeans_k3"
    assert record["seed"] == C.ALERT_BAND_SEED


def test_scoring_serves_the_clustered_band(client, promoted, sample_csv):
    df = pd.read_csv(sample_csv, parse_dates=["DVRDate", "next_DVRDate"])
    df = df[(df.KDVR == KDVR) & (df.ChanNo == CHAN)].head(20)
    records = [{"DVRDate": r.DVRDate.isoformat(),
                "next_DVRDate": r.next_DVRDate.isoformat(),
                "Count": float(r.Count), "Count_IN": float(r.Count_IN),
                "Count_OUT": float(r.Count_OUT)} for r in df.itertuples()]

    resp = client.post(SCORE_URL, json={"records": records, "write_to_db": False})
    assert resp.status_code == 200, resp.text
    cfg = resp.json()["resolved_config"]
    assert cfg["risk_threshold"]["source"] in ("model_kmeans", "system_default")
    assert cfg["critical_risk_threshold"]["source"] in (
        "model_kmeans", "system_default")

    from registry import models as mreg

    bands = alb.bands_from_meta(mreg.get_model_meta(promoted["model_id"]))
    if bands:
        assert cfg["risk_threshold"]["value"] == pytest.approx(bands["warning"])
        assert cfg["critical_risk_threshold"]["value"] == pytest.approx(
            bands["critical"])


def test_the_band_also_gates_replenishment(client, promoted, sample_csv):
    """Alerting and ordering must read the same measured number."""
    from registry.resolver import resolve
    from server.service import _alert_cfg, _policy
    from registry import models as mreg

    cfg = resolve(ENTITY, {}, model_meta=mreg.get_model_meta(promoted["model_id"]))
    assert _policy(cfg).risk_threshold == pytest.approx(cfg["risk_threshold"])
    assert _alert_cfg(cfg).warning_risk == pytest.approx(cfg["risk_threshold"])
    assert _alert_cfg(cfg).critical_risk == pytest.approx(
        cfg["critical_risk_threshold"])


def test_entity_config_cannot_hold_a_band(client, promoted, sample_csv):
    """The band is measured; writing it as configuration must not take effect."""
    from registry.resolver import resolve
    from registry import models as mreg

    meta = mreg.get_model_meta(promoted["model_id"])
    before = resolve(ENTITY, {}, model_meta=meta)["risk_threshold"]
    client.put("/v1/entities", json={"entity_id": ENTITY, "kdvr": KDVR,
                                     "chan_no": CHAN, "site_key": SITE,
                                     "risk_threshold": 0.99})
    after = resolve(ENTITY, {}, model_meta=meta)
    assert after["risk_threshold"] == pytest.approx(before)
    assert after.sources["risk_threshold"] != "entity_config"


def test_a_per_call_override_is_allowed_but_announced(promoted):
    """An audited exception stays possible, and is never silent."""
    from registry.resolver import REQUEST, resolve
    from registry import models as mreg

    meta = mreg.get_model_meta(promoted["model_id"])
    res = resolve(ENTITY, {"risk_threshold": 0.42}, model_meta=meta)
    assert res["risk_threshold"] == pytest.approx(0.42)
    assert res.sources["risk_threshold"] == REQUEST
    assert any("overridden for this call" in w for w in res.warnings)


def test_a_model_without_bands_falls_back_and_says_so():
    """A real model row that carries no bands must fall back loudly."""
    from registry.resolver import resolve

    res = resolve(ENTITY, {}, model_meta={"model_id": "mdl_legacy",
                                          "red_model": 15, "amber_model": 20})
    assert res["risk_threshold"] == pytest.approx(C.DEFAULT_BAND_WARNING)
    assert res.sources["risk_threshold"] == "system_default"
    assert any("not clustered" in w for w in res.warnings)


def test_training_time_resolution_does_not_advise_a_retrain():
    """Before the model exists there are no bands yet; telling the trainer to
    retrain would be nonsense."""
    from registry.resolver import resolve

    res = resolve(ENTITY, {}, model_meta=None)
    assert res["risk_threshold"] == pytest.approx(C.DEFAULT_BAND_WARNING)
    assert not any("Retrain" in w for w in res.warnings)


def test_bands_are_never_reported_as_unconfigured():
    """They are measured, not configuration, so they are not an operator gap."""
    from registry.resolver import resolve

    res = resolve(ENTITY, {}, model_meta=None)
    assert "risk_threshold" not in res.unconfigured
    assert "critical_risk_threshold" not in res.unconfigured


# --------------------------------------------------------------------------
# D. site-scoped routes
# --------------------------------------------------------------------------
def test_entity_identity_includes_the_site(promoted):
    assert promoted["entity_id"] == ENTITY
    assert ENTITY.count(":") == 2


def test_site_less_routes_are_gone(client):
    cases = (
        ("post", f"/v1/entities/{KDVR}/{CHAN}/score", {"records": []}),
        ("get", f"/v1/entities/{KDVR}/{CHAN}/model", None),
        ("post", f"/v1/entities/{KDVR}/{CHAN}/models/rollback",
         {"promoted_by": "x"}),
    )
    for method, url, body in cases:
        r = (client.get(url) if body is None
             else getattr(client, method)(url, json=body))
        assert r.status_code == 410, f"{url} -> {r.status_code}"
        detail = r.json()["detail"]
        assert detail["code"] == "ROUTE_MOVED"
        assert "/v1/sites/" in detail["use"]
