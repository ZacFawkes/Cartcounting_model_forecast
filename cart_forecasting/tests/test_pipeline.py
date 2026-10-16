"""Unit tests for the cart forecasting pipeline.

Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from alert_engine import AlertEngine, AlertEngineConfig, evaluate_warnings, run_alert_engine  # noqa: E402
from data_loader import prepare  # noqa: E402
from data_validation import check_feature_leakage, check_split_purging  # noqa: E402
from feature_engineering import build_features  # noqa: E402
from replenishment_optimizer import classify_urgency, recommend_quantity  # noqa: E402
from simulator import _count_events, simulate_policy  # noqa: E402
from target_builder import assign_splits, build_targets, shortage_events  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _synthetic(counts, start="2026-01-01 08:00:00", step_min=20.0 / 3.0, traffic_in=None):
    n = len(counts)
    t0 = pd.Timestamp(start)
    starts = [t0 + pd.Timedelta(minutes=step_min * i) for i in range(n)]
    ends = [s + pd.Timedelta(minutes=step_min) for s in starts]
    return pd.DataFrame(
        {
            "CountID": range(n),
            "DVRDate": starts,
            "next_DVRDate": ends,
            "KDVR": 215,
            "ChanNo": 0,
            "CatID": 4000,
            "AreaID": 69,
            "Count": counts,
            "CNN_Index": range(n),
            "siteKey": 163,
            "Count_IN": traffic_in if traffic_in is not None else [5] * n,
            "Count_OUT": [4] * n,
        }
    )


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    counts = np.clip(30 + np.cumsum(rng.normal(0, 2, 400)), 3, 58).round().astype(int)
    df = prepare(_synthetic(counts.tolist()))
    return build_targets(build_features(df))


# --------------------------------------------------------------------------
# 1. Rolling features
# --------------------------------------------------------------------------
def test_rolling_features_use_only_past_and_current():
    counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    f = build_features(prepare(_synthetic(counts)))

    # Window of 3 ending at index 4 covers rows 2,3,4 -> mean 40, min 30, max 50
    assert f.loc[4, "cart_mean_3"] == pytest.approx(40.0)
    assert f.loc[4, "cart_min_3"] == pytest.approx(30.0)
    assert f.loc[4, "cart_max_3"] == pytest.approx(50.0)
    assert f.loc[4, "cart_range_3"] == pytest.approx(20.0)

    # Lags and momentum
    assert f.loc[5, "cart_lag_1"] == pytest.approx(50.0)
    assert f.loc[5, "cart_lag_3"] == pytest.approx(30.0)
    assert f.loc[5, "cart_momentum_3"] == pytest.approx(30.0)

    # First rows must be NaN, never back-filled from the future
    assert pd.isna(f.loc[0, "cart_mean_3"])
    assert pd.isna(f.loc[1, "cart_lag_3"])

    # Linear series -> slope constant and R^2 == 1
    slope = f.loc[6, "cart_slope_3"]
    expected = 10.0 / (20.0 / 3.0)  # 10 carts per 6.667-minute record
    assert slope == pytest.approx(expected, rel=1e-6)
    assert f.loc[6, "cart_trend_r2_3"] == pytest.approx(1.0)

    # Changing a *future* value must not change a past feature
    counts2 = counts.copy()
    counts2[-1] = 9999
    f2 = build_features(prepare(_synthetic(counts2)))
    assert f2.loc[4, "cart_mean_3"] == pytest.approx(f.loc[4, "cart_mean_3"])
    assert f2.loc[6, "cart_slope_3"] == pytest.approx(f.loc[6, "cart_slope_3"])


def test_rolling_windows_do_not_cross_sequence_breaks():
    df = _synthetic([10, 11, 12, 13, 14, 15])
    df.loc[3:, "DVRDate"] += pd.Timedelta(hours=4)
    df.loc[3:, "next_DVRDate"] += pd.Timedelta(hours=4)
    f = build_features(prepare(df))
    assert f["block_id"].nunique() == 2
    # The first row of the second block has no 3-window available
    assert pd.isna(f.loc[3, "cart_mean_3"])
    assert pd.isna(f.loc[3, "cart_lag_1"])


# --------------------------------------------------------------------------
# 2. Target alignment
# --------------------------------------------------------------------------
def test_target_alignment():
    counts = [50, 40, 30, 20, 10, 25, 35, 45, 55, 50, 48, 47]
    t = build_targets(build_features(prepare(_synthetic(counts))))

    assert t.loc[0, "y_cart_3"] == 20          # counts[3]
    assert t.loc[0, "y_cart_6"] == 35          # counts[6]
    assert t.loc[0, "y_min_3"] == 20           # min(40,30,20)
    assert t.loc[0, "y_min_6"] == 10           # min(40,30,20,10,25,35)
    assert t.loc[0, "shortage_next_6"] == 1.0  # 10 < RED(15)
    assert t.loc[0, "shortage_next_3"] == 0.0  # min over 3 is 20

    # Targets must be NaN where the horizon runs past the end of the data
    assert pd.isna(t["y_cart_6"].iloc[-1])
    assert pd.isna(t["y_min_6"].iloc[-6])  is False
    assert pd.isna(t["y_min_6"].iloc[-1])

    # Minimum target must never exceed either endpoint target
    valid = t[["y_cart_3", "y_cart_6", "y_min_6"]].dropna()
    assert (valid["y_min_6"] <= valid["y_cart_3"]).all()
    assert (valid["y_min_6"] <= valid["y_cart_6"]).all()


def test_minimum_target_is_not_endpoint_interpolation():
    # A dip strictly between the endpoints must be captured by y_min_6.
    counts = [40, 40, 40, 5, 40, 40, 40, 40, 40, 40]
    t = build_targets(build_features(prepare(_synthetic(counts))))
    assert t.loc[0, "y_cart_3"] == 5
    assert t.loc[1, "y_cart_3"] == 40 and t.loc[1, "y_cart_6"] == 40
    assert t.loc[1, "y_min_6"] == 5  # endpoints miss it, the minimum does not


# --------------------------------------------------------------------------
# 3. Temporal leakage
# --------------------------------------------------------------------------
def test_no_temporal_leakage_in_features(frame):
    from feature_engineering import candidate_features

    cols = candidate_features(frame)
    rep = check_feature_leakage(frame.dropna(subset=cols), cols, [3, 6])
    assert not rep.failures, rep.checks

    forbidden = {"y_cart_3", "y_cart_6", "y_min_3", "y_min_6",
                 "shortage_next_3", "shortage_next_6", "horizon_minutes_3"}
    assert not forbidden.intersection(cols)


def test_split_purging_prevents_boundary_overlap(frame):
    f, meta = assign_splits(frame, purge_records=C.PLANNING_HORIZON)
    usable = f[f["y_min_6"].notna() & ~f["purged"]]
    splits = {k: usable[usable["split"] == k] for k in ("train", "validation", "holdout")}
    rep = check_split_purging(splits, C.PLANNING_HORIZON)
    assert not rep.failures, rep.checks
    assert set(meta["train_days"]).isdisjoint(meta["validation_days"])
    assert set(meta["validation_days"]).isdisjoint(meta["holdout_days"])


# --------------------------------------------------------------------------
# 4. Shortage event grouping
# --------------------------------------------------------------------------
def test_shortage_event_grouping_with_recovery_rule():
    # Two sub-RED runs separated by a single recovered record: with a
    # 2-record recovery rule this is ONE event, not two.
    counts = [30, 10, 12, 30, 11, 10, 30, 30, 30, 30]
    f = build_features(prepare(_synthetic(counts)))
    ev = shortage_events(f, red=15, recovery_records=2)
    assert len(ev) == 1
    assert ev.loc[0, "min_cart"] == 10
    assert ev.loc[0, "severity_cart_records"] == (5 + 3 + 4 + 5)

    # With an immediate recovery rule the same series yields two events.
    ev2 = shortage_events(f, red=15, recovery_records=1)
    assert len(ev2) == 2


def test_shortage_events_are_independent_and_ordered():
    counts = [30, 10, 30, 30, 30, 30, 12, 30, 30, 30, 30, 30]
    ev = shortage_events(build_features(prepare(_synthetic(counts))), red=15, recovery_records=2)
    assert len(ev) == 2
    assert ev["onset_time"].is_monotonic_increasing
    assert (ev["recovery_time"] >= ev["onset_time"]).all()


# --------------------------------------------------------------------------
# 5. Replenishment timing and quantity
# --------------------------------------------------------------------------
def test_replenishment_quantity_rules():
    cfg = C.RunConfig(target_stock=25, min_carts_per_trip=3, max_carts_per_trip=20,
                      quantity_rule="point")
    q_point, q_risk, q = recommend_quantity(predicted_min=18.2, lower_bound_min=12.4, cfg=cfg)
    assert q_point == 7          # ceil(25 - 18.2)
    assert q_risk == 13          # ceil(25 - 12.4)
    assert q == q_point

    cfg_risk = cfg.replace(quantity_rule="risk")
    assert recommend_quantity(18.2, 12.4, cfg_risk)[2] == 13

    # No top-up needed when the forecast minimum already exceeds target stock
    assert recommend_quantity(30.0, 28.0, cfg)[2] == 0
    # Minimum trip size is enforced
    assert recommend_quantity(24.5, 24.0, cfg)[2] == 3
    # Maximum trip size is enforced
    assert recommend_quantity(0.0, 0.0, cfg)[2] == 20


def test_urgency_accounts_for_lead_time():
    cfg = C.RunConfig(lead_time_min=10.0)
    # Breach expected in 5 minutes, carts arrive in 10 -> urgent
    assert classify_urgency(12.0, 8.0, minutes_to_predicted_breach=5.0, cfg=cfg) == "URGENT"
    # Same forecast but the breach is 30 minutes away -> actionable in time
    assert classify_urgency(12.0, 8.0, minutes_to_predicted_breach=30.0, cfg=cfg) == "HIGH"
    assert classify_urgency(18.0, 17.0, np.inf, cfg) == "MEDIUM"
    assert classify_urgency(30.0, 28.0, np.inf, cfg) == "LOW"


# --------------------------------------------------------------------------
# 6. Pending replenishment tracking
# --------------------------------------------------------------------------
def _sim_frame(counts):
    f = build_targets(build_features(prepare(_synthetic(counts))))
    return f.dropna(subset=["y_min_6"]).reset_index(drop=True)


LAMBDAS = {"shortage": 10.0, "trip": 5.0, "cart": 0.25, "excess": 0.1}


def test_pending_orders_prevent_double_counting():
    counts = [10] * 30
    f = _sim_frame(counts)
    n = len(f)
    cfg = C.RunConfig(risk_threshold=0.2, cooldown_min=0.0, lead_time_min=10.0,
                      target_stock=25, retention_halflife_min=1e9, bay_capacity=None)
    res = simulate_policy(
        f, np.full(n, 10.0), np.full(n, 8.0), np.full(n, 0.99), cfg, LAMBDAS
    )
    d = res["decisions"]
    assert len(d) > 0
    # No two decisions may be issued while an order is still in transit.
    dt = pd.to_datetime(d["decision_time"]).sort_values()
    gaps = dt.diff().dropna().dt.total_seconds() / 60.0
    assert (gaps >= cfg.lead_time_min).all()
    # Every order becomes effective exactly lead_time_min after the decision.
    eff = pd.to_datetime(d["effective_time"]) - pd.to_datetime(d["decision_time"])
    assert np.allclose(eff.dt.total_seconds() / 60.0, cfg.lead_time_min)


def test_cooldown_blocks_rapid_reordering():
    counts = [10] * 40
    f = _sim_frame(counts)
    n = len(f)
    cfg = C.RunConfig(risk_threshold=0.2, cooldown_min=60.0, lead_time_min=5.0,
                      target_stock=25, retention_halflife_min=1e9, bay_capacity=None)
    res = simulate_policy(f, np.full(n, 10.0), np.full(n, 8.0), np.full(n, 0.99), cfg, LAMBDAS)
    dt = pd.to_datetime(res["decisions"]["decision_time"]).sort_values()
    gaps = dt.diff().dropna().dt.total_seconds() / 60.0
    assert (gaps >= 60.0).all()


def test_replenishment_effect_respects_lead_time_and_capacity():
    counts = [10] * 20
    f = _sim_frame(counts)
    n = len(f)
    cfg = C.RunConfig(risk_threshold=0.2, cooldown_min=1e9, lead_time_min=20.0,
                      target_stock=25, retention_halflife_min=1e9, bay_capacity=18)
    res = simulate_policy(f, np.full(n, 10.0), np.full(n, 8.0), np.full(n, 0.99), cfg, LAMBDAS)
    s = res["series"]
    d = res["decisions"]
    assert len(d) == 1
    eff = pd.Timestamp(d.loc[0, "effective_time"])
    before = s[pd.to_datetime(s["decision_time"]) < eff]
    after = s[pd.to_datetime(s["decision_time"]) >= eff]
    assert (before["replenishment_boost"] == 0).all()   # nothing before the lead time
    assert (after["replenishment_boost"] > 0).any()     # effect after the lead time
    assert s["sim_cart"].max() <= cfg.bay_capacity      # capacity respected


# --------------------------------------------------------------------------
# 7. Alert deduplication
# --------------------------------------------------------------------------
def test_alert_deduplication_and_cooldown():
    cfg = AlertEngineConfig(warning_risk=0.3, critical_risk=0.8, cooldown_min=20.0)
    eng = AlertEngine(cfg)
    t0 = pd.Timestamp("2026-01-01 10:00")
    fired = []
    for k in range(10):
        r = eng.step(t0 + pd.Timedelta(minutes=5 * k), 0.5, 30.0)
        fired.append(r["new_predictive_alert"])
    # Sustained high risk must raise exactly one alert, not ten.
    assert sum(fired) == 1
    assert fired[0] is True


def test_alert_hysteresis_and_escalation():
    cfg = AlertEngineConfig(warning_risk=0.30, critical_risk=0.60,
                            cooldown_min=30.0, hysteresis=0.10)
    eng = AlertEngine(cfg)
    t0 = pd.Timestamp("2026-01-01 10:00")
    a = eng.step(t0, 0.40, 30.0)
    assert a["state"] == "WARNING" and a["new_predictive_alert"]
    # Escalation bypasses the cooldown
    b = eng.step(t0 + pd.Timedelta(minutes=5), 0.70, 30.0)
    assert b["state"] == "CRITICAL" and b["escalation"] and b["new_predictive_alert"]
    # Risk just under the threshold keeps the engine elevated (hysteresis)
    c = eng.step(t0 + pd.Timedelta(minutes=10), 0.25, 30.0)
    assert c["state"] == "WARNING"
    # Clearly below threshold -> recovery
    d = eng.step(t0 + pd.Timedelta(minutes=15), 0.05, 30.0)
    assert d["state"] == "NORMAL" and d["recovery"]


def test_reactive_alerts_are_not_credited_as_predictive():
    cfg = AlertEngineConfig(warning_risk=0.3, critical_risk=0.8, cooldown_min=0.0)
    eng = AlertEngine(cfg)
    r = eng.step(pd.Timestamp("2026-01-01 10:00"), 0.9, observed_cart=5.0)
    assert r["reactive_critical"] is True
    assert r["new_predictive_alert"] is False


def test_warning_lead_time_is_measured_before_onset():
    counts = [40] * 10 + [10] * 3 + [40] * 10
    f = build_features(prepare(_synthetic(counts)))
    risk = np.zeros(len(f))
    risk[7:10] = 0.9  # high risk shortly before the onset at index 10
    cfg = AlertEngineConfig(warning_risk=0.5, critical_risk=0.9, cooldown_min=0.0)
    alerts = run_alert_engine(f, risk, cfg)
    ev = shortage_events(f, red=15, recovery_records=2)
    summary, matched = evaluate_warnings(alerts, ev, f)
    assert summary["n_events"] == 1
    assert summary["n_detected"] == 1
    assert summary["event_recall"] == 1.0
    assert matched.loc[0, "lead_time_min"] > 0
    assert pd.Timestamp(matched.loc[0, "first_warning_time"]) < pd.Timestamp(
        matched.loc[0, "onset_time"]
    )


# --------------------------------------------------------------------------
# 8. Backtest consistency
# --------------------------------------------------------------------------
def test_simulator_event_counter_matches_event_grouper():
    counts = [30, 10, 12, 30, 11, 10, 30, 30, 9, 30, 30, 30]
    f = build_features(prepare(_synthetic(counts)))
    n_ev, n_rec = _count_events(f["Count"].to_numpy(dtype=float), 15, recovery=2)
    grouped = shortage_events(f, red=15, recovery_records=2)
    assert n_ev == len(grouped)
    assert n_rec == (f["Count"] < 15).sum()


def test_simulation_never_reduces_cart_count_below_baseline():
    counts = list(np.clip(np.arange(40) % 30 + 5, 3, 58))
    f = _sim_frame(counts)
    n = len(f)
    cfg = C.RunConfig(risk_threshold=0.3, lead_time_min=10.0, bay_capacity=60)
    res = simulate_policy(f, np.full(n, 12.0), np.full(n, 9.0), np.full(n, 0.9), cfg, LAMBDAS)
    s = res["series"]
    assert (s["sim_cart"] >= s["cart_current"] - 1e-9).all()
    m = res["metrics"]
    assert m["policy_severity_cart_records"] <= m["baseline_severity_cart_records"] + 1e-9
    assert m["total_carts_recommended"] == res["decisions"]["quantity"].sum()


def test_no_policy_action_when_risk_below_threshold():
    counts = [40] * 25
    f = _sim_frame(counts)
    n = len(f)
    cfg = C.RunConfig(risk_threshold=0.5)
    res = simulate_policy(f, np.full(n, 35.0), np.full(n, 30.0), np.full(n, 0.01), cfg, LAMBDAS)
    assert len(res["decisions"]) == 0
    assert res["metrics"]["total_carts_recommended"] == 0
    assert (res["series"]["sim_cart"] == res["series"]["cart_current"]).all()


def test_backtest_is_deterministic():
    counts = list(np.clip(30 + np.sin(np.arange(60) / 3) * 20, 3, 58).round().astype(int))
    f = _sim_frame(counts)
    n = len(f)
    cfg = C.RunConfig(risk_threshold=0.3)
    args = (f, np.full(n, 14.0), np.full(n, 11.0), np.full(n, 0.8), cfg, LAMBDAS)
    a = simulate_policy(*args)["metrics"]
    b = simulate_policy(*args)["metrics"]
    assert a == b
