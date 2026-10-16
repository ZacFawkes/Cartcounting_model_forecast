"""Unit tests for the audited replenishment simulation.

Covers the ten test categories required by the simulation audit:
delivery timing, pending orders, duplicate prevention, inventory capacity,
event grouping, event attribution, leakage, replay consistency, closed-loop
state updates and inventory conservation.

Run with:  python -m pytest tests/test_audit.py -q
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

from audit.attribution import attribute_events, attribution_summary  # noqa: E402
from audit.audited_simulator import (  # noqa: E402
    PolicyConfig,
    ReplayPredictor,
    run_simulation,
)
from audit.inventory_model import (  # noqa: E402
    InventoryConfig,
    apply_capacity,
    consume_surplus,
    group_shortage_events,
    observed_inventory,
)
from audit.order_book import CANCELLED, COMPLETED, PENDING, OrderBook, overlap_audit  # noqa: E402
from data_loader import prepare  # noqa: E402
from feature_engineering import build_features  # noqa: E402
from target_builder import build_targets, shortage_events  # noqa: E402

STEP = 20.0 / 3.0
RED = 15.0


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _frame(counts, traffic_in=None, start="2026-01-01 08:00:00", step_min=STEP):
    n = len(counts)
    t0 = pd.Timestamp(start)
    starts = [t0 + pd.Timedelta(minutes=step_min * i) for i in range(n)]
    ends = [s + pd.Timedelta(minutes=step_min) for s in starts]
    df = pd.DataFrame(
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
            "Count_IN": traffic_in if traffic_in is not None else [0] * n,
            "Count_OUT": [0] * n,
        }
    )
    out = build_targets(build_features(prepare(df)))
    out["global_index"] = np.arange(len(out))
    return out


def _const_predictor(n, pred_min=10.0, risk=0.99):
    class _P:
        mode = "test"

        def predict(self, origin_pos, surplus_projected, **_):
            pm = pred_min + surplus_projected
            return {
                "pred_min_6": pm,
                "pred_min_3": pm,
                "pred_cart_3": pm,
                "pred_cart_6": pm,
                "lower_min_6": pm - 3.0,
                "risk": risk if surplus_projected < 5 else max(0.0, risk - 0.9),
            }

    return _P()


def _origins(rows):
    return {i: i for i in range(len(rows))}


INV = InventoryConfig(bay_capacity=60.0, usage="MEDIUM", reserve="UNLIMITED",
                      consumption_model="persistent")


# --------------------------------------------------------------------------
# 1. Replenishment delivery timing
# --------------------------------------------------------------------------
def test_delivery_never_credited_before_lead_time_elapses():
    book = OrderBook()
    t0 = pd.Timestamp("2026-01-01 10:00")
    order = book.place(decision_time=t0, quantity=15, lead_time_min=10.0)
    assert order.expected_completion_time == t0 + pd.Timedelta(minutes=10)

    # A record whose interval ENDS at 10:10 began at 10:03 -> must not deliver,
    # because its cart count was aggregated before the carts arrived.
    assert book.deliver_due(
        record_time=t0 + pd.Timedelta(minutes=10),
        record_start_time=t0 + pd.Timedelta(minutes=3),
    ) == []
    assert order.status == PENDING

    # A record whose interval STARTS at 10:10 or later may be credited.
    delivered = book.deliver_due(
        record_time=t0 + pd.Timedelta(minutes=17),
        record_start_time=t0 + pd.Timedelta(minutes=10),
    )
    assert [o.order_id for o in delivered] == [order.order_id]
    assert order.status == COMPLETED
    assert order.delivered_quantity == 15
    assert order.delivery_record_time >= order.expected_completion_time


def test_simulated_inventory_unchanged_before_delivery():
    rows = _frame([10] * 20)
    pol = PolicyConfig(target_stock=25, risk_threshold=0.5, lead_time_min=20.0,
                       cooldown_min=1e9, min_carts_per_trip=3)
    res = run_simulation(rows, _origins(rows), _const_predictor(len(rows)), pol, INV)
    d = res["decisions"]
    assert len(d) == 1
    eff = pd.Timestamp(d.loc[0, "expected_completion_time"])
    s = res["series"]
    before = s[pd.to_datetime(s["DVRDate"]) < eff]
    after = s[pd.to_datetime(s["DVRDate"]) >= eff]
    assert (before["sim_cart"] == before["baseline_cart"]).all()
    assert (after["sim_cart"] > after["baseline_cart"]).any()


def test_irregular_intervals_do_not_advance_delivery():
    interval = 25.0  # intervals far longer than the lead time
    rows = _frame([10] * 14, step_min=interval)
    pol = PolicyConfig(target_stock=25, risk_threshold=0.5, lead_time_min=10.0,
                       cooldown_min=1e9)
    res = run_simulation(rows, _origins(rows), _const_predictor(len(rows)), pol, INV)
    orders = res["orders"].to_frame()
    delays = orders["delivery_delay_min"].dropna()
    # Delivery is never early, and is credited to the end of the first record
    # whose interval *starts* after completion, so lateness measured against the
    # record end is bounded by two intervals.
    assert (delays >= 0).all()
    assert delays.max() <= 2 * interval + 1e-6
    assert res["orders"].audit()["check_delivery_not_early"].all()


# --------------------------------------------------------------------------
# 2. / 3. Pending orders and duplicate prevention
# --------------------------------------------------------------------------
def test_order_lifecycle_states():
    book = OrderBook()
    t0 = pd.Timestamp("2026-01-01 10:00")
    a = book.place(t0, 10, 10.0)
    b = book.place(t0 + pd.Timedelta(minutes=5), 5, 10.0)
    assert book.pending_quantity() == 15
    book.deliver_due(t0 + pd.Timedelta(minutes=12), t0 + pd.Timedelta(minutes=11))
    assert a.status == COMPLETED and b.status == PENDING
    assert book.pending_quantity() == 5
    book.cancel(b, "no longer required")
    assert b.status == CANCELLED and book.pending_quantity() == 0
    audit = book.audit()
    assert audit["all_checks_pass"].all()
    assert set(audit["status"]) == {COMPLETED, CANCELLED}


def test_pending_quantity_is_netted_not_double_ordered():
    """The same shortfall must never be ordered twice."""
    rows = _frame([10] * 40)
    pol = PolicyConfig(target_stock=25, risk_threshold=0.5, lead_time_min=30.0,
                       cooldown_min=0.0, min_carts_per_trip=1, max_carts_per_trip=50)
    res = run_simulation(rows, _origins(rows), _const_predictor(len(rows)), pol, INV)
    d = res["decisions"]
    assert len(d) >= 1
    # First order covers the whole 15-cart shortfall; while it is in transit the
    # netted requirement is zero, so no further order may be issued.
    assert d.loc[0, "quantity"] == 15
    in_transit = d[
        pd.to_datetime(d["decision_time"])
        < pd.Timestamp(d.loc[0, "expected_completion_time"])
    ]
    assert len(in_transit) == 1
    assert res["metrics"]["n_skipped_pending"] > 0


def test_overlap_audit_reports_in_transit_orders():
    book = OrderBook()
    t0 = pd.Timestamp("2026-01-01 10:00")
    book.place(t0, 10, 20.0)
    book.place(t0 + pd.Timedelta(minutes=5), 4, 20.0)
    ov = overlap_audit(book)
    assert ov.loc[0, "n_overlapping"] == 0
    assert ov.loc[1, "n_overlapping"] == 1
    assert ov.loc[1, "pending_quantity_at_decision"] == 10


def test_cooldown_blocks_rapid_reordering():
    rows = _frame([5] * 40)
    pol = PolicyConfig(target_stock=25, risk_threshold=0.5, lead_time_min=5.0,
                       cooldown_min=60.0, max_carts_per_trip=5)
    res = run_simulation(rows, _origins(rows), _const_predictor(len(rows)), pol, INV)
    dt = pd.to_datetime(res["decisions"]["decision_time"]).sort_values()
    gaps = dt.diff().dropna().dt.total_seconds() / 60.0
    assert (gaps >= 60.0).all()
    assert res["metrics"]["n_skipped_cooldown"] > 0


# --------------------------------------------------------------------------
# 4. Inventory capacity
# --------------------------------------------------------------------------
def test_capacity_absorbs_excess_permanently():
    cfg = InventoryConfig(bay_capacity=60.0, consumption_model="persistent")
    surplus, absorbed = apply_capacity(20.0, baseline_cart=55.0, cfg=cfg)
    assert surplus == 5.0 and absorbed == 15.0
    # When the baseline falls again the absorbed carts must NOT reappear.
    surplus2, absorbed2 = apply_capacity(surplus, baseline_cart=20.0, cfg=cfg)
    assert surplus2 == 5.0 and absorbed2 == 0.0
    assert observed_inventory(55.0, 5.0, cfg) == 60.0


def test_simulated_inventory_respects_capacity_and_floor():
    rows = _frame(list(np.clip(np.arange(30) % 25 + 35, 0, 58)))
    cfg = InventoryConfig(bay_capacity=50.0, consumption_model="persistent")
    pol = PolicyConfig(target_stock=60, risk_threshold=0.1, lead_time_min=5.0,
                       cooldown_min=0.0, max_carts_per_trip=20)
    res = run_simulation(rows, _origins(rows), _const_predictor(len(rows)), pol, cfg)
    s = res["series"]
    # The cap bounds the surplus, but an observed baseline above the assumed
    # capacity is never contradicted by the simulator.
    bound = np.maximum(50.0, s["baseline_cart"].to_numpy())
    assert (s["sim_cart"].to_numpy() <= bound + 1e-9).all()
    assert s["sim_cart"].min() >= 0.0
    assert (s["sim_cart"] >= s["baseline_cart"] - 1e-9).all()
    # Where the baseline is below the cap, the surplus is clipped to it.
    under = s[s["baseline_cart"] < 50.0]
    assert (under["sim_cart"] <= 50.0 + 1e-9).all()


# --------------------------------------------------------------------------
# 5. Shortage event grouping (identical rule for both trajectories)
# --------------------------------------------------------------------------
def test_group_shortage_events_matches_pipeline_implementation():
    counts = [30, 10, 12, 30, 11, 10, 30, 30, 9, 30, 30, 30]
    f = _frame(counts)
    ev_audit = group_shortage_events(
        f["Count"].to_numpy(dtype=float),
        pd.to_datetime(f["DVRDate"]).to_numpy(),
        pd.to_datetime(f["decision_time"]).to_numpy(),
        f["interval_duration"].to_numpy(dtype=float),
        RED,
        recovery_records=2,
    )
    ev_pipeline = shortage_events(f, red=int(RED), recovery_records=2)
    assert len(ev_audit) == len(ev_pipeline)
    assert list(ev_audit["min_cart"]) == list(ev_pipeline["min_cart"])
    assert list(ev_audit["severity_cart_records"]) == list(
        ev_pipeline["severity_cart_records"]
    )


def test_same_grouping_rule_applied_to_both_trajectories():
    f = _frame([30, 10, 12, 30, 30, 30, 8, 9, 30, 30, 30, 30])
    start = pd.to_datetime(f["DVRDate"]).to_numpy()
    end = pd.to_datetime(f["decision_time"]).to_numpy()
    dur = f["interval_duration"].to_numpy(dtype=float)
    base = f["Count"].to_numpy(dtype=float)
    sim = base + 10.0
    nb = len(group_shortage_events(base, start, end, dur, RED, 2))
    ns = len(group_shortage_events(sim, start, end, dur, RED, 2))
    assert nb == 2 and ns == 0


# --------------------------------------------------------------------------
# 6. Event attribution
# --------------------------------------------------------------------------
def _attribution_fixture(surplus_before_onset: bool):
    f = _frame([30, 30, 30, 10, 9, 30, 30, 30, 30, 30])
    start = pd.to_datetime(f["DVRDate"]).to_numpy()
    end = pd.to_datetime(f["decision_time"]).to_numpy()
    dur = f["interval_duration"].to_numpy(dtype=float)
    base = f["Count"].to_numpy(dtype=float)
    base_ev = group_shortage_events(base, start, end, dur, RED, 2)

    series = f.copy()
    series["baseline_cart"] = base
    boost = np.zeros(len(f))
    if surplus_before_onset:
        boost[3:] = 10.0
    series["surplus"] = boost
    series["sim_cart"] = base + boost
    pol_ev = group_shortage_events(
        series["sim_cart"].to_numpy(), start, end, dur, RED, 2
    )
    onset = pd.Timestamp(base_ev.loc[0, "onset_time"])
    orders = pd.DataFrame(
        [
            {
                "order_id": "ORD-0001",
                "decision_time": onset - pd.Timedelta(minutes=20),
                "quantity": 10,
                "expected_completion_time": onset - pd.Timedelta(minutes=10),
                "status": "COMPLETED",
                "delivered_quantity": 10.0,
                "delivery_record_time": onset - pd.Timedelta(minutes=5),
            }
        ]
    )
    return series, base_ev, pol_ev, orders


def test_attribution_credits_prevention_only_with_a_delivered_order():
    series, base_ev, pol_ev, orders = _attribution_fixture(surplus_before_onset=True)
    att = attribute_events(series, base_ev, pol_ev, orders, None, RED)
    assert len(att) == 1
    row = att.iloc[0]
    assert row["prevented_onset"] is np.True_ or row["prevented_onset"] is True
    assert row["delivered_before_onset"]
    assert row["contributing_previous_orders"] == "ORD-0001"
    assert row["outcome"] == "PREVENTED_ONSET"
    assert "ORD-0001" in row["attribution_explanation"]
    assert attribution_summary(att)["traceable_prevention"] == 1


def test_attribution_refuses_prevention_without_carts_before_onset():
    series, base_ev, pol_ev, orders = _attribution_fixture(surplus_before_onset=False)
    att = attribute_events(series, base_ev, pol_ev, orders, None, RED)
    row = att.iloc[0]
    assert not row["prevented_onset"]
    assert row["outcome"] == "UNCHANGED"
    assert row["residual_shortage_minutes"] > 0
    assert attribution_summary(att)["traceable_prevention"] == 0


def test_attribution_reports_new_events_created_by_the_policy():
    series, base_ev, pol_ev, orders = _attribution_fixture(surplus_before_onset=True)
    fake_new = pd.DataFrame(
        [
            {
                "event_id": 99, "onset_time": pd.Timestamp("2026-01-01 09:30"),
                "end_time": pd.Timestamp("2026-01-01 09:40"), "start_index": 100,
                "end_index": 101, "min_cart": 5.0, "duration_min": 10.0,
                "severity_cart_records": 20.0, "n_records": 2, "mean_cart": 6.0,
                "onset_decision_time": pd.Timestamp("2026-01-01 09:35"),
            }
        ]
    )
    att = attribute_events(series, base_ev, fake_new, orders, None, RED)
    assert (att["outcome"] == "NEW_EVENT_UNDER_POLICY").sum() == 1
    assert attribution_summary(att)["new_events_under_policy"] == 1


# --------------------------------------------------------------------------
# 7. No future information leakage in the simulator
# --------------------------------------------------------------------------
def test_simulation_decisions_do_not_depend_on_future_records():
    """Corrupting the tail of the window must not change earlier decisions."""
    counts = [10] * 30
    rows = _frame(counts)
    pol = PolicyConfig(target_stock=25, risk_threshold=0.5, lead_time_min=10.0,
                       cooldown_min=15.0, max_carts_per_trip=6)
    base = run_simulation(rows, _origins(rows), _const_predictor(len(rows)), pol, INV)

    mutated = rows.copy()
    mutated.loc[mutated.index[-8:], "Count"] = 55
    mutated.loc[mutated.index[-8:], "Count_IN"] = 99
    alt = run_simulation(mutated, _origins(mutated), _const_predictor(len(mutated)), pol, INV)

    cut = pd.Timestamp(rows["decision_time"].iloc[-9])
    a = base["decisions"]
    b = alt["decisions"]
    a = a[pd.to_datetime(a["decision_time"]) < cut]
    b = b[pd.to_datetime(b["decision_time"]) < cut]
    assert len(a) == len(b)
    assert list(a["quantity"]) == list(b["quantity"])
    assert list(a["decision_time"]) == list(b["decision_time"])


def test_replay_predictor_never_sees_future_traffic():
    """Projection uses the predictor's surplus argument only, not later rows."""
    pred = ReplayPredictor([20.0, 20.0], [20.0, 20.0], [20.0, 20.0], [20.0, 20.0],
                           conformal=_FakeConformal(), red=RED)
    out = pred.predict(0, surplus_projected=5.0)
    assert out["pred_min_6"] == 25.0
    assert out["lower_min_6"] == 22.0


class _FakeConformal:
    residuals_ = np.linspace(-5, 5, 101)

    def lower_bound(self, x):
        return np.asarray(x, dtype=float) - 3.0

    def exceedance_probability(self, x, threshold):
        return np.clip((threshold - np.asarray(x, dtype=float)) / 20.0 + 0.5, 0, 1)


# --------------------------------------------------------------------------
# 8. Fixed-action replay consistency
# --------------------------------------------------------------------------
def test_replay_is_deterministic():
    rows = _frame([12] * 25)
    pol = PolicyConfig(target_stock=25, risk_threshold=0.5, lead_time_min=10.0)
    a = run_simulation(rows, _origins(rows), _const_predictor(len(rows)), pol, INV)["metrics"]
    b = run_simulation(rows, _origins(rows), _const_predictor(len(rows)), pol, INV)["metrics"]
    assert a == b


def test_no_action_leaves_trajectory_identical_to_baseline():
    rows = _frame([12] * 25)
    pol = PolicyConfig(risk_threshold=0.99)
    res = run_simulation(rows, _origins(rows), _const_predictor(len(rows), risk=0.1), pol, INV)
    s = res["series"]
    assert len(res["decisions"]) == 0
    assert (s["sim_cart"] == s["baseline_cart"]).all()
    m = res["metrics"]
    assert m["policy_events"] == m["baseline_events"]
    assert m["policy_shortage_minutes"] == m["baseline_shortage_minutes"]
    assert m["total_carts_recommended"] == 0


def test_decisions_only_taken_at_scoreable_origins():
    rows = _frame([10] * 20)
    origins = {i: i for i in range(len(rows)) if i % 5 == 0}
    pol = PolicyConfig(target_stock=25, risk_threshold=0.5, lead_time_min=5.0,
                       cooldown_min=0.0, max_carts_per_trip=4)
    res = run_simulation(rows, origins, _const_predictor(len(rows)), pol, INV)
    assert set(res["decisions"]["row_index"]).issubset(set(origins))
    # Every record is still simulated even though only some are decision points.
    assert len(res["series"]) == len(rows)


# --------------------------------------------------------------------------
# 9. Closed-loop state updates
# --------------------------------------------------------------------------
def test_closed_loop_predictor_receives_simulated_state():
    rows = _frame([10] * 20)
    seen = {}

    class _Recorder:
        mode = "closed_loop_test"

        def set_simulated_count(self, gi, value):
            seen[int(gi)] = float(value)

        def predict(self, origin_pos, surplus_projected, **_):
            return {
                "pred_min_6": 10.0 + surplus_projected,
                "pred_min_3": 10.0 + surplus_projected,
                "pred_cart_3": 10.0 + surplus_projected,
                "pred_cart_6": 10.0 + surplus_projected,
                "lower_min_6": 7.0 + surplus_projected,
                "risk": 0.99,
            }

    pol = PolicyConfig(target_stock=25, risk_threshold=0.5, lead_time_min=5.0,
                       cooldown_min=1e9)
    res = run_simulation(rows, _origins(rows), _Recorder(), pol, INV)
    s = res["series"]
    assert len(seen) == len(rows)
    for gi, sim in zip(s["global_index"], s["sim_cart"]):
        assert seen[int(gi)] == pytest.approx(float(sim))
    assert max(seen.values()) > 10.0  # the boosted state was fed back


def test_predictor_returning_none_skips_the_decision():
    rows = _frame([10] * 12)

    class _NoFeatures:
        mode = "closed_loop_test"

        def predict(self, *a, **k):
            return None

    pol = PolicyConfig(risk_threshold=0.1)
    res = run_simulation(rows, _origins(rows), _NoFeatures(), pol, INV)
    assert len(res["decisions"]) == 0
    assert (res["series"]["sim_cart"] == res["series"]["baseline_cart"]).all()


# --------------------------------------------------------------------------
# 10. Inventory conservation
# --------------------------------------------------------------------------
def test_conservation_ledger_balances():
    rows = _frame(list(np.clip(np.arange(40) % 30 + 8, 0, 58)), traffic_in=[12] * 40)
    cfg = InventoryConfig(bay_capacity=45.0, usage="MEDIUM", reserve="UNLIMITED",
                          consumption_model="unmet_demand")
    pol = PolicyConfig(target_stock=28, risk_threshold=0.4, lead_time_min=10.0,
                       cooldown_min=10.0, max_carts_per_trip=15)
    res = run_simulation(rows, _origins(rows), _const_predictor(len(rows)), pol, cfg)
    led = res["ledger"].summary()
    assert led["conserved"], led
    assert led["balance_error"] == pytest.approx(0.0, abs=1e-9)
    assert led["carts_delivered"] == pytest.approx(
        led["carts_consumed_by_customers"]
        + led["carts_absorbed_by_capacity"]
        + led["residual_surplus_at_end"]
    )
    assert led["carts_delivered"] <= res["decisions"]["quantity"].sum()


def test_no_carts_created_from_nothing():
    rows = _frame([20] * 20)
    pol = PolicyConfig(risk_threshold=0.99)
    res = run_simulation(rows, _origins(rows), _const_predictor(len(rows), risk=0.0), pol, INV)
    assert res["ledger"].delivered == 0.0
    assert (res["series"]["surplus"] == 0).all()


@pytest.mark.parametrize("model", ["persistent", "unmet_demand", "turnover", "halflife"])
def test_consumption_models_never_increase_surplus(model):
    cfg = InventoryConfig(consumption_model=model, usage="MEDIUM",
                          retention_halflife_min=45.0)
    new, consumed = consume_surplus(10.0, baseline_cart=5.0, traffic_in=20.0,
                                    minutes=6.7, cfg=cfg)
    assert 0.0 <= new <= 10.0
    assert consumed == pytest.approx(10.0 - new)
    assert consumed >= 0.0


def test_unmet_demand_model_only_consumes_when_baseline_is_short():
    cfg = InventoryConfig(consumption_model="unmet_demand", usage="MEDIUM")
    # Plenty of baseline carts: demand is served without touching the surplus.
    kept, consumed = consume_surplus(10.0, baseline_cart=40.0, traffic_in=20.0,
                                     minutes=6.7, cfg=cfg)
    assert kept == 10.0 and consumed == 0.0
    # Baseline exhausted: the surplus serves the unmet demand.
    kept2, consumed2 = consume_surplus(10.0, baseline_cart=2.0, traffic_in=20.0,
                                       minutes=6.7, cfg=cfg)
    assert consumed2 == pytest.approx(min(10.0, 0.5 * 20.0 - 2.0))
    assert kept2 == pytest.approx(10.0 - consumed2)
