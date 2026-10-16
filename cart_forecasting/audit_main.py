"""Simulation audit orchestrator.

Runs the full audit described in optimization.md without overwriting the
preserved benchmark in ``outputs/``. Everything is written to
``simulation_audit/``.

Run with:  python audit_main.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
import main as pipeline  # noqa: E402
from audit import audit_plots  # noqa: E402
from audit.attribution import (  # noqa: E402
    attribute_events,
    attribution_summary,
    event_grouping_validation,
)
from audit.audit_report import build_audit_report  # noqa: E402
from audit.audited_simulator import (  # noqa: E402
    ClosedLoopPredictor,
    PolicyConfig,
    ReplayPredictor,
    run_simulation,
)
from audit.audits import (  # noqa: E402
    feasibility_audit,
    interval_irregularity_report,
    leakage_audit,
    timing_audit,
)
from audit.inventory_model import InventoryConfig  # noqa: E402
from audit.order_book import overlap_audit  # noqa: E402
from audit.sensitivity import (  # noqa: E402
    SCENARIOS,
    fragility_ranking,
    scenario_analysis,
    search_policy,
    sensitivity_analysis,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*highly fragmented.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("audit")

AUDIT_DIR = ROOT / "simulation_audit"
FIG_DIR = AUDIT_DIR / "figures"
BASELINE_DIR = AUDIT_DIR / "original_baseline"
for _d in (AUDIT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BASE_INVENTORY = InventoryConfig(
    bay_capacity=C.BAY_CAPACITY, usage="MEDIUM", reserve="LIMITED",
    consumption_model="unmet_demand",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _evaluation_rows(feature_frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Every record of a split, not only the scoreable origins."""
    f = feature_frame.reset_index(drop=True)
    sub = f[f["split"] == split].copy()
    sub["global_index"] = sub.index.to_numpy()
    return sub.reset_index(drop=True)


def _origin_map(rows: pd.DataFrame, origins: pd.DataFrame) -> dict:
    """Map evaluation-row position -> position in the cached prediction arrays."""
    lookup = {
        pd.Timestamp(t): k for k, t in enumerate(pd.to_datetime(origins["decision_time"]))
    }
    out = {}
    for i, t in enumerate(pd.to_datetime(rows["decision_time"])):
        pos = lookup.get(pd.Timestamp(t))
        if pos is not None:
            out[i] = pos
    return out


def _replay_predictor(origins: pd.DataFrame, conformal, red):
    return ReplayPredictor(
        pred_min_6=origins["pred_y_min_6"].to_numpy(),
        pred_min_3=origins["pred_y_min_3"].to_numpy(),
        pred_cart_3=origins["pred_y_cart_3"].to_numpy(),
        pred_cart_6=origins["pred_y_cart_6"].to_numpy(),
        conformal=conformal,
        red=red,
    )


def _policy_from_original(frozen: dict) -> PolicyConfig:
    return PolicyConfig(
        red_threshold=float(frozen["red_threshold"]),
        amber_threshold=float(frozen["amber_threshold"]),
        target_stock=float(frozen["target_stock"]),
        risk_threshold=float(frozen["risk_threshold"]),
        lead_time_min=float(frozen["lead_time_min"]),
        cooldown_min=float(frozen["cooldown_min"]),
        min_carts_per_trip=int(frozen["min_carts_per_trip"]),
        max_carts_per_trip=frozen["max_carts_per_trip"],
        quantity_rule=str(frozen["quantity_rule"]),
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> dict:
    np.random.seed(C.RANDOM_SEED)
    audit: dict = {}

    # ---------------- STEP 1/2 — reproduce the benchmark ----------------
    LOGGER.info("STEP 1/2 — re-running the original pipeline for reproduction")
    results = pipeline.main(write_outputs=False, generate_report=False)
    state = results["state"]
    audit["results"] = results
    audit["state"] = state

    repro = _reproduction_check(results)
    audit["reproduction"] = repro
    LOGGER.info(
        "Reproduction: %d/%d reported metrics matched",
        int(repro["match"].sum()), len(repro),
    )

    # ---------------- STEP 3 — temporal leakage ----------------
    LOGGER.info("STEP 3 — temporal leakage audit")
    leak = leakage_audit(state, results)
    audit["leakage"] = leak
    LOGGER.info(
        "Leakage audit: %d PASS / %d FAIL",
        int((leak["status"] == "PASS").sum()), int((leak["status"] == "FAIL").sum()),
    )

    # ---------------- Evaluation windows ----------------
    feat = state["feature_frame"]
    ho_rows = _evaluation_rows(feat, "holdout")
    va_rows = _evaluation_rows(feat, "validation")
    ho_origins = _origin_map(ho_rows, state["holdout"])
    va_origins = _origin_map(va_rows, state["validation"])
    audit["window"] = {
        "holdout_records": len(ho_rows),
        "holdout_origins": len(ho_origins),
        "validation_records": len(va_rows),
        "validation_origins": len(va_origins),
        "original_holdout_rows": len(state["holdout"]),
    }
    LOGGER.info(
        "Holdout window: %d records, %d scoreable origins (original simulator used %d rows)",
        len(ho_rows), len(ho_origins), len(state["holdout"]),
    )

    red = float(C.RED_THRESHOLD)
    conformal = state["conformal"]
    orig_policy = _policy_from_original(results["frozen_policy"])
    audit["original_policy"] = results["frozen_policy"]

    replay_ho = _replay_predictor(state["holdout"], conformal, red)
    replay_va = _replay_predictor(state["validation"], conformal, red)

    # ---------------- STEP 4-9 — audited replay, same policy ----------------
    LOGGER.info("STEP 4-9 — audited fixed-action replay with the original policy")
    audited = run_simulation(ho_rows, ho_origins, replay_ho, orig_policy, BASE_INVENTORY)
    audit["audited_replay"] = audited

    orders_df = audited["orders"].to_frame()
    audit["timing"] = timing_audit(
        audited["decisions"], orders_df, audited["series"],
        audited["baseline_events"], orig_policy.lead_time_min,
    )
    audit["interval_report"] = interval_irregularity_report(audited["series"])
    audit["pending_audit"] = audited["orders"].audit()
    audit["overlap_audit"] = overlap_audit(audited["orders"])
    audit["conservation"] = _conservation_frame(audited)

    alerts = state["alerts_holdout"].copy()
    alerts["alert_time"] = pd.to_datetime(state["holdout"]["decision_time"].to_numpy())
    att = attribute_events(
        audited["series"], audited["baseline_events"], audited["policy_events"],
        orders_df, alerts, red,
    )
    audit["attribution"] = att
    audit["attribution_summary"] = attribution_summary(att)
    LOGGER.info("Attribution: %s", audit["attribution_summary"])

    audit["event_grouping"] = event_grouping_validation(audited["series"], red)
    audit["feasibility"] = feasibility_audit(
        audited["decisions"], audited["series"], orders_df,
        orig_policy, BASE_INVENTORY, att,
    )

    # ---------------- STEP 11 — closed-loop ----------------
    LOGGER.info("STEP 11 — closed-loop simulation (state-dependent features)")
    closed_pred = ClosedLoopPredictor(
        full_frame=feat,
        feature_cols=state["feature_cols"],
        clip_bounds=state["clip_bounds"],
        estimators={t: state["models"][t][state["selected"][t]].estimator
                    for t in state["selected"]},
        conformal=conformal,
        red=red,
        cart_bounds=state["cart_bounds"],
    )
    closed = run_simulation(ho_rows, ho_origins, closed_pred, orig_policy, BASE_INVENTORY)
    audit["closed_loop"] = closed
    audit["closed_loop_consistency"] = _closed_loop_consistency(
        feat, ho_rows, ho_origins, state, conformal, red
    )

    # ---------------- STEP 12 — sensitivity ----------------
    LOGGER.info("STEP 12 — sensitivity analysis")
    grid, ofat = sensitivity_analysis(
        ho_rows, ho_origins, replay_ho, orig_policy, BASE_INVENTORY
    )
    scen = scenario_analysis(ho_rows, ho_origins, replay_ho, orig_policy, BASE_INVENTORY)
    audit["sensitivity_grid"] = grid
    audit["sensitivity_ofat"] = ofat
    audit["scenarios"] = scen
    audit["fragility"] = fragility_ranking(ofat)
    LOGGER.info("Most fragile assumption: %s", audit["fragility"].iloc[0]["assumption"])

    # ---------------- STEP 13 — policy re-optimisation ----------------
    LOGGER.info("STEP 13 — policy re-optimisation on validation data")
    new_policy, policy_table = search_policy(
        va_rows, va_origins, replay_va, orig_policy, BASE_INVENTORY
    )
    audit["policy_search"] = policy_table
    audit["reoptimised_policy"] = new_policy
    reopt = run_simulation(ho_rows, ho_origins, replay_ho, new_policy, BASE_INVENTORY)
    audit["reoptimised"] = reopt
    reopt_att = attribute_events(
        reopt["series"], reopt["baseline_events"], reopt["policy_events"],
        reopt["orders"].to_frame(), alerts, red,
    )
    audit["reoptimised_attribution"] = reopt_att
    audit["reoptimised_attribution_summary"] = attribution_summary(reopt_att)

    # ---------------- STEP 14 — before/after ----------------
    LOGGER.info("STEP 14 — before/after comparison")
    audit["before_after"] = _before_after(results, audited, att)

    # ---------------- STEP 15/16 — outputs ----------------
    LOGGER.info("STEP 15 — writing CSV outputs and figures")
    _write_csvs(audit)
    audit["figures"] = _make_figures(audit)

    LOGGER.info("STEP 16 — generating the audit report")
    build_audit_report(audit)
    LOGGER.info("Audit complete: %s", AUDIT_DIR)
    return audit


# --------------------------------------------------------------------------
# Sub-steps
# --------------------------------------------------------------------------
REPORTED = {
    "Minimum cart forecast MAE (+40 min)": 3.16,
    "Shortage risk PR-AUC": 0.843,
    "Shortage event recall (%)": 53.8,
    "Detected shortage events": 7,
    "Median warning lead time (min)": 33.4,
    "False alerts per day": 4.53,
    "Recommended replenishment actions": 14,
    "Average top-up (carts)": 10.5,
    "Simulated shortage event reduction (%)": 92.3,
    "Simulated shortage severity reduction (%)": 98.9,
}


def _reproduction_check(results: dict) -> pd.DataFrame:
    mc = results["model_comparison"]
    sel = results["selected_models"]["y_min_6"]
    row = mc[(mc.target == "y_min_6") & (mc.model == sel) & (mc.split == "holdout")].iloc[0]
    ws = results["warning_summary"]
    pm = results["policy_metrics"]
    rec = results["replenishment_decisions"]

    recomputed = {
        "Minimum cart forecast MAE (+40 min)": float(row["MAE"]),
        "Shortage risk PR-AUC": float(results["risk_holdout_metrics"]["PR_AUC"]),
        "Shortage event recall (%)": float(ws["event_recall"] * 100),
        "Detected shortage events": int(ws["n_detected"]),
        "Median warning lead time (min)": float(ws["median_lead_time_min"]),
        "False alerts per day": float(ws["false_alerts_per_day"]),
        "Recommended replenishment actions": int(pm["n_replenishment_actions"]),
        "Average top-up (carts)": float(rec["quantity"].mean()) if len(rec) else np.nan,
        "Simulated shortage event reduction (%)": float(pm["event_reduction_pct"]),
        "Simulated shortage severity reduction (%)": float(pm["severity_reduction_pct"]),
    }
    rows = []
    for k, reported in REPORTED.items():
        got = recomputed[k]
        tol = max(abs(reported) * 0.01, 0.05)
        rows.append(
            {
                "metric": k,
                "reported_in_report": reported,
                "recomputed_from_source": round(float(got), 4),
                "absolute_difference": round(abs(float(got) - reported), 4),
                "match": bool(abs(float(got) - reported) <= tol),
            }
        )
    return pd.DataFrame(rows)


def _conservation_frame(result: dict) -> pd.DataFrame:
    led = result["ledger"].summary()
    s = result["series"]
    rows = [
        {"quantity": "Carts delivered into the bay", "carts": led["carts_delivered"]},
        {"quantity": "Carts consumed by customers", "carts": led["carts_consumed_by_customers"]},
        {"quantity": "Carts absorbed by bay capacity", "carts": led["carts_absorbed_by_capacity"]},
        {"quantity": "Surplus remaining at window end", "carts": led["residual_surplus_at_end"]},
        {"quantity": "Balance error (must be 0)", "carts": led["balance_error"]},
        {"quantity": "Carts drawn from reserve", "carts": led["reserve_drawn"]},
    ]
    out = pd.DataFrame(rows)
    out["check"] = ""
    out.loc[out["quantity"].str.startswith("Balance"), "check"] = (
        "PASS" if led["conserved"] else "FAIL"
    )
    bounds_ok = bool(
        (s["sim_cart"] >= -1e-9).all()
        and (s["sim_cart"] <= (C.BAY_CAPACITY or np.inf) + 1e-9).all()
    )
    out.loc[len(out)] = {
        "quantity": f"0 <= SimulatedCart <= {C.BAY_CAPACITY} enforced",
        "carts": float(s["sim_cart"].max()),
        "check": "PASS" if bounds_ok else "FAIL",
    }
    never_below = bool((s["sim_cart"] >= s["baseline_cart"] - 1e-9).all())
    out.loc[len(out)] = {
        "quantity": "Simulated inventory never below observed baseline",
        "carts": float((s["sim_cart"] - s["baseline_cart"]).min()),
        "check": "PASS" if never_below else "FAIL",
    }
    return out


def _closed_loop_consistency(feat, ho_rows, ho_origins, state, conformal, red) -> dict:
    """With no interventions the closed loop must reproduce the cached forecasts."""
    pred = ClosedLoopPredictor(
        full_frame=feat,
        feature_cols=state["feature_cols"],
        clip_bounds=state["clip_bounds"],
        estimators={t: state["models"][t][state["selected"][t]].estimator
                    for t in state["selected"]},
        conformal=conformal,
        red=red,
        cart_bounds=state["cart_bounds"],
    )
    cached = state["holdout"]["pred_y_min_6"].to_numpy()
    diffs = []
    gi = ho_rows["global_index"].to_numpy()
    for i, pos in list(ho_origins.items())[:120]:
        out = pred.predict(pos, 0.0, global_index=int(gi[i]))
        if out is not None:
            diffs.append(abs(out["pred_min_6"] - cached[pos]))
    return {
        "n_compared": len(diffs),
        "max_abs_difference": float(np.max(diffs)) if diffs else np.nan,
        "mean_abs_difference": float(np.mean(diffs)) if diffs else np.nan,
        "consistent": bool(diffs and np.max(diffs) < 1e-6),
    }


def _before_after(results: dict, audited: dict, att: pd.DataFrame) -> pd.DataFrame:
    o = results["policy_metrics"]
    a = audited["metrics"]
    dec_o = results["replenishment_decisions"]
    dec_a = audited["decisions"]
    att_hist = att[~att["event_id"].astype(str).str.startswith("NEW-")]

    def _mk(metric, orig, aud, explanation):
        try:
            diff = float(aud) - float(orig)
        except (TypeError, ValueError):
            diff = np.nan
        return {
            "metric": metric,
            "original_implementation": orig,
            "audited_implementation": aud,
            "difference": round(diff, 3) if np.isfinite(diff) else "n/a",
            "explanation": explanation,
        }

    rows = [
        _mk("Records in evaluation window", len(results["state"]["holdout"]),
            a["n_records"],
            "The original simulator evaluated only scoreable origins; the audited simulator "
            "replays every record of the holdout window so shortage minutes are complete."),
        _mk("Baseline shortage events", o["baseline_events"], a["baseline_events"],
            "Same grouping rule, but applied to the complete record stream."),
        _mk("Simulated shortage events", o["policy_events"], a["policy_events"],
            "Audited simulator enforces interval-start delivery, traffic-driven consumption and "
            "a finite daily reserve."),
        _mk("Baseline shortage duration (min)", round(o["baseline_shortage_minutes"], 1),
            round(a["baseline_shortage_minutes"], 1),
            "Omitted records restored to the trajectory."),
        _mk("Simulated shortage duration (min)", round(o["policy_shortage_minutes"], 1),
            round(a["policy_shortage_minutes"], 1), "Consequence of the above changes."),
        _mk("Baseline shortage severity (cart-records)",
            round(o["baseline_severity_cart_records"], 1),
            round(a["baseline_severity_cart_records"], 1),
            "Severity now covers every record below RED."),
        _mk("Simulated shortage severity (cart-records)",
            round(o["policy_severity_cart_records"], 1),
            round(a["policy_severity_cart_records"], 1), "Consequence of the above changes."),
        _mk("Simulated event reduction (%)", round(o["event_reduction_pct"], 1),
            round(a["event_reduction_pct"], 1),
            "Headline claim under audit."),
        _mk("Simulated duration reduction (%)", round(o["duration_reduction_pct"], 1),
            round(a["duration_reduction_pct"], 1), "Headline claim under audit."),
        _mk("Simulated severity reduction (%)", round(o["severity_reduction_pct"], 1),
            round(a["severity_reduction_pct"], 1), "Headline claim under audit."),
        _mk("Retrieval trips", o["n_replenishment_actions"], a["n_replenishment_actions"],
            "Pending-aware netting changes how often an order is required."),
        _mk("Carts moved", o["total_carts_recommended"], a["total_carts_recommended"],
            "Quantities are netted against carts already in transit."),
        _mk("Average carts per trip", round(float(o["avg_carts_per_action"]), 2),
            round(float(a["avg_carts_per_action"]), 2) if np.isfinite(a["avg_carts_per_action"]) else "n/a",
            "Driven by the netting rule and the reserve budget."),
        _mk("Unnecessary replenishment actions",
            int(o["n_unnecessary_actions"]),
            int((audited["decisions"]["predicted_min_6"] >= float(results["frozen_policy"]["target_stock"])).sum())
            if len(dec_a) else 0,
            "An action is unnecessary when the forecast minimum already met target stock."),
        _mk("Actions arriving before shortage onset",
            "not measured",
            int(att_hist["delivered_before_onset"].sum()),
            "The original implementation never checked delivery against event onset."),
        _mk("Actions arriving after shortage onset",
            "not measured",
            int((~att_hist["delivered_before_onset"]).sum()),
            "These can no longer be credited with preventing an onset."),
        _mk("Events with traceable prevention", "not measured",
            int((att_hist["prevented_onset"] & (att_hist["n_contributing_orders"] > 0)).sum()),
            "Every prevented event is now linked to specific delivered orders."),
        _mk("Operational cost J", round(o["policy_cost_J"], 1), round(a["policy_cost_J"], 1),
            "Same cost coefficients; different trajectory and workload."),
    ]
    return pd.DataFrame(rows)


def _write_csvs(audit: dict) -> None:
    d = AUDIT_DIR
    audit["leakage"].to_csv(d / "data_leakage_audit.csv", index=False)
    audit["timing"].to_csv(d / "timing_audit.csv", index=False)
    audit["pending_audit"].to_csv(d / "pending_orders_audit.csv", index=False)
    audit["overlap_audit"].to_csv(d / "pending_orders_overlap.csv", index=False)
    audit["conservation"].to_csv(d / "inventory_conservation_audit.csv", index=False)
    audit["attribution"].to_csv(d / "event_action_attribution.csv", index=False)
    audit["feasibility"].to_csv(d / "feasibility_audit.csv", index=False)
    audit["event_grouping"].to_csv(d / "event_grouping_validation.csv", index=False)
    audit["reproduction"].to_csv(d / "baseline_reproduction.csv", index=False)

    hv = audit["audited_replay"]["series"][
        ["DVRDate", "decision_time", "baseline_cart", "sim_cart", "surplus",
         "pending_quantity", "carts_consumed", "Count_IN", "Count_OUT", "interval_duration"]
    ]
    hv.to_csv(d / "historical_vs_simulated.csv", index=False)

    audit["audited_replay"]["decisions"].to_csv(d / "fixed_action_replay.csv", index=False)
    audit["closed_loop"]["decisions"].to_csv(d / "closed_loop_simulation.csv", index=False)

    cols = ["DVRDate", "decision_time", "baseline_cart", "sim_cart", "surplus",
            "pending_quantity", "carts_consumed"]
    merged = audit["audited_replay"]["series"][cols].rename(
        columns={"sim_cart": "replay_sim_cart", "surplus": "replay_surplus",
                 "pending_quantity": "replay_pending", "carts_consumed": "replay_consumed"}
    )
    cl = audit["closed_loop"]["series"][["decision_time", "sim_cart", "surplus"]].rename(
        columns={"sim_cart": "closed_loop_sim_cart", "surplus": "closed_loop_surplus"}
    )
    merged.merge(cl, on="decision_time", how="left").to_csv(
        d / "actual_vs_replay_vs_closed_loop.csv", index=False
    )

    # The actions a production system should actually issue: audited simulator,
    # re-optimised policy, base-case assumptions.
    audit["reoptimised"]["decisions"].to_csv(
        d / "recommended_actions_audited.csv", index=False
    )
    audit["reoptimised"]["orders"].to_frame().to_csv(
        d / "recommended_actions_order_book.csv", index=False
    )
    audit["reoptimised_attribution"].to_csv(
        d / "recommended_actions_event_attribution.csv", index=False
    )

    sens = pd.concat(
        [
            audit["sensitivity_grid"].assign(analysis="grid"),
            audit["sensitivity_ofat"].assign(analysis="one_factor"),
            audit["scenarios"].assign(analysis="scenario"),
        ],
        ignore_index=True,
    )
    sens.to_csv(d / "sensitivity_analysis.csv", index=False)
    audit["fragility"].to_csv(d / "assumption_fragility.csv", index=False)
    audit["policy_search"].to_csv(d / "policy_comparison.csv", index=False)
    audit["before_after"].to_csv(d / "before_after_comparison.csv", index=False)

    summary = {
        "window": audit["window"],
        "reproduction_all_matched": bool(audit["reproduction"]["match"].all()),
        "leakage_failures": int((audit["leakage"]["status"] == "FAIL").sum()),
        "original_policy": audit["original_policy"],
        "reoptimised_policy": audit["reoptimised_policy"].__dict__,
        "attribution_summary": audit["attribution_summary"],
        "closed_loop_consistency": audit["closed_loop_consistency"],
        "audited_metrics": {k: str(v) for k, v in audit["audited_replay"]["metrics"].items()},
        "closed_loop_metrics": {k: str(v) for k, v in audit["closed_loop"]["metrics"].items()},
        "reoptimised_metrics": {k: str(v) for k, v in audit["reoptimised"]["metrics"].items()},
    }
    with open(d / "audit_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


def _make_figures(audit: dict) -> dict:
    red = float(C.RED_THRESHOLD)
    rep = audit["audited_replay"]
    orders = rep["orders"].to_frame()
    figs = {
        "inventory_timeline": audit_plots.inventory_timeline(rep["series"], red, FIG_DIR),
        "replenishment_timing": audit_plots.replenishment_timing(
            rep["series"], orders, red, FIG_DIR
        ),
        "pending_orders": audit_plots.pending_orders(rep["series"], orders, FIG_DIR),
        "event_attribution": audit_plots.event_attribution(audit["attribution"], FIG_DIR),
        "replay_vs_closed_loop": audit_plots.replay_vs_closed_loop(
            rep["metrics"], audit["closed_loop"]["metrics"], FIG_DIR
        ),
        "actual_vs_modes": audit_plots.replay_vs_closed_loop_timeseries(
            rep["series"], audit["closed_loop"]["series"], orders,
            audit["closed_loop"]["orders"].to_frame(), red, FIG_DIR
        ),
        "lead_time_sensitivity": audit_plots.lead_time_sensitivity(
            audit["sensitivity_ofat"], FIG_DIR
        ),
        "target_stock_sensitivity": audit_plots.target_stock_sensitivity(
            audit["sensitivity_ofat"], FIG_DIR
        ),
        "shortage_vs_workload": audit_plots.shortage_vs_workload(
            audit["sensitivity_grid"], audit["scenarios"], FIG_DIR
        ),
        "sensitivity_heatmap": audit_plots.sensitivity_heatmap(
            audit["sensitivity_grid"], FIG_DIR
        ),
        "before_after_comparison": audit_plots.before_after_comparison(
            audit["results"]["policy_metrics"], rep["metrics"], FIG_DIR
        ),
        "operational_cost": audit_plots.operational_cost(
            audit["results"]["policy_metrics"], rep["metrics"], audit["scenarios"], FIG_DIR
        ),
    }
    return figs


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Audit failed: %s", exc)
        raise
