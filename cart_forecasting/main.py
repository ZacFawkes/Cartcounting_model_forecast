"""End-to-end pipeline: audit -> features -> models -> risk -> policy -> report.

Run with:  python main.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config as C  # noqa: E402
import plots  # noqa: E402
from alert_engine import evaluate_warnings, run_alert_engine  # noqa: E402
from backtesting import (  # noqa: E402
    backtest_forecasts,
    cost_sensitivity,
    robustness_analysis,
    search_alert_thresholds,
    search_policy,
    walk_forward_backtest,
)
from data_loader import (  # noqa: E402
    daily_completeness,
    dataset_overview,
    load_prepared,
    summarise_horizon_timing,
)
from data_validation import (  # noqa: E402
    check_feature_leakage,
    check_split_purging,
    validate_dataset,
)
from feature_engineering import (  # noqa: E402
    apply_value_clipper,
    build_features,
    candidate_features,
    fit_value_clipper,
    select_features,
)
from forecasting import (  # noqa: E402
    ConformalCalibrator,
    PersistenceModel,
    feature_importance,
    make_lgbm,
    make_ridge,
    select_model,
    train_target_models,
)
from report_generator import build_report  # noqa: E402
from shortage_risk import (  # noqa: E402
    ForecastDerivedRisk,
    calibration_table,
    classification_metrics,
    metrics_at_threshold,
    train_risk_classifier,
)
from simulator import simulate_policy  # noqa: E402
from target_builder import (  # noqa: E402
    SHORTAGE_COLS,
    TARGET_COLS,
    assign_splits,
    build_targets,
    modelling_mask,
    shortage_events,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*highly fragmented.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("pipeline")

LAMBDAS = {
    "shortage": C.COST_SHORTAGE,
    "trip": C.COST_TRIP,
    "cart": C.COST_CART_MOVED,
    "excess": C.COST_EXCESS,
}
PATH_COLS = [f"path_p{k}" for k in range(1, C.PLANNING_HORIZON + 1)]
PATH_LOWER_COLS = [f"path_lo_p{k}" for k in range(1, C.PLANNING_HORIZON + 1)]


def _build_predicted_path(
    frame: pd.DataFrame, p3: np.ndarray, p6: np.ndarray, lower_offset: float = 0.0
) -> pd.DataFrame:
    """Interpolate a per-record predicted trajectory C(t+1..t+6).

    Anchors: observed C(t), predicted C(t+3), predicted C(t+6). A pessimistic
    copy shifted by the conformal lower offset is also produced. Both are used
    only for *timing* (when a breach is expected), never as an accuracy claim.
    """
    out = frame.copy()
    c0 = out["cart_current"].to_numpy(dtype=float)
    for k in range(1, C.PLANNING_HORIZON + 1):
        if k <= 3:
            w = k / 3.0
            vals = c0 + w * (p3 - c0)
        else:
            w = (k - 3) / 3.0
            vals = p3 + w * (p6 - p3)
        out[f"path_p{k}"] = vals
        out[f"path_lo_p{k}"] = vals + lower_offset
    return out


def main(write_outputs: bool = True, generate_report: bool = True) -> dict:
    np.random.seed(C.RANDOM_SEED)
    results: dict = {}

    # ------------------------------------------------------------------
    # Step 1 — Audit
    # ------------------------------------------------------------------
    LOGGER.info("STEP 1 — data audit")
    raw = load_prepared(C.RAW_DATA_FILE)
    overview = dataset_overview(raw)
    validation_report = validate_dataset(raw)
    daily = daily_completeness(raw)
    results["overview"] = overview
    results["validation"] = validation_report.to_frame()
    results["daily_completeness"] = daily

    # ------------------------------------------------------------------
    # Steps 2-3 — targets, features, splits
    # ------------------------------------------------------------------
    LOGGER.info("STEP 2/3 — feature engineering and chronological splitting")
    feat = build_features(raw)
    feat = build_targets(feat)
    feat, split_meta = assign_splits(feat)
    results["split_meta"] = split_meta
    results["horizon_timing"] = summarise_horizon_timing(raw, C.HORIZONS)

    candidates = candidate_features(feat)
    train_mask_raw = (feat["split"] == "train") & feat[TARGET_COLS].notna().all(axis=1)
    feature_cols, feature_report = select_features(feat, candidates, train_mask_raw)
    results["feature_report"] = feature_report
    results["n_candidate_features"] = len(candidates)
    results["n_selected_features"] = len(feature_cols)

    mask = modelling_mask(feat, feature_cols)
    model_frame = feat.loc[mask].copy()

    # Robust clipping bounds are learned on the training rows only and then
    # applied to every split, so extreme ratio tails cannot drive forecasts.
    clip_bounds = fit_value_clipper(
        model_frame, feature_cols, model_frame["split"] == "train"
    )
    model_frame = apply_value_clipper(model_frame, clip_bounds)
    results["clip_bounds"] = clip_bounds
    results["n_modelling_rows"] = int(len(model_frame))
    results["n_excluded_unreliable"] = int(
        (~feat["horizon_reliable"].fillna(False) & feat[TARGET_COLS].notna().all(axis=1)).sum()
    )

    splits = {
        name: model_frame.loc[model_frame["split"] == name].copy()
        for name in ("train", "validation", "holdout")
    }
    for k, v in splits.items():
        LOGGER.info("  %s: %d rows (%s -> %s)", k, len(v),
                    v["decision_time"].min(), v["decision_time"].max())

    leak_feat = check_feature_leakage(model_frame, feature_cols, C.HORIZONS)
    leak_split = check_split_purging(splits, C.PLANNING_HORIZON)
    results["leakage_checks"] = pd.concat(
        [leak_feat.to_frame(), leak_split.to_frame()], ignore_index=True
    )
    if leak_feat.failures or leak_split.failures:
        raise RuntimeError(f"Leakage check failed: {leak_feat.failures + leak_split.failures}")

    # ------------------------------------------------------------------
    # Step 4 — forecasting models
    # ------------------------------------------------------------------
    LOGGER.info("STEP 4 — training forecasting models")
    tr, va, ho = splits["train"], splits["validation"], splits["holdout"]
    Xtr, Xva, Xho = tr[feature_cols], va[feature_cols], ho[feature_cols]

    models: dict[str, dict] = {}
    val_tables = []
    selected: dict[str, str] = {}
    for target in TARGET_COLS:
        ytr = tr[target].to_numpy(dtype=float)
        yva = va[target].to_numpy(dtype=float)
        near_va = (yva < C.RED_THRESHOLD + 10) | (
            va["cart_current"].to_numpy() < C.RED_THRESHOLD + 10
        )
        res, table = train_target_models(Xtr, ytr, Xva, yva, target, near_va)
        models[target] = res
        val_tables.append(table)
        selected[target] = select_model(res, target)
        LOGGER.info("  %s -> selected %s", target, selected[target])

    results["validation_model_table"] = pd.concat(val_tables, ignore_index=True)
    results["selected_models"] = selected

    # Baseline reproduction: persistence benchmark (the original 20-min holdout
    # approach) is retained in the comparison table above.
    results["model_comparison"] = backtest_forecasts(
        splits, models, feature_cols, TARGET_COLS, C.RED_THRESHOLD
    )

    # Walk-forward check on train+validation for the planning-horizon minimum.
    wf_frame = pd.concat([tr, va], ignore_index=True)
    best_min6 = selected["y_min_6"]
    if best_min6 == "LightGBM":
        factory = lambda: make_lgbm(models["y_min_6"]["LightGBM"].params)  # noqa: E731
    elif best_min6 == "Ridge":
        factory = lambda: make_ridge(models["y_min_6"]["Ridge"].params["alpha"])  # noqa: E731
    else:
        factory = PersistenceModel  # noqa: E731
    results["walk_forward"] = walk_forward_backtest(
        wf_frame, feature_cols, "y_min_6", factory
    )

    # ------------------------------------------------------------------
    # Predictions on each split
    # ------------------------------------------------------------------
    cart_floor, cart_ceiling = 0.0, float(tr["cart_current"].max()) * 1.1

    def _pred(target, frame):
        raw = models[target][selected[target]].estimator.predict(frame[feature_cols])
        return np.clip(raw, cart_floor, cart_ceiling)

    for name, frame, X in (("validation", va, Xva), ("holdout", ho, Xho)):
        for target in TARGET_COLS:
            frame[f"pred_{target}"] = _pred(target, frame)

    # ------------------------------------------------------------------
    # Step 4.3 — conformal uncertainty (calibrated on validation only)
    # ------------------------------------------------------------------
    LOGGER.info("STEP 4.3 — conformal calibration of the 40-min minimum forecast")
    conformal = ConformalCalibrator(alpha=C.CONFORMAL_ALPHA).fit(
        va["y_min_6"].to_numpy(dtype=float), va["pred_y_min_6"].to_numpy(dtype=float)
    )
    conformal_3 = ConformalCalibrator(alpha=C.CONFORMAL_ALPHA).fit(
        va["y_min_3"].to_numpy(dtype=float), va["pred_y_min_3"].to_numpy(dtype=float)
    )
    for frame in (va, ho):
        frame["lower_y_min_6"] = conformal.lower_bound(frame["pred_y_min_6"].to_numpy())
        frame["upper_y_min_6"] = conformal.upper_bound(frame["pred_y_min_6"].to_numpy())
        frame["lower_y_min_3"] = conformal_3.lower_bound(frame["pred_y_min_3"].to_numpy())

    results["coverage_holdout"] = conformal.coverage(
        ho["y_min_6"].to_numpy(dtype=float), ho["pred_y_min_6"].to_numpy(dtype=float)
    )
    results["coverage_validation"] = conformal.coverage(
        va["y_min_6"].to_numpy(dtype=float), va["pred_y_min_6"].to_numpy(dtype=float)
    )
    results["residual_normality_note"] = {
        "residual_skew": float(pd.Series(conformal.residuals_).skew()),
        "residual_kurtosis": float(pd.Series(conformal.residuals_).kurtosis()),
        "n_calibration": int(len(conformal.residuals_)),
    }

    # ------------------------------------------------------------------
    # Step 5 — shortage risk
    # ------------------------------------------------------------------
    LOGGER.info("STEP 5 — shortage risk estimation")
    fdr = ForecastDerivedRisk(conformal, C.RED_THRESHOLD)
    va["risk_forecast"] = fdr.predict_proba(va["pred_y_min_6"].to_numpy())
    ho["risk_forecast"] = fdr.predict_proba(ho["pred_y_min_6"].to_numpy())

    y_risk_tr = tr["shortage_next_6"].to_numpy(dtype=int)
    y_risk_va = va["shortage_next_6"].to_numpy(dtype=int)
    y_risk_ho = ho["shortage_next_6"].to_numpy(dtype=int)
    clf_models, clf_table = train_risk_classifier(Xtr, y_risk_tr, Xva, y_risk_va)

    risk_rows = [{"model": "ForecastDerived", **classification_metrics(y_risk_va, va["risk_forecast"].to_numpy())}]
    for name, m in clf_models.items():
        p = m.predict_proba(Xva)[:, 1]
        va[f"risk_{name}"] = p
        risk_rows.append({"model": name, **classification_metrics(y_risk_va, p)})
    risk_table = pd.DataFrame(risk_rows)
    results["risk_model_table"] = risk_table

    best_risk_name = risk_table.sort_values("PR_AUC", ascending=False).iloc[0]["model"]
    LOGGER.info("  selected risk model: %s", best_risk_name)
    results["selected_risk_model"] = best_risk_name
    if best_risk_name == "ForecastDerived":
        risk_va = va["risk_forecast"].to_numpy()
        risk_ho = ho["risk_forecast"].to_numpy()
        risk_estimator = None
    else:
        risk_estimator = clf_models[best_risk_name]
        risk_va = risk_estimator.predict_proba(Xva)[:, 1]
        risk_ho = risk_estimator.predict_proba(Xho)[:, 1]
    va["risk"] = risk_va
    ho["risk"] = risk_ho

    results["risk_holdout_metrics"] = classification_metrics(y_risk_ho, risk_ho)
    results["risk_calibration_holdout"] = calibration_table(y_risk_ho, risk_ho)
    results["risk_threshold_table"] = pd.DataFrame(
        [metrics_at_threshold(y_risk_va, risk_va, t) for t in np.arange(0.05, 0.95, 0.05)]
    )

    # ------------------------------------------------------------------
    # Step 6/7 — alert calibration and replenishment policy search (validation)
    # ------------------------------------------------------------------
    LOGGER.info("STEP 6/7 — alert calibration and policy search on validation")
    alert_cfg, alert_table = search_alert_thresholds(va, risk_va, C.RED_THRESHOLD)
    results["alert_calibration_table"] = alert_table
    results["alert_cfg"] = alert_cfg

    lo_off = conformal.lower_offset
    va = _build_predicted_path(
        va, va["pred_y_cart_3"].to_numpy(), va["pred_y_cart_6"].to_numpy(), lo_off
    )
    ho = _build_predicted_path(
        ho, ho["pred_y_cart_3"].to_numpy(), ho["pred_y_cart_6"].to_numpy(), lo_off
    )

    base_cfg = C.RunConfig(risk_threshold=alert_cfg.warning_risk, cooldown_min=alert_cfg.cooldown_min)
    best_cfg, policy_table = search_policy(
        va,
        va["pred_y_min_6"].to_numpy(),
        va["lower_y_min_6"].to_numpy(),
        risk_va,
        base_cfg,
        LAMBDAS,
        path_cols=PATH_LOWER_COLS,
    )
    results["policy_search_table"] = policy_table
    results["frozen_policy"] = {
        k: v for k, v in best_cfg.__dict__.items() if k != "windows"
    }
    LOGGER.info("  frozen policy: %s", results["frozen_policy"])

    # ------------------------------------------------------------------
    # Steps 7-8 — holdout backtest with the frozen policy
    # ------------------------------------------------------------------
    LOGGER.info("STEP 7/8 — holdout alert and replenishment backtest")
    alert_cfg_final = alert_cfg
    alerts_ho = run_alert_engine(ho, risk_ho, alert_cfg_final)
    events_ho = shortage_events(ho, red=C.RED_THRESHOLD, recovery_records=C.EVENT_RECOVERY_RECORDS)
    warn_summary, matched = evaluate_warnings(alerts_ho, events_ho, ho)
    results["warning_summary"] = warn_summary
    results["matched_events"] = matched
    results["events_holdout"] = events_ho
    results["events_all"] = shortage_events(
        feat, red=C.RED_THRESHOLD, recovery_records=C.EVENT_RECOVERY_RECORDS
    )

    sim = simulate_policy(
        ho,
        ho["pred_y_min_6"].to_numpy(),
        ho["lower_y_min_6"].to_numpy(),
        risk_ho,
        best_cfg,
        LAMBDAS,
        path_cols=PATH_LOWER_COLS,
    )
    results["policy_metrics"] = sim["metrics"]
    results["replenishment_decisions"] = sim["decisions"]
    results["policy_series"] = sim["series"]

    # Robustness and cost sensitivity are computed on validation only.
    LOGGER.info("STEP 8 — robustness and cost sensitivity (validation)")
    results["robustness"] = robustness_analysis(
        va,
        va["pred_y_min_6"].to_numpy(),
        va["lower_y_min_6"].to_numpy(),
        risk_va,
        best_cfg,
        LAMBDAS,
        path_cols=PATH_LOWER_COLS,
    )
    results["cost_sensitivity"] = cost_sensitivity(
        va,
        va["pred_y_min_6"].to_numpy(),
        va["lower_y_min_6"].to_numpy(),
        risk_va,
        best_cfg,
        C.COST_SENSITIVITY_GRID,
        LAMBDAS,
        path_cols=PATH_LOWER_COLS,
    )

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    imp = feature_importance(models["y_min_6"][selected["y_min_6"]].estimator, feature_cols, 15)
    results["feature_importance"] = imp
    results["depletion_analysis"] = _depletion_analysis(ho)
    results["surge_analysis"] = _surge_analysis(ho)

    # Full pipeline state, exposed so downstream tooling (e.g. the simulation
    # audit) can reuse the fitted artefacts without retraining or overwriting
    # the preserved benchmark outputs.
    results["state"] = {
        "raw": raw,
        "feature_frame": feat,
        "train": tr,
        "validation": va,
        "holdout": ho,
        "models": models,
        "selected": selected,
        "feature_cols": feature_cols,
        "clip_bounds": clip_bounds,
        "conformal": conformal,
        "conformal_min3": conformal_3,
        "risk_estimator": risk_estimator,
        "risk_validation": risk_va,
        "risk_holdout": risk_ho,
        "alert_cfg": alert_cfg,
        "policy_cfg": best_cfg,
        "lambdas": LAMBDAS,
        "path_cols": PATH_COLS,
        "path_lower_cols": PATH_LOWER_COLS,
        "cart_bounds": (cart_floor, cart_ceiling),
        "alerts_holdout": alerts_ho,
        "events_holdout": events_ho,
        "simulation": sim,
    }

    # ------------------------------------------------------------------
    # Step 9 — outputs
    # ------------------------------------------------------------------
    if write_outputs:
        LOGGER.info("STEP 9 — writing CSV outputs, figures and report")
        _write_outputs(results, ho, alerts_ho, feat)
        results["figures"] = _make_figures(
            results, raw, ho, alerts_ho, events_ho, risk_ho, sim, imp
        )
        _persist_models(
            models, selected, conformal, risk_estimator, feature_cols,
            best_cfg, alert_cfg, clip_bounds,
        )
    if generate_report:
        build_report(results)
        LOGGER.info("Pipeline complete. Report: %s", C.REPORT_FILE)
    return results


def _depletion_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["bucket"] = pd.cut(
        df["cart_depletion_rate_6"],
        bins=[-np.inf, -0.5, -0.2, -0.05, 0.05, np.inf],
        labels=["steep decline", "moderate decline", "mild decline", "flat", "rising"],
    )
    grp = df.groupby("bucket", observed=True)
    out = pd.DataFrame(
        {
            "n": grp.size(),
            "mean_cart": grp["cart_current"].mean(),
            "observed_shortage_rate_40min": grp["shortage_next_6"].mean(),
            "mean_predicted_min": grp["pred_y_min_6"].mean(),
        }
    ).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    return out


def _surge_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["bucket"] = pd.cut(
        df["traffic_surge_ratio"],
        bins=[-np.inf, 0.75, 1.25, 2.0, np.inf],
        labels=["falling traffic", "stable", "surge", "strong surge"],
    )
    grp = df.groupby("bucket", observed=True)
    out = pd.DataFrame(
        {
            "n": grp.size(),
            "mean_traffic_in_rate": grp["traffic_in_rate"].mean(),
            "mean_cart": grp["cart_current"].mean(),
            "observed_shortage_rate_40min": grp["shortage_next_6"].mean(),
        }
    ).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    return out


def _write_outputs(results: dict, ho: pd.DataFrame, alerts: pd.DataFrame, feat: pd.DataFrame):
    o = C.OUTPUT_DIR

    fc_cols = ["DVRDate", "decision_time", "split", "cart_current"]
    fc = ho[fc_cols + [f"pred_{t}" for t in TARGET_COLS] + TARGET_COLS +
            ["lower_y_min_6", "upper_y_min_6", "horizon_minutes_3", "horizon_minutes_6"]].copy()
    fc.to_csv(o / "forecasts.csv", index=False)

    sp = ho[["DVRDate", "decision_time", "cart_current", "pred_y_min_6", "lower_y_min_6",
             "risk"] + SHORTAGE_COLS].copy()
    sp["alert_state"] = alerts["state"].to_numpy()
    sp["new_predictive_alert"] = alerts["new_predictive_alert"].to_numpy()
    sp["reactive_critical"] = alerts["reactive_critical"].to_numpy()
    sp.to_csv(o / "shortage_predictions.csv", index=False)

    results["events_all"].to_csv(o / "shortage_events.csv", index=False)
    results["replenishment_decisions"].to_csv(o / "replenishment_recommendations.csv", index=False)
    results["model_comparison"].to_csv(o / "model_comparison.csv", index=False)

    pb = pd.DataFrame([results["policy_metrics"]])
    pb.to_csv(o / "policy_backtest.csv", index=False)

    results["policy_search_table"].to_csv(o / "policy_search_validation.csv", index=False)
    results["robustness"].to_csv(o / "policy_robustness.csv", index=False)
    results["feature_report"].to_csv(o / "feature_selection_report.csv", index=False)
    results["matched_events"].to_csv(o / "warning_event_matching.csv", index=False)
    results["policy_series"].to_csv(o / "policy_simulated_series.csv", index=False)
    results["risk_model_table"].to_csv(o / "risk_model_comparison.csv", index=False)

    with open(o / "run_summary.json", "w") as fh:
        json.dump(
            {
                "overview": {k: str(v) for k, v in results["overview"].items()},
                "split_meta": results["split_meta"],
                "selected_models": results["selected_models"],
                "selected_risk_model": results["selected_risk_model"],
                "frozen_policy": {k: str(v) for k, v in results["frozen_policy"].items()},
                "warning_summary": {k: str(v) for k, v in results["warning_summary"].items()},
                "policy_metrics": {k: str(v) for k, v in results["policy_metrics"].items()},
            },
            fh,
            indent=2,
        )


def _make_figures(results, raw, ho, alerts, events, risk, sim, imp) -> dict:
    figs = {}
    figs["historical"] = plots.historical_cart_traffic(raw)
    figs["forecast_20m"] = plots.actual_vs_predicted(
        ho, "y_cart_3", "pred_y_cart_3",
        "Cart count at +3 records (~20 min) — holdout", "forecast_20m.png",
    )
    figs["forecast_40m"] = plots.actual_vs_predicted(
        ho, "y_cart_6", "pred_y_cart_6",
        "Cart count at +6 records (~40 min) — holdout", "forecast_40m.png",
    )
    figs["minimum"] = plots.minimum_forecast_figure(ho, "pred_y_min_6", "lower_y_min_6")
    figs["warning_timeline"] = plots.shortage_warning_timeline(ho, alerts, events, risk)
    figs["replenishment"] = plots.replenishment_timeline(sim["series"], sim["decisions"])
    figs["policy_comparison"] = plots.policy_comparison(sim["metrics"])
    figs["feature_importance"] = plots.feature_importance_figure(imp)
    figs["lead_time"] = plots.lead_time_distribution(results["matched_events"])
    figs["risk_calibration"] = plots.risk_calibration_figure(results["risk_calibration_holdout"])
    return figs


def _persist_models(models, selected, conformal, risk_estimator, feature_cols,
                    policy_cfg, alert_cfg, clip_bounds=None):
    payload = {
        "estimators": {t: models[t][selected[t]].estimator for t in selected},
        "selected": selected,
        "params": {t: models[t][selected[t]].params for t in selected},
        "conformal_residuals": conformal.residuals_,
        "conformal_alpha": conformal.alpha,
        "risk_estimator": risk_estimator,
        "feature_cols": feature_cols,
        "clip_bounds": clip_bounds,
        "policy_cfg": policy_cfg.__dict__,
        "alert_cfg": alert_cfg.__dict__,
        "random_seed": C.RANDOM_SEED,
    }
    with open(C.MODEL_DIR / "cart_forecasting_models.pkl", "wb") as fh:
        pickle.dump(payload, fh)
    LOGGER.info("Models persisted to %s", C.MODEL_DIR / "cart_forecasting_models.pkl")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Pipeline failed: %s", exc)
        raise
