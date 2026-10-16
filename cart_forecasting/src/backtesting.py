"""Chronological backtesting: forecasts, policy search and robustness."""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from alert_engine import AlertEngineConfig, evaluate_warnings, run_alert_engine
from config import POLICY_GRID, ROBUSTNESS_GRID, RunConfig
from forecasting import regression_metrics
from simulator import simulate_policy
from target_builder import shortage_events

LOGGER = logging.getLogger(__name__)


def backtest_forecasts(
    frames: dict,
    models: dict,
    feature_cols: list[str],
    targets: list[str],
    red: int,
) -> pd.DataFrame:
    """Score every model/target/split combination on chronological data."""
    rows = []
    for target in targets:
        for model_name, fm in models[target].items():
            for split_name, fr in frames.items():
                if fr.empty:
                    continue
                X = fr[feature_cols]
                y = fr[target].to_numpy(dtype=float)
                pred = fm.estimator.predict(X)
                near = (y < red + 10) | (fr["cart_current"].to_numpy() < red + 10)
                m = regression_metrics(y, pred, near)
                rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "split": split_name,
                        "params": str(fm.params),
                        **m,
                    }
                )
    return pd.DataFrame(rows)


def walk_forward_backtest(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    model_factory,
    n_folds: int = 4,
    min_train: int = 400,
) -> pd.DataFrame:
    """Expanding-window walk-forward evaluation on the pre-holdout data."""
    rows = []
    n = len(frame)
    if n <= min_train + n_folds:
        return pd.DataFrame(rows)
    fold_size = (n - min_train) // n_folds
    for k in range(n_folds):
        tr_end = min_train + k * fold_size
        te_end = min(tr_end + fold_size, n)
        if te_end - tr_end < 20:
            continue
        tr = frame.iloc[:tr_end]
        te = frame.iloc[tr_end:te_end]
        est = model_factory()
        est.fit(tr[feature_cols], tr[target].to_numpy(dtype=float))
        pred = est.predict(te[feature_cols])
        m = regression_metrics(te[target].to_numpy(dtype=float), pred)
        rows.append(
            {
                "fold": k + 1,
                "target": target,
                "train_end": tr["decision_time"].max(),
                "test_start": te["decision_time"].min(),
                "test_end": te["decision_time"].max(),
                **m,
            }
        )
    return pd.DataFrame(rows)


def search_policy(
    frame: pd.DataFrame,
    predicted_min: np.ndarray,
    lower_bound_min: np.ndarray,
    risk: np.ndarray,
    base_cfg: RunConfig,
    lambdas: dict,
    grid: dict | None = None,
    path_cols: list[str] | None = None,
) -> tuple[RunConfig, pd.DataFrame]:
    """Grid-search the replenishment policy on validation data only."""
    grid = grid or POLICY_GRID
    keys = list(grid)
    rows = []
    best, best_cost = None, np.inf

    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        cfg = base_cfg.replace(**params)
        res = simulate_policy(
            frame, predicted_min, lower_bound_min, risk, cfg, lambdas, path_cols
        )
        m = res["metrics"]
        rows.append({**params, **{k: m[k] for k in sorted(m)}})
        if m["policy_cost_J"] < best_cost:
            best_cost, best = m["policy_cost_J"], cfg

    table = pd.DataFrame(rows).sort_values("policy_cost_J").reset_index(drop=True)
    LOGGER.info("Policy search complete: %d candidates, best J=%.2f", len(table), best_cost)
    return best, table


def search_alert_thresholds(
    frame: pd.DataFrame,
    risk: np.ndarray,
    red: int,
    warning_grid=(0.10, 0.15, 0.20, 0.30, 0.40, 0.50),
    cooldown_grid=(10.0, 20.0, 30.0),
    max_false_per_day: float = 6.0,
) -> tuple[AlertEngineConfig, pd.DataFrame]:
    """Calibrate alert thresholds on validation data.

    Selection maximises event recall subject to a false-alert budget, breaking
    ties on median lead time.
    """
    events = shortage_events(frame, red=red)
    rows = []
    best, best_key = None, None
    for w, cd in itertools.product(warning_grid, cooldown_grid):
        cfg = AlertEngineConfig(
            warning_risk=w, critical_risk=min(0.95, w + 0.25), cooldown_min=cd, red_threshold=red
        )
        alerts = run_alert_engine(frame, risk, cfg)
        summary, _ = evaluate_warnings(alerts, events, frame)
        rows.append({"warning_risk": w, "cooldown_min": cd, **summary})
        feasible = summary["false_alerts_per_day"] <= max_false_per_day
        key = (
            feasible,
            summary["event_recall"] if not np.isnan(summary["event_recall"]) else 0,
            summary["median_lead_time_min"] if not np.isnan(summary["median_lead_time_min"]) else 0,
            -summary["false_alerts_per_day"],
        )
        if best_key is None or key > best_key:
            best_key, best = key, cfg
    table = pd.DataFrame(rows)
    LOGGER.info(
        "Alert calibration: selected warning_risk=%.2f cooldown=%.0f",
        best.warning_risk,
        best.cooldown_min,
    )
    return best, table


def robustness_analysis(
    frame: pd.DataFrame,
    predicted_min: np.ndarray,
    lower_bound_min: np.ndarray,
    risk: np.ndarray,
    cfg: RunConfig,
    lambdas: dict,
    grid: dict | None = None,
    path_cols: list[str] | None = None,
) -> pd.DataFrame:
    """One-factor-at-a-time sensitivity around the frozen policy."""
    grid = grid or ROBUSTNESS_GRID
    rows = []
    for param, values in grid.items():
        for v in values:
            c = cfg.replace(**{param: v})
            res = simulate_policy(
                frame, predicted_min, lower_bound_min, risk, c, lambdas, path_cols
            )
            m = res["metrics"]
            rows.append(
                {
                    "parameter": param,
                    "value": v,
                    "policy_events": m["policy_events"],
                    "baseline_events": m["baseline_events"],
                    "events_prevented": m["events_prevented"],
                    "severity_reduction_pct": m["severity_reduction_pct"],
                    "n_replenishment_actions": m["n_replenishment_actions"],
                    "total_carts_recommended": m["total_carts_recommended"],
                    "policy_cost_J": m["policy_cost_J"],
                }
            )
    return pd.DataFrame(rows)


def cost_sensitivity(
    frame: pd.DataFrame,
    predicted_min: np.ndarray,
    lower_bound_min: np.ndarray,
    risk: np.ndarray,
    base_cfg: RunConfig,
    cost_grid: dict,
    base_lambdas: dict,
    path_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Sensitivity of the selected policy to the unknown cost coefficients."""
    rows = []
    for shortage_cost in cost_grid["COST_SHORTAGE"]:
        for trip_cost in cost_grid["COST_TRIP"]:
            lam = {**base_lambdas, "shortage": shortage_cost, "trip": trip_cost}
            cfg_best, table = search_policy(
                frame,
                predicted_min,
                lower_bound_min,
                risk,
                base_cfg,
                lam,
                grid={
                    "risk_threshold": POLICY_GRID["risk_threshold"],
                    "target_stock": POLICY_GRID["target_stock"],
                },
                path_cols=path_cols,
            )
            top = table.iloc[0]
            rows.append(
                {
                    "lambda_shortage": shortage_cost,
                    "lambda_trip": trip_cost,
                    "best_risk_threshold": cfg_best.risk_threshold,
                    "best_target_stock": cfg_best.target_stock,
                    "events_prevented": top["events_prevented"],
                    "n_replenishment_actions": top["n_replenishment_actions"],
                    "severity_reduction_pct": top["severity_reduction_pct"],
                }
            )
    return pd.DataFrame(rows)
