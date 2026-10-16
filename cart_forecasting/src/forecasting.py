"""Forecasting models, chronological selection and conformal uncertainty."""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CONFORMAL_ALPHA,
    LGBM_FIXED_PARAMS,
    LGBM_PARAM_GRID,
    RANDOM_SEED,
    RED_THRESHOLD,
    RIDGE_ALPHAS,
)

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False
    LOGGER.warning("LightGBM unavailable; gradient boosting models will be skipped")


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------
class PersistenceModel:
    """Predict the future cart count with the current cart count."""

    name = "Persistence"

    def __init__(self, base_col: str = "cart_current"):
        self.base_col = base_col
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y) -> "PersistenceModel":
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.base_col].to_numpy(dtype=float)


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=alpha, random_state=None))]
    )


def make_lgbm(params: dict):
    return LGBMRegressor(random_state=RANDOM_SEED, **LGBM_FIXED_PARAMS, **params)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, near_mask: np.ndarray | None = None
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    res = {
        "n": int(len(y_true)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Bias": float(np.mean(err)),
        "MedAE": float(np.median(np.abs(err))),
    }
    if near_mask is not None and near_mask.sum() > 0:
        res["MAE_near_shortage"] = float(np.mean(np.abs(err[near_mask])))
        res["Bias_near_shortage"] = float(np.mean(err[near_mask]))
        res["n_near_shortage"] = int(near_mask.sum())
    else:
        res["MAE_near_shortage"] = np.nan
        res["Bias_near_shortage"] = np.nan
        res["n_near_shortage"] = 0
    return res


def shortage_classification_from_forecast(
    y_true: np.ndarray, y_pred: np.ndarray, red: int = RED_THRESHOLD
) -> dict:
    actual = y_true < red
    pred = y_pred < red
    tp = int(np.sum(actual & pred))
    fp = int(np.sum(~actual & pred))
    fn = int(np.sum(actual & ~pred))
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) > 0 else np.nan
    return {
        "shortage_precision": prec,
        "shortage_recall": rec,
        "shortage_f1": f1,
        "n_actual_shortage": int(actual.sum()),
    }


# --------------------------------------------------------------------------
# Training / selection
# --------------------------------------------------------------------------
@dataclass
class FittedModel:
    name: str
    target: str
    estimator: object
    params: dict
    val_metrics: dict


def _sample_lgbm_grid(max_configs: int = 12) -> list[dict]:
    keys = list(LGBM_PARAM_GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*LGBM_PARAM_GRID.values())]
    rng = np.random.default_rng(RANDOM_SEED)
    if len(combos) > max_configs:
        idx = rng.choice(len(combos), size=max_configs, replace=False)
        combos = [combos[i] for i in sorted(idx)]
    return combos


def train_target_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    target: str,
    near_mask_val: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Fit Persistence / Ridge / LightGBM for one target; score on validation."""
    results: dict[str, FittedModel] = {}
    rows = []

    pers = PersistenceModel().fit(X_train, y_train)
    m = regression_metrics(y_val, pers.predict(X_val), near_mask_val)
    m.update(shortage_classification_from_forecast(y_val, pers.predict(X_val)))
    results["Persistence"] = FittedModel("Persistence", target, pers, {}, m)
    rows.append({"model": "Persistence", "target": target, "params": "{}", **m})

    best_ridge, best_ridge_mae, best_alpha = None, np.inf, None
    for a in RIDGE_ALPHAS:
        est = make_ridge(a).fit(X_train, y_train)
        mae = float(np.mean(np.abs(est.predict(X_val) - y_val)))
        if mae < best_ridge_mae:
            best_ridge, best_ridge_mae, best_alpha = est, mae, a
    pv = best_ridge.predict(X_val)
    m = regression_metrics(y_val, pv, near_mask_val)
    m.update(shortage_classification_from_forecast(y_val, pv))
    results["Ridge"] = FittedModel("Ridge", target, best_ridge, {"alpha": best_alpha}, m)
    rows.append({"model": "Ridge", "target": target, "params": str({"alpha": best_alpha}), **m})

    if HAS_LGBM:
        best_lgb, best_lgb_mae, best_params = None, np.inf, None
        for params in _sample_lgbm_grid():
            est = make_lgbm(params).fit(X_train, y_train)
            mae = float(np.mean(np.abs(est.predict(X_val) - y_val)))
            if mae < best_lgb_mae:
                best_lgb, best_lgb_mae, best_params = est, mae, params
        pv = best_lgb.predict(X_val)
        m = regression_metrics(y_val, pv, near_mask_val)
        m.update(shortage_classification_from_forecast(y_val, pv))
        results["LightGBM"] = FittedModel("LightGBM", target, best_lgb, best_params, m)
        rows.append({"model": "LightGBM", "target": target, "params": str(best_params), **m})

    return results, pd.DataFrame(rows)


def select_model(results: dict, target: str) -> str:
    """Operational selection rule.

    Overall MAE alone is not sufficient: for minimum-count targets the model
    must also be useful near the shortage boundary. We rank by a composite of
    overall MAE, near-shortage MAE and (for minimum targets) shortage recall.
    """
    rows = []
    for name, fm in results.items():
        m = fm.val_metrics
        near = m.get("MAE_near_shortage")
        near = m["MAE"] if near is None or np.isnan(near) else near
        recall = m.get("shortage_recall")
        recall = 0.0 if recall is None or np.isnan(recall) else recall
        if target.startswith("y_min"):
            score = 0.4 * m["MAE"] + 0.4 * near + 0.2 * (1.0 - recall) * m["MAE"]
        else:
            score = 0.6 * m["MAE"] + 0.4 * near
        rows.append((name, score))
    rows.sort(key=lambda r: r[1])
    LOGGER.info("Selection scores for %s: %s", target, rows)
    return rows[0][0]


# --------------------------------------------------------------------------
# Conformal prediction intervals
# --------------------------------------------------------------------------
@dataclass
class ConformalCalibrator:
    """Split-conformal one-sided lower bound plus a residual CDF risk model."""

    alpha: float = CONFORMAL_ALPHA
    residuals_: np.ndarray | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalCalibrator":
        self.residuals_ = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        return self

    @property
    def lower_offset(self) -> float:
        """Quantile ``alpha`` of signed residuals (negative => bound below)."""
        if self.residuals_ is None or not len(self.residuals_):
            return 0.0
        return float(np.quantile(self.residuals_, self.alpha))

    @property
    def upper_offset(self) -> float:
        if self.residuals_ is None or not len(self.residuals_):
            return 0.0
        return float(np.quantile(self.residuals_, 1 - self.alpha))

    def lower_bound(self, y_pred: np.ndarray) -> np.ndarray:
        return np.asarray(y_pred, dtype=float) + self.lower_offset

    def upper_bound(self, y_pred: np.ndarray) -> np.ndarray:
        return np.asarray(y_pred, dtype=float) + self.upper_offset

    def exceedance_probability(self, y_pred: np.ndarray, threshold: float) -> np.ndarray:
        """Empirical P(Y < threshold) = F_resid(threshold - yhat).

        Uses the empirical residual distribution from the calibration period
        instead of a normality assumption.
        """
        res = np.sort(self.residuals_)
        need = np.asarray(threshold - np.asarray(y_pred, dtype=float), dtype=float)
        idx = np.searchsorted(res, need, side="right")
        return idx / len(res)

    def coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        lo = self.lower_bound(y_pred)
        hi = self.upper_bound(y_pred)
        y_true = np.asarray(y_true, dtype=float)
        return {
            "target_lower_coverage": 1 - self.alpha,
            "empirical_lower_coverage": float(np.mean(y_true >= lo)),
            "target_interval_coverage": 1 - 2 * self.alpha,
            "empirical_interval_coverage": float(np.mean((y_true >= lo) & (y_true <= hi))),
            "mean_interval_width": float(np.mean(hi - lo)),
            "lower_offset": self.lower_offset,
            "upper_offset": self.upper_offset,
        }


def feature_importance(model, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    est = model
    if isinstance(est, Pipeline):
        est = est.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        kind = "gain-split importance"
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
        kind = "|standardised coefficient|"
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])
    total = imp.sum()
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": imp,
            "importance_pct": imp / total * 100 if total > 0 else imp,
            "importance_type": kind,
        }
    )
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
