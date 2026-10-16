"""Shortage risk estimation: forecast-derived and direct classifiers."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
                ),
            ),
        ]
    )


def make_lgbm_classifier() -> "LGBMClassifier":
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=-1,
    )


def classification_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    pos = int(y_true.sum())
    out = {
        "n": int(len(y_true)),
        "n_positive": pos,
        "base_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "PR_AUC": float(average_precision_score(y_true, p)) if 0 < pos < len(y_true) else np.nan,
        "ROC_AUC": float(roc_auc_score(y_true, p)) if 0 < pos < len(y_true) else np.nan,
        "Brier": float(brier_score_loss(y_true, p)) if len(y_true) else np.nan,
    }
    return out


def metrics_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    pred = (np.asarray(p, dtype=float) >= thr).astype(int)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec > 0) else np.nan
    return {
        "threshold": thr,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "alert_rate": float(pred.mean()) if len(pred) else np.nan,
    }


def best_threshold_by_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if y_true.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = np.where((prec[:-1] + rec[:-1]) > 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]), 0)
    if not len(f1):
        return 0.5
    return float(thr[int(np.argmax(f1))])


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(p, dtype=float)})
    edges = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["p"], bins=edges, include_lowest=True)
    grp = df.groupby("bin", observed=True)
    out = pd.DataFrame(
        {
            "n": grp.size(),
            "mean_predicted": grp["p"].mean(),
            "observed_rate": grp["y"].mean(),
        }
    ).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def train_risk_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """Fit candidate direct classifiers and score them chronologically."""
    models: dict[str, object] = {}
    rows = []

    if y_train.sum() < 10 or y_train.sum() == len(y_train):
        LOGGER.warning("Too few positive shortage labels for a direct classifier")
        return models, pd.DataFrame(rows)

    lr = make_logistic().fit(X_train, y_train)
    p = lr.predict_proba(X_val)[:, 1]
    models["LogisticRegression"] = lr
    rows.append({"model": "LogisticRegression", **classification_metrics(y_val, p)})

    if HAS_LGBM:
        base = make_lgbm_classifier()
        try:
            clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
            clf.fit(X_train, y_train)
        except Exception:  # pragma: no cover
            clf = base.fit(X_train, y_train)
        p = clf.predict_proba(X_val)[:, 1]
        models["LightGBMClassifier"] = clf
        rows.append({"model": "LightGBMClassifier", **classification_metrics(y_val, p)})

    return models, pd.DataFrame(rows)


class ForecastDerivedRisk:
    """P(min future cart < RED) from the conformal residual distribution."""

    name = "ForecastDerived"

    def __init__(self, calibrator, threshold: float):
        self.calibrator = calibrator
        self.threshold = threshold

    def predict_proba(self, y_pred_min: np.ndarray) -> np.ndarray:
        return self.calibrator.exceedance_probability(y_pred_min, self.threshold)


def blend_risk(p_forecast: np.ndarray, p_classifier: np.ndarray, w: float = 0.5) -> np.ndarray:
    return np.clip(w * np.asarray(p_forecast) + (1 - w) * np.asarray(p_classifier), 0, 1)
