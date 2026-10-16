"""Automatic calibration of shortage risk for a trained candidate.

Runs on every successful training job with no manual API call. Residuals come
from the CALIBRATION partition only — a window the model never trained on and
which is disjoint from the backtest holdout.

Scopes, in order of preference:

    entity           enough samples and both outcome classes present
    global_fallback  an explicitly validated global calibrator exists
    insufficient_data no probabilities are invented; promotion is blocked for
                     calibrated risk scoring
"""
from __future__ import annotations

import functools
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from settings import (
    CONFORMAL_ALPHA,
    MIN_CALIBRATION_POSITIVES,
    MIN_CALIBRATION_SAMPLES,
)

from forecasting import ConformalCalibrator  # noqa: E402
from registry.database import as_json, execute, query_one, utc_now  # noqa: E402

LOGGER = logging.getLogger("platform.calibration")

FITTED, INSUFFICIENT, FAILED = "fitted", "insufficient_data", "failed"
SCOPE_ENTITY, SCOPE_GLOBAL, SCOPE_NONE = "entity", "global_fallback", "none"


def _reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["p"], bins=edges, include_lowest=True)
    g = df.groupby("bin", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "mean_predicted": g["p"].mean(),
        "observed_rate": g["y"].mean(),
    }).reset_index()
    out["bin"] = out["bin"].astype(str)
    out["gap"] = (out["mean_predicted"] - out["observed_rate"]).abs()
    return out


def _ece(table: pd.DataFrame) -> float:
    total = table["n"].sum()
    if total == 0:
        return float("nan")
    return float((table["n"] / total * table["gap"]).sum())


def fit_calibrator(
    y_true_min: np.ndarray,
    y_pred_min: np.ndarray,
    red_threshold: float,
    alpha: float = CONFORMAL_ALPHA,
) -> tuple[ConformalCalibrator, dict]:
    """Split-conformal calibration on out-of-training residuals."""
    y_true_min = np.asarray(y_true_min, dtype=float)
    y_pred_min = np.asarray(y_pred_min, dtype=float)
    calib = ConformalCalibrator(alpha=alpha).fit(y_true_min, y_pred_min)

    actual_event = (y_true_min < red_threshold).astype(int)
    # In-sample reliability of the fitted calibrator (reported, not a guarantee).
    p_cal = calib.exceedance_probability(y_pred_min, red_threshold)
    # Uncalibrated reference: hard 0/1 from the point forecast.
    p_raw = (y_pred_min < red_threshold).astype(float)

    table = _reliability_table(actual_event, p_cal)
    diagnostics = {
        "n_samples": int(len(y_true_min)),
        "n_positive": int(actual_event.sum()),
        "base_rate": float(actual_event.mean()) if len(actual_event) else float("nan"),
        "alpha": float(alpha),
        "lower_offset": float(calib.lower_offset),
        "upper_offset": float(calib.upper_offset),
        "brier": float(np.mean((p_cal - actual_event) ** 2)),
        "brier_uncalibrated": float(np.mean((p_raw - actual_event) ** 2)),
        "ece": _ece(table),
        "coverage": float(np.mean(y_true_min >= calib.lower_bound(y_pred_min))),
        "residual_skew": float(pd.Series(calib.residuals_).skew()),
        "residual_kurtosis": float(pd.Series(calib.residuals_).kurtosis()),
    }
    diagnostics["brier_improvement"] = (
        diagnostics["brier_uncalibrated"] - diagnostics["brier"]
    )
    return calib, {"diagnostics": diagnostics, "reliability": table}


def auto_calibrate(
    entity_id: str,
    model_id: str,
    job_id: str,
    calibration_frame: pd.DataFrame,
    pred_col: str,
    target_col: str,
    red_threshold: float,
    artifact_dir: Path,
    alpha: float = CONFORMAL_ALPHA,
    allow_global_fallback: bool = True,
    config_hash: str | None = None,
) -> dict:
    """Fit, diagnose and persist a calibrator. Never fabricates probabilities."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cal_id = f"cal_{entity_id.replace(':', '_')}_{uuid.uuid4().hex[:8]}"
    warnings: list[str] = []
    # The RED this calibrator is fitted against is recorded so a later
    # operational change cannot silently reinterpret these probabilities.
    _save = functools.partial(_persist, calibration_red=float(red_threshold),
                              config_hash=config_hash)

    usable = calibration_frame.dropna(subset=[pred_col, target_col])
    n = len(usable)
    y = usable[target_col].to_numpy(dtype=float)
    n_pos = int((y < red_threshold).sum())
    n_neg = int(n - n_pos)

    if n < MIN_CALIBRATION_SAMPLES or n_pos < MIN_CALIBRATION_POSITIVES or n_neg == 0:
        reason = (
            f"calibration window has {n} samples with {n_pos} shortage event(s) and "
            f"{n_neg} non-event(s); need >= {MIN_CALIBRATION_SAMPLES} samples, "
            f">= {MIN_CALIBRATION_POSITIVES} events and both classes present"
        )
        scope, status = SCOPE_NONE, INSUFFICIENT
        fallback = _global_fallback() if allow_global_fallback else None
        if fallback is not None:
            scope, status = SCOPE_GLOBAL, FITTED
            warnings.append(
                "per-entity calibration not possible; using the validated global "
                "fallback calibrator. Shortage probabilities are NOT calibrated to "
                "this entity."
            )
            warnings.append(reason)
            record = _save(
                cal_id, job_id, entity_id, model_id, status, scope, "conformal_global",
                {"n_samples": n, "n_positive": n_pos,
                 "lower_offset": float(fallback["lower_offset"] or 0.0),
                 "alpha": float(fallback["alpha"] or alpha)},
                None, fallback["artifact_path"], None, usable, warnings,
            )
            record["fallback_calibration_id"] = fallback["calibration_id"]
            return record
        warnings.append(reason)
        LOGGER.warning("%s: %s", entity_id, reason)
        return _save(cal_id, job_id, entity_id, model_id, INSUFFICIENT, SCOPE_NONE,
                        None, {"n_samples": n, "n_positive": n_pos}, None, None, None,
                        usable, warnings)

    try:
        calib, detail = fit_calibrator(
            usable[target_col].to_numpy(dtype=float),
            usable[pred_col].to_numpy(dtype=float),
            red_threshold, alpha,
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Calibration failed for %s", entity_id)
        return _save(cal_id, job_id, entity_id, model_id, FAILED, SCOPE_NONE, None,
                        {"error": str(exc)}, None, None, None, usable,
                        warnings + [f"calibration failed: {exc}"])

    art = artifact_dir / "calibrator.npy"
    np.save(art, calib.residuals_)
    rel_path = artifact_dir / "reliability.csv"
    detail["reliability"].to_csv(rel_path, index=False)

    d = detail["diagnostics"]
    if d["brier_improvement"] < 0:
        warnings.append(
            f"calibrated Brier ({d['brier']:.4f}) is worse than the uncalibrated "
            f"reference ({d['brier_uncalibrated']:.4f}); reported, not hidden"
        )
    LOGGER.info("%s calibrated: n=%d events=%d Brier=%.4f ECE=%.4f",
                entity_id, d["n_samples"], d["n_positive"], d["brier"], d["ece"])
    return _save(cal_id, job_id, entity_id, model_id, FITTED, SCOPE_ENTITY,
                    "split_conformal", d, str(art), str(rel_path), None, usable,
                    warnings)


def _global_fallback() -> dict | None:
    return query_one(
        "SELECT * FROM calibrator_registry WHERE scope='global' AND status='active' "
        "ORDER BY fitted_at DESC"
    )


def _persist(cal_id, job_id, entity_id, model_id, status, scope, method, metrics,
             artifact_path, reliability_path, _unused, frame, warnings,
             calibration_red=None, config_hash=None) -> dict:
    m = metrics or {}
    window_start = str(frame["decision_time"].min()) if len(frame) else None
    window_end = str(frame["decision_time"].max()) if len(frame) else None
    execute(
        """INSERT OR REPLACE INTO calibration_records
           (calibration_id, job_id, entity_id, model_id, status, scope, method,
            n_samples, n_positive, alpha, lower_offset, brier, brier_uncalibrated,
            ece, coverage, window_start, window_end, artifact_path,
            reliability_path, metrics_json, warnings_json, created_at,
            calibration_red, config_hash)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            cal_id, job_id, entity_id, model_id, status, scope, method,
            m.get("n_samples"), m.get("n_positive"), m.get("alpha"),
            m.get("lower_offset"), m.get("brier"), m.get("brier_uncalibrated"),
            m.get("ece"), m.get("coverage"), window_start, window_end,
            artifact_path, reliability_path, as_json(m), as_json(warnings), utc_now(),
            calibration_red, config_hash,
        ),
    )
    return {
        "calibration_id": cal_id, "status": status, "scope": scope, "method": method,
        "metrics": m, "artifact_path": artifact_path,
        "reliability_path": reliability_path, "warnings": warnings,
        "window_start": window_start, "window_end": window_end,
        "calibration_red": calibration_red, "config_hash": config_hash,
    }


def get_calibration(calibration_id: str) -> dict | None:
    return query_one("SELECT * FROM calibration_records WHERE calibration_id=?",
                     (calibration_id,))


def calibration_for_job(job_id: str, entity_id: str | None = None) -> dict | None:
    if entity_id:
        return query_one(
            "SELECT * FROM calibration_records WHERE job_id=? AND entity_id=?",
            (job_id, entity_id),
        )
    return query_one("SELECT * FROM calibration_records WHERE job_id=?", (job_id,))


def load_calibrator_from_record(record: dict) -> ConformalCalibrator | None:
    """Rebuild a usable calibrator object from a persisted record."""
    if record is None or record.get("status") != FITTED:
        return None
    path = record.get("artifact_path")
    if not path or not Path(path).exists():
        return None
    calib = ConformalCalibrator(alpha=float(record.get("alpha") or CONFORMAL_ALPHA))
    calib.residuals_ = np.load(path)
    return calib
