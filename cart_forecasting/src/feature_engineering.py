"""Unified advanced feature engineering.

Every feature is computed from information available **at or before** the
decision time of record ``t`` (i.e. ``next_DVRDate`` of record ``t``). Rolling
windows never cross a sequence break (``block_id``).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*highly fragmented.*")

from config import (
    AMBER_THRESHOLD,
    CART_COL,
    EPS,
    FEATURE_WINDOWS,
    IN_COL,
    OUT_COL,
    RED_THRESHOLD,
    TIME_COL,
)

LOGGER = logging.getLogger(__name__)

# Columns that encode future information, identifiers or bookkeeping and must
# never enter the feature matrix.
LABEL_COLS = {
    "y_cart_3",
    "y_cart_6",
    "y_min_3",
    "y_min_6",
    "y_max_6",
    "shortage_next_3",
    "shortage_next_6",
    "onset_next_3",
    "onset_next_6",
    "shortage_active",
    "shortage_onset",
    "shortage_recovery",
    "horizon_reliable",
    "purged",
}

LABEL_PREFIXES = ("target_", "y_", "pred_", "lower_", "upper_", "risk", "horizon_", "path_")

NON_FEATURE_COLS = {
    "CountID",
    TIME_COL,
    "next_DVRDate",
    "decision_time",
    "target_end_time",
    "date",
    "block_id",
    "block_seq",
    "entity_id",
    "split",
    "KDVR",
    "ChanNo",
    "CatID",
    "AreaID",
    "CNN_Index",
    "siteKey",
    "sequence_break_after",
    "gap_to_next_min",
    "long_interval",
    "elapsed_from_start_min",
    CART_COL,
    IN_COL,
    OUT_COL,
} | LABEL_COLS


def _safe_div(num, den, clip: float = 1e4):
    out = np.asarray(num, dtype=float) / (np.asarray(den, dtype=float) + EPS)
    return np.clip(np.nan_to_num(out, nan=np.nan, posinf=clip, neginf=-clip), -clip, clip)


def _rolling_slope_r2(
    values: np.ndarray, times: np.ndarray, block: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling OLS slope (units per minute) and R^2 over ``window`` points."""
    n = len(values)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    for i in range(n):
        start = i - window + 1
        if start < 0:
            continue
        if block[start] != block[i]:
            continue
        y = values[start : i + 1].astype(float)
        x = times[start : i + 1].astype(float)
        if np.any(~np.isfinite(y)) or np.any(~np.isfinite(x)):
            continue
        xm, ym = x.mean(), y.mean()
        sxx = float(((x - xm) ** 2).sum())
        if sxx <= EPS:
            continue
        b = float(((x - xm) * (y - ym)).sum()) / sxx
        slope[i] = b
        syy = float(((y - ym) ** 2).sum())
        if syy <= EPS:
            r2[i] = 1.0
        else:
            pred = ym + b * (x - xm)
            r2[i] = 1.0 - float(((y - pred) ** 2).sum()) / syy
    return slope, r2


def _grouped_shift(s: pd.Series, groups: pd.Series, periods: int) -> pd.Series:
    return s.groupby(groups, observed=True).shift(periods)


def _grouped_roll(s: pd.Series, groups: pd.Series, window: int, how: str) -> pd.Series:
    r = s.groupby(groups, observed=True).rolling(window=window, min_periods=window)
    return getattr(r, how)().reset_index(level=0, drop=True)


def build_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    red_threshold: float | None = None,
    amber_threshold: float | None = None,
) -> pd.DataFrame:
    """Return the dataframe augmented with the full feature set.

    ``red_threshold`` / ``amber_threshold`` are **model-dependent** parameters:
    17 of the engineered features encode them (``below_red``, ``below_amber``,
    ``frac_below_amber_*``, ``min_distance_to_red_*``, ``distance_to_*``,
    ``ttb_red_*``, ``ttb_amber_*``). They must therefore be supplied per entity
    at training time and frozen with the model, never taken from a global.

    Both default to the pipeline constants so existing single-entity callers
    behave exactly as before.
    """
    red = float(RED_THRESHOLD if red_threshold is None else red_threshold)
    amber = float(AMBER_THRESHOLD if amber_threshold is None else amber_threshold)
    windows = windows or FEATURE_WINDOWS
    out = df.copy()
    g = out["block_id"]

    # ---------------- 3.1 current state ----------------
    out["cart_current"] = out[CART_COL].astype(float)
    out["traffic_in_current"] = out[IN_COL].astype(float)
    out["traffic_out_current"] = out[OUT_COL].astype(float)
    out["traffic_net_current"] = out["traffic_in_current"] - out["traffic_out_current"]
    dur = out["interval_duration"].clip(lower=0.5)
    out["traffic_in_rate"] = out["traffic_in_current"] / dur
    out["traffic_out_rate"] = out["traffic_out_current"] / dur
    out["traffic_net_rate"] = out["traffic_in_rate"] - out["traffic_out_rate"]

    # ---------------- 3.2 / 3.3 lags ----------------
    for k in (1, 2, 3, 6, 9):
        out[f"cart_lag_{k}"] = _grouped_shift(out["cart_current"], g, k)
    for k in (1, 2, 3, 6):
        out[f"traffic_in_lag_{k}"] = _grouped_shift(out["traffic_in_current"], g, k)
        out[f"traffic_out_lag_{k}"] = _grouped_shift(out["traffic_out_current"], g, k)

    # ---------------- 3.4 rolling statistics ----------------
    for w in windows:
        out[f"cart_mean_{w}"] = _grouped_roll(out["cart_current"], g, w, "mean")
        out[f"cart_median_{w}"] = _grouped_roll(out["cart_current"], g, w, "median")
        out[f"cart_min_{w}"] = _grouped_roll(out["cart_current"], g, w, "min")
        out[f"cart_max_{w}"] = _grouped_roll(out["cart_current"], g, w, "max")
        out[f"cart_std_{w}"] = _grouped_roll(out["cart_current"], g, w, "std")
        out[f"cart_range_{w}"] = out[f"cart_max_{w}"] - out[f"cart_min_{w}"]

        out[f"traffic_in_sum_{w}"] = _grouped_roll(out["traffic_in_current"], g, w, "sum")
        out[f"traffic_out_sum_{w}"] = _grouped_roll(out["traffic_out_current"], g, w, "sum")
        out[f"traffic_net_sum_{w}"] = (
            out[f"traffic_in_sum_{w}"] - out[f"traffic_out_sum_{w}"]
        )
        out[f"traffic_in_rate_mean_{w}"] = _grouped_roll(out["traffic_in_rate"], g, w, "mean")
        out[f"traffic_out_rate_mean_{w}"] = _grouped_roll(out["traffic_out_rate"], g, w, "mean")
        out[f"traffic_in_rate_max_{w}"] = _grouped_roll(out["traffic_in_rate"], g, w, "max")
        out[f"traffic_out_rate_max_{w}"] = _grouped_roll(out["traffic_out_rate"], g, w, "max")
        out[f"traffic_in_volatility_{w}"] = _grouped_roll(out["traffic_in_rate"], g, w, "std")
        out[f"traffic_out_volatility_{w}"] = _grouped_roll(out["traffic_out_rate"], g, w, "std")

        # ---------------- 3.5 ratios ----------------
        out[f"ratio_in_out_{w}"] = _safe_div(
            out[f"traffic_in_sum_{w}"] + EPS, out[f"traffic_out_sum_{w}"] + EPS, clip=50.0
        )
        out[f"ratio_net_{w}"] = _safe_div(
            out[f"traffic_in_sum_{w}"] - out[f"traffic_out_sum_{w}"],
            out[f"traffic_in_sum_{w}"] + out[f"traffic_out_sum_{w}"],
            clip=5.0,
        )
        out[f"traffic_pressure_{w}"] = _safe_div(
            out[f"traffic_in_sum_{w}"], out["cart_current"], clip=50.0
        )

    out["traffic_surge_ratio"] = _safe_div(
        out["traffic_in_rate_mean_3"], out["traffic_in_rate_mean_9"], clip=20.0
    )
    out["traffic_out_surge_ratio"] = _safe_div(
        out["traffic_out_rate_mean_3"], out["traffic_out_rate_mean_9"], clip=20.0
    )

    # ---------------- 3.6 momentum / 3.7 depletion ----------------
    t_min = (out["decision_time"] - out["decision_time"].iloc[0]).dt.total_seconds() / 60.0
    out["_t_min"] = t_min
    for w in windows:
        lag_cart = _grouped_shift(out["cart_current"], g, w)
        lag_time = _grouped_shift(t_min, g, w)
        out[f"cart_momentum_{w}"] = out["cart_current"] - lag_cart
        elapsed = (t_min - lag_time).clip(lower=0.5)
        out[f"cart_depletion_rate_{w}"] = out[f"cart_momentum_{w}"] / elapsed
        out[f"elapsed_min_{w}"] = elapsed

    # ---------------- 3.8 / 3.9 slopes ----------------
    blocks = out["block_id"].to_numpy()
    tt = t_min.to_numpy()
    for w in windows:
        s, r2 = _rolling_slope_r2(out["cart_current"].to_numpy(), tt, blocks, w)
        out[f"cart_slope_{w}"] = s
        out[f"cart_trend_r2_{w}"] = r2
        s_in, _ = _rolling_slope_r2(out["traffic_in_rate"].to_numpy(), tt, blocks, w)
        s_out, _ = _rolling_slope_r2(out["traffic_out_rate"].to_numpy(), tt, blocks, w)
        out[f"traffic_in_slope_{w}"] = s_in
        out[f"traffic_out_slope_{w}"] = s_out

    # ---------------- 3.10 acceleration ----------------
    for name, col in (
        ("cart_acceleration", "cart_slope_3"),
        ("traffic_in_acceleration", "traffic_in_slope_3"),
        ("traffic_out_acceleration", "traffic_out_slope_3"),
    ):
        prev = _grouped_shift(out[col], g, 3)
        prev_t = _grouped_shift(t_min, g, 3)
        out[name] = (out[col] - prev) / (t_min - prev_t).clip(lower=0.5)
    out["cart_depletion_acceleration"] = (
        out["cart_depletion_rate_3"] - out["cart_depletion_rate_6"]
    )

    # ---------------- 3.11 volatility ----------------
    cart_diff = out["cart_current"] - _grouped_shift(out["cart_current"], g, 1)
    out["cart_diff_1"] = cart_diff
    for w in windows:
        out[f"cart_volatility_{w}"] = out[f"cart_std_{w}"]
        out[f"cart_diff_volatility_{w}"] = _grouped_roll(cart_diff, g, w, "std")
        out[f"cart_diff_mean_{w}"] = _grouped_roll(cart_diff, g, w, "mean")

    # ---------------- 3.12 threshold distance ----------------
    out["distance_to_red"] = out["cart_current"] - red
    out["distance_to_amber"] = out["cart_current"] - amber
    out["below_red"] = (out["cart_current"] < red).astype(int)
    out["below_amber"] = (out["cart_current"] < amber).astype(int)
    for w in windows:
        out[f"min_distance_to_red_{w}"] = out[f"cart_min_{w}"] - red
        out[f"frac_below_amber_{w}"] = _grouped_roll(
            out["below_amber"].astype(float), g, w, "mean"
        )

    # ---------------- 3.13 time-to-breach heuristic ----------------
    for w in windows:
        depl = out[f"cart_depletion_rate_{w}"]
        depleting = depl < 0
        for thr, label in ((red, "red"), (amber, "amber")):
            above = out["cart_current"] > thr
            ttb = (out["cart_current"] - thr) / (depl.abs() + EPS)
            ttb = ttb.where(depleting & above, np.nan)
            out[f"ttb_{label}_{w}"] = ttb.clip(upper=480.0).fillna(480.0)
            out[f"ttb_{label}_{w}_finite"] = (depleting & above).astype(int)

    # ---------------- 3.14 temporal ----------------
    ts = out["decision_time"]
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    tod = out["hour"] + out["minute"] / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * tod / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * tod / 24.0)
    out["weekday_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["weekday_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["minutes_since_midnight"] = tod * 60.0

    out = out.drop(columns=["_t_min"])
    out = out.copy()  # defragment after the many column insertions
    # Provenance for auditing: which thresholds shaped this feature matrix.
    out.attrs["red_threshold"] = red
    out.attrs["amber_threshold"] = amber
    LOGGER.info("Engineered %d candidate feature columns", len(candidate_features(out)))
    return out


def candidate_features(frame: pd.DataFrame) -> list[str]:
    cols = []
    for c in frame.columns:
        if c in NON_FEATURE_COLS or c.startswith(LABEL_PREFIXES):
            continue
        if not np.issubdtype(frame[c].dtype, np.number):
            continue
        cols.append(c)
    return cols


def fit_value_clipper(
    frame: pd.DataFrame,
    cols: list[str],
    train_mask: pd.Series,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
    margin: float = 0.25,
) -> pd.DataFrame:
    """Learn robust clipping bounds on training data only.

    Ratio and rate features are heavy-tailed (traffic can be zero at night),
    which makes linear models produce physically impossible forecasts. Bounds
    are the training quantiles widened by ``margin`` of the inter-quantile
    range, so ordinary variation is untouched while extreme tails are capped.
    """
    train = frame.loc[train_mask, cols]
    lo = train.quantile(lower_q)
    hi = train.quantile(upper_q)
    span = (hi - lo).abs()
    return pd.DataFrame({"lower": lo - margin * span, "upper": hi + margin * span})


def apply_value_clipper(frame: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in bounds.index:
        if c in out.columns:
            out[c] = out[c].clip(bounds.loc[c, "lower"], bounds.loc[c, "upper"])
    return out


def select_features(
    frame: pd.DataFrame,
    candidates: list[str],
    train_mask: pd.Series,
    max_missing: float = 0.20,
    corr_threshold: float = 0.995,
) -> tuple[list[str], pd.DataFrame]:
    """Conservative feature screening on training data only.

    Drops constant, excessively missing, non-finite-dominated and perfectly
    redundant features. The final holdout is never consulted.
    """
    train = frame.loc[train_mask, candidates]
    report_rows = []
    kept: list[str] = []

    for c in candidates:
        s = train[c]
        miss = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        std = float(s.std(skipna=True)) if nunique > 1 else 0.0
        if miss > max_missing:
            report_rows.append((c, "dropped", f"missing rate {miss:.2%}"))
            continue
        if nunique <= 1 or not np.isfinite(std) or std <= 1e-12:
            report_rows.append((c, "dropped", "constant / zero variance"))
            continue
        if not np.isfinite(s.dropna().to_numpy()).all():
            report_rows.append((c, "dropped", "non-finite values"))
            continue
        kept.append(c)

    # Redundancy pruning (keep the first of each near-duplicate pair).
    corr = train[kept].corr().abs()
    drop_redundant: set[str] = set()
    for i, a in enumerate(kept):
        if a in drop_redundant:
            continue
        for b in kept[i + 1 :]:
            if b in drop_redundant:
                continue
            v = corr.loc[a, b]
            if pd.notna(v) and v >= corr_threshold:
                drop_redundant.add(b)
                report_rows.append((b, "dropped", f"redundant with {a} (|r|={v:.4f})"))
    final = [c for c in kept if c not in drop_redundant]
    for c in final:
        report_rows.append((c, "kept", ""))

    report = pd.DataFrame(report_rows, columns=["feature", "decision", "reason"])
    LOGGER.info(
        "Feature selection: %d candidates -> %d retained", len(candidates), len(final)
    )
    return final, report
