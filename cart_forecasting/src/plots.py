"""Figure generation for the DOCX report."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import AMBER_THRESHOLD, FIGURE_DIR, RED_THRESHOLD  # noqa: E402

LOGGER = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C_CART = "#1f4e79"
C_PRED = "#d1495b"
C_IN = "#2a9d8f"
C_OUT = "#e9c46a"


def _save(fig, name: str) -> Path:
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Figure written: %s", path.name)
    return path


def historical_cart_traffic(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    t = df["decision_time"]
    axes[0].plot(t, df["Count"], color=C_CART, lw=0.8)
    axes[0].axhline(RED_THRESHOLD, color="red", ls="--", lw=1, label=f"RED = {RED_THRESHOLD}")
    axes[0].axhline(
        AMBER_THRESHOLD, color="orange", ls="--", lw=1, label=f"AMBER = {AMBER_THRESHOLD}"
    )
    axes[0].fill_between(
        t, 0, df["Count"].max() * 1.05, where=df["Count"] < RED_THRESHOLD,
        color="red", alpha=0.15, step="mid",
    )
    axes[0].set_ylabel("Cart count")
    axes[0].set_title("Site 163 — cart availability and customer traffic")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, df["Count_IN"], color=C_IN, lw=0.7, label="Traffic IN")
    axes[1].plot(t, df["Count_OUT"], color=C_OUT, lw=0.7, label="Traffic OUT")
    axes[1].set_ylabel("Customers / interval")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].set_xlabel("Date")
    return _save(fig, "historical_cart_traffic.png")


def actual_vs_predicted(
    frame: pd.DataFrame, y_col: str, pred_col: str, title: str, filename: str
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    t = frame["decision_time"]
    axes[0].plot(t, frame[y_col], color=C_CART, lw=0.9, label="Actual")
    axes[0].plot(t, frame[pred_col], color=C_PRED, lw=0.9, ls="--", label="Predicted")
    axes[0].axhline(RED_THRESHOLD, color="red", ls=":", lw=1)
    axes[0].set_title(title)
    axes[0].set_ylabel("Cart count")
    axes[0].legend(fontsize=8)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lab in axes[0].get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")

    axes[1].scatter(frame[y_col], frame[pred_col], s=6, alpha=0.4, color=C_CART)
    lim = [
        min(frame[y_col].min(), frame[pred_col].min()) - 2,
        max(frame[y_col].max(), frame[pred_col].max()) + 2,
    ]
    axes[1].plot(lim, lim, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Holdout calibration")
    return _save(fig, filename)


def minimum_forecast_figure(frame: pd.DataFrame, pred_col: str, lower_col: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    t = frame["decision_time"]
    ax.plot(t, frame["y_min_6"], color=C_CART, lw=0.9, label="Actual 40-min minimum")
    ax.plot(t, frame[pred_col], color=C_PRED, lw=0.9, ls="--", label="Predicted minimum")
    ax.fill_between(
        t, frame[lower_col], frame[pred_col], color=C_PRED, alpha=0.15,
        label="90% lower bound band",
    )
    ax.axhline(RED_THRESHOLD, color="red", ls=":", lw=1, label=f"RED = {RED_THRESHOLD}")
    ax.set_ylabel("Cart count")
    ax.set_title("Future 40-minute minimum cart forecast with conformal lower bound (holdout)")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
    return _save(fig, "minimum_cart_forecast.png")


def shortage_warning_timeline(
    frame: pd.DataFrame, alerts: pd.DataFrame, events: pd.DataFrame, risk: np.ndarray
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    t = frame["decision_time"]
    axes[0].plot(t, frame["cart_current"], color=C_CART, lw=0.9, label="Observed carts")
    axes[0].axhline(RED_THRESHOLD, color="red", ls="--", lw=1)
    for _, ev in events.iterrows():
        axes[0].axvspan(ev["onset_time"], ev["recovery_time"], color="red", alpha=0.15)
    warn = alerts[alerts["new_predictive_alert"]]
    if len(warn):
        axes[0].scatter(
            warn["alert_time"],
            warn["observed_cart"],
            marker="v",
            s=28,
            color="darkorange",
            zorder=5,
            label="Predictive warning",
        )
    axes[0].set_ylabel("Cart count")
    axes[0].set_title("Shortage warning timeline (holdout)")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].plot(t, risk, color="#8a2be2", lw=0.8)
    axes[1].set_ylabel("P(shortage within 40 min)")
    axes[1].set_ylim(0, 1)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lab in axes[1].get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
    return _save(fig, "shortage_warning_timeline.png")


def replenishment_timeline(series: pd.DataFrame, decisions: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.0))
    t = series["decision_time"]
    ax.plot(t, series["cart_current"], color=C_CART, lw=0.9, label="Observed baseline")
    ax.plot(t, series["sim_cart"], color="#2a9d8f", lw=0.9, label="Simulated with policy")
    ax.axhline(RED_THRESHOLD, color="red", ls="--", lw=1, label=f"RED = {RED_THRESHOLD}")
    if len(decisions):
        ax.scatter(
            pd.to_datetime(decisions["decision_time"]),
            decisions["current_cart"],
            marker="^",
            s=36,
            color="darkorange",
            zorder=5,
            label="Replenishment decision",
        )
        ax.scatter(
            pd.to_datetime(decisions["effective_time"]),
            decisions["current_cart"] + decisions["quantity"],
            marker="*",
            s=44,
            color="green",
            zorder=5,
            label="Carts delivered (decision + lead time)",
        )
    ax.set_ylabel("Cart count")
    ax.set_title("Replenishment decisions and simulated inventory (holdout)")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
    return _save(fig, "replenishment_timeline.png")


def policy_comparison(metrics: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    pairs = [
        ("Shortage events", metrics["baseline_events"], metrics["policy_events"]),
        (
            "Shortage minutes",
            metrics["baseline_shortage_minutes"],
            metrics["policy_shortage_minutes"],
        ),
        (
            "Severity (cart-records)",
            metrics["baseline_severity_cart_records"],
            metrics["policy_severity_cart_records"],
        ),
    ]
    for ax, (title, b, p) in zip(axes, pairs):
        ax.bar(["Historical\nbaseline", "Simulated\npolicy"], [b, p], color=[C_CART, "#2a9d8f"])
        ax.set_title(title, fontsize=9)
        for i, v in enumerate([b, p]):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(b, p) * 1.2 + 1)
    fig.suptitle("Observed baseline vs simulated policy (scenario estimate)", fontsize=10)
    return _save(fig, "policy_comparison.png")


def feature_importance_figure(imp: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    d = imp.iloc[::-1]
    ax.barh(d["feature"], d["importance"], color=C_CART)
    ax.set_xlabel(d["importance_type"].iloc[0] if len(d) else "importance")
    ax.set_title("Top 15 features — selected 40-minute minimum cart model")
    return _save(fig, "feature_importance.png")


def lead_time_distribution(matched: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    leads = matched.loc[matched["detected"], "lead_time_min"].dropna()
    if len(leads):
        ax.hist(leads, bins=min(12, max(3, len(leads))), color=C_CART, alpha=0.85, edgecolor="white")
        ax.axvline(leads.median(), color=C_PRED, ls="--", lw=1.4, label=f"median = {leads.median():.1f} min")
        ax.axvline(10, color="green", ls=":", lw=1.2, label="10 min")
        ax.axvline(20, color="purple", ls=":", lw=1.2, label="20 min")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No detected events", ha="center", transform=ax.transAxes)
    ax.set_xlabel("Warning lead time (minutes before shortage onset)")
    ax.set_ylabel("Number of events")
    ax.set_title("Warning lead-time distribution (holdout)")
    return _save(fig, "lead_time_distribution.png")


def risk_calibration_figure(cal: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.9, label="Perfect calibration")
    ax.plot(cal["mean_predicted"], cal["observed_rate"], "o-", color=C_CART, label="Model")
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_predicted"], r["observed_rate"]), fontsize=7,
                    textcoords="offset points", xytext=(4, -8))
    ax.set_xlabel("Mean predicted shortage probability")
    ax.set_ylabel("Observed shortage rate")
    ax.set_title("Shortage risk calibration (holdout)")
    ax.legend(fontsize=8)
    return _save(fig, "risk_calibration.png")
