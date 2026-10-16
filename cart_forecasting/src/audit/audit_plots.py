"""Figures for the simulation audit report."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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

C_OBS = "#1f4e79"      # observed data
C_SIM = "#2a9d8f"      # simulated outcome
C_PRED = "#d1495b"     # model prediction
C_ALT = "#8a2be2"


def _save(fig, outdir: Path, name: str) -> Path:
    path = Path(outdir) / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Figure written: %s", path.name)
    return path


def inventory_timeline(series: pd.DataFrame, red: float, outdir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    t = pd.to_datetime(series["decision_time"])
    axes[0].plot(t, series["baseline_cart"], color=C_OBS, lw=0.9, label="Observed baseline (data)")
    axes[0].plot(t, series["sim_cart"], color=C_SIM, lw=0.9, label="Simulated policy (estimate)")
    axes[0].axhline(red, color="red", ls="--", lw=1, label=f"RED = {red:.0f}")
    axes[0].fill_between(t, 0, series["baseline_cart"].max() * 1.05,
                         where=series["baseline_cart"] < red, color="red", alpha=0.12, step="mid")
    axes[0].set_ylabel("Cart count")
    axes[0].set_title("Historical vs simulated inventory — audited simulator")
    axes[0].legend(fontsize=8, ncol=3, loc="upper right")

    axes[1].fill_between(t, 0, series["surplus"], color=C_SIM, alpha=0.5,
                         label="Surplus carts from replenishment")
    axes[1].plot(t, series["carts_consumed"], color=C_PRED, lw=0.7,
                 label="Surplus consumed by customers")
    axes[1].set_ylabel("Carts")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lab in axes[1].get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
    return _save(fig, outdir, "inventory_timeline.png")


def replenishment_timing(
    series: pd.DataFrame, orders: pd.DataFrame, red: float, outdir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    t = pd.to_datetime(series["decision_time"])
    ax.plot(t, series["baseline_cart"], color=C_OBS, lw=0.8, alpha=0.8, label="Observed baseline")
    ax.plot(t, series["sim_cart"], color=C_SIM, lw=1.0, label="Simulated policy")
    ax.axhline(red, color="red", ls="--", lw=1)
    if len(orders):
        d = pd.to_datetime(orders["decision_time"])
        e = pd.to_datetime(orders["delivery_record_time"])
        for di, ei in zip(d, e):
            if pd.notna(ei):
                ax.annotate(
                    "", xy=(ei, red + 2), xytext=(di, red + 2),
                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.1),
                )
        ax.scatter(d, [red + 2] * len(d), marker="o", s=22, color="darkorange",
                   zorder=5, label="Decision")
        ax.scatter(e.dropna(), [red + 2] * e.notna().sum(), marker="*", s=50,
                   color="green", zorder=5, label="Delivery credited")
    ax.set_ylabel("Cart count")
    ax.set_title("Replenishment decision → delivery timing (arrow length = lead time)")
    ax.legend(fontsize=8, ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
    return _save(fig, outdir, "replenishment_timing.png")


def pending_orders(series: pd.DataFrame, orders: pd.DataFrame, outdir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.4]})
    t = pd.to_datetime(series["decision_time"])
    axes[0].step(t, series["pending_quantity"], where="post", color=C_ALT, lw=1.0)
    axes[0].fill_between(t, 0, series["pending_quantity"], step="post", color=C_ALT, alpha=0.3)
    axes[0].set_ylabel("Carts in transit")
    axes[0].set_title("Pending replenishment queue (carts ordered but not yet delivered)")

    if len(orders):
        for k, (_, o) in enumerate(orders.iterrows()):
            start = pd.Timestamp(o["decision_time"])
            end = (
                pd.Timestamp(o["delivery_record_time"])
                if pd.notna(o["delivery_record_time"])
                else pd.Timestamp(o["expected_completion_time"])
            )
            color = "green" if o["status"] == "COMPLETED" else "grey"
            axes[1].plot([start, end], [k, k], color=color, lw=3, solid_capstyle="butt")
            axes[1].text(end, k, f"  {o['order_id']} (+{int(o['quantity'])})",
                         va="center", fontsize=6.5)
        axes[1].set_ylim(-1, len(orders))
    axes[1].set_ylabel("Order")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Order lifecycle: decision → delivery (green = COMPLETED)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lab in axes[1].get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
    return _save(fig, outdir, "pending_orders.png")


def event_attribution(att: pd.DataFrame, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1, 1.3]})
    hist = att[~att["event_id"].astype(str).str.startswith("NEW-")]
    order = ["PREVENTED_ONSET", "DELAYED_ONSET", "REDUCED_DURATION",
             "REDUCED_SEVERITY", "UNCHANGED"]
    counts = hist["outcome"].value_counts().reindex(order).fillna(0)
    colors = ["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#d1495b"]
    axes[0].bar(range(len(counts)), counts.to_numpy(), color=colors)
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels([o.replace("_", "\n").title() for o in order], fontsize=7)
    axes[0].set_ylabel("Historical events")
    axes[0].set_title("Simulated outcome per historical event")
    for i, v in enumerate(counts.to_numpy()):
        axes[0].text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)

    y = np.arange(len(hist))
    axes[1].barh(y, hist["actual_duration_minutes"], color=C_OBS, label="Observed duration")
    axes[1].barh(y, hist["residual_shortage_minutes"], color=C_SIM,
                 label="Residual under policy")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([f"E{int(e)}" for e in hist["event_id"]], fontsize=7)
    axes[1].set_xlabel("Shortage minutes")
    axes[1].set_title("Shortage duration per event: observed vs simulated")
    axes[1].legend(fontsize=8)
    axes[1].invert_yaxis()
    return _save(fig, outdir, "event_attribution.png")


def replay_vs_closed_loop_timeseries(
    replay_series: pd.DataFrame,
    closed_series: pd.DataFrame,
    replay_orders: pd.DataFrame,
    closed_orders: pd.DataFrame,
    red: float,
    outdir: Path,
) -> Path:
    """Observed telemetry against both backtest modes on one time axis."""
    fig, axes = plt.subplots(
        2, 1, figsize=(11.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]}
    )
    t = pd.to_datetime(replay_series["decision_time"])
    tc = pd.to_datetime(closed_series["decision_time"])

    ax = axes[0]
    ax.fill_between(
        t, 0, replay_series["baseline_cart"].max() * 1.08,
        where=replay_series["baseline_cart"] < red,
        color="red", alpha=0.10, step="mid", label="Observed shortage window",
    )
    ax.plot(t, replay_series["baseline_cart"], color=C_OBS, lw=1.3,
            label="Actual observed cart count (telemetry)", zorder=4)
    ax.plot(t, replay_series["sim_cart"], color=C_SIM, lw=1.1, ls="--",
            label="Backtest A — fixed-action replay (simulated)", zorder=3)
    ax.plot(tc, closed_series["sim_cart"], color=C_ALT, lw=1.1, ls=":",
            label="Backtest B — closed-loop (simulated)", zorder=3)
    ax.axhline(red, color="red", ls="--", lw=1.1, label=f"RED threshold = {red:.0f}")

    if len(replay_orders):
        d = pd.to_datetime(replay_orders["delivery_record_time"]).dropna()
        idx = replay_series.set_index(t)["sim_cart"].reindex(d, method="nearest")
        ax.scatter(d, idx.to_numpy(), marker="*", s=90, color=C_SIM,
                   edgecolor="black", linewidth=0.4, zorder=6,
                   label="Replay delivery")
    if len(closed_orders):
        dc = pd.to_datetime(closed_orders["delivery_record_time"]).dropna()
        idxc = closed_series.set_index(tc)["sim_cart"].reindex(dc, method="nearest")
        ax.scatter(dc, idxc.to_numpy(), marker="P", s=55, color=C_ALT,
                   edgecolor="black", linewidth=0.4, zorder=6,
                   label="Closed-loop delivery")

    ax.set_ylabel("Cart count")
    ax.set_title(
        "Holdout backtest — actual cart availability versus fixed-action replay "
        "and closed-loop simulation"
    )
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", framealpha=0.9)
    ax.set_ylim(0, replay_series["baseline_cart"].max() * 1.22)

    ax2 = axes[1]
    ax2.fill_between(t, 0, replay_series["surplus"], color=C_SIM, alpha=0.40,
                     step="post", label="Replay surplus in bay")
    ax2.fill_between(tc, 0, closed_series["surplus"], color=C_ALT, alpha=0.30,
                     step="post", label="Closed-loop surplus in bay")
    ax2.set_ylabel("Surplus carts")
    ax2.set_xlabel("Time")
    ax2.legend(fontsize=8, ncol=2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lab in ax2.get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
    return _save(fig, outdir, "actual_vs_replay_vs_closed_loop.png")


def replay_vs_closed_loop(replay: dict, closed: dict, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.4))
    metrics = [
        ("Shortage events", "policy_events"),
        ("Shortage minutes", "policy_shortage_minutes"),
        ("Retrieval trips", "n_replenishment_actions"),
        ("Carts moved", "total_carts_recommended"),
    ]
    for ax, (title, key) in zip(axes, metrics):
        vals = [replay[key], closed[key]]
        ax.bar(["Replay\n(Mode A)", "Closed-loop\n(Mode B)"], vals, color=[C_OBS, C_ALT])
        ax.set_title(title, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.25 + 1)
    fig.suptitle("Fixed-action replay vs closed-loop simulation (same policy, same data)",
                 fontsize=10)
    return _save(fig, outdir, "replay_vs_closed_loop.png")


def lead_time_sensitivity(ofat: pd.DataFrame, outdir: Path) -> Path:
    sub = ofat[ofat["parameter"] == "lead_time_min"].copy()
    sub["value"] = sub["value"].astype(float)
    sub = sub.sort_values("value")
    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    ax1.plot(sub["value"], sub["severity_reduction_pct"], "o-", color=C_SIM,
             label="Severity reduction")
    ax1.plot(sub["value"], sub["event_reduction_pct"], "s--", color=C_OBS,
             label="Event reduction")
    ax1.set_xlabel("Assumed replenishment lead time (minutes)")
    ax1.set_ylabel("Simulated reduction (%)")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.bar(sub["value"], sub["n_replenishment_actions"], width=1.6, alpha=0.2,
            color=C_PRED, label="Trips")
    ax2.set_ylabel("Retrieval trips")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax1.set_title("Sensitivity to the assumed lead time")
    return _save(fig, outdir, "lead_time_sensitivity.png")


def target_stock_sensitivity(ofat: pd.DataFrame, outdir: Path) -> Path:
    sub = ofat[ofat["parameter"] == "target_stock"].copy()
    sub["value"] = sub["value"].astype(float)
    sub = sub.sort_values("value")
    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    ax1.plot(sub["value"], sub["severity_reduction_pct"], "o-", color=C_SIM,
             label="Severity reduction")
    ax1.plot(sub["value"], sub["event_reduction_pct"], "s--", color=C_OBS,
             label="Event reduction")
    ax1.set_xlabel("Target stock (carts)")
    ax1.set_ylabel("Simulated reduction (%)")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.bar(sub["value"], sub["carts_per_day"], width=1.4, alpha=0.2, color=C_PRED,
            label="Carts moved per day")
    ax2.set_ylabel("Carts moved per day")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax1.set_title("Sensitivity to target stock")
    return _save(fig, outdir, "target_stock_sensitivity.png")


def shortage_vs_workload(grid: pd.DataFrame, scenarios: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    sc = ax.scatter(
        grid["carts_per_day"], grid["severity_reduction_pct"],
        c=grid["lead_time_min"], cmap="viridis", s=38, alpha=0.85,
    )
    plt.colorbar(sc, ax=ax, label="Lead time (min)")
    for _, r in scenarios.iterrows():
        ax.scatter(r["carts_per_day"], r["severity_reduction_pct"], marker="*", s=220,
                   edgecolor="black", linewidth=0.6,
                   color={"BEST_CASE": "#2a9d8f", "BASE_CASE": "#1f4e79",
                          "CONSERVATIVE": "#d1495b"}[r["scenario"]])
        ax.annotate(r["scenario"].replace("_", " ").title(),
                    (r["carts_per_day"], r["severity_reduction_pct"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("Employee workload (carts retrieved per day)")
    ax.set_ylabel("Simulated shortage severity reduction (%)")
    ax.set_title("Shortage reduction versus retrieval workload")
    return _save(fig, outdir, "shortage_vs_workload.png")


def sensitivity_heatmap(grid: pd.DataFrame, outdir: Path) -> Path:
    piv = grid.pivot_table(
        index="lead_time_min", columns="target_stock", values="severity_reduction_pct"
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v:.0f}" for v in piv.index])
    ax.set_xlabel("Target stock (carts)")
    ax.set_ylabel("Lead time (minutes)")
    ax.set_title("Simulated shortage severity reduction (%)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy()[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="black")
    plt.colorbar(im, ax=ax, label="Severity reduction (%)")
    ax.grid(False)
    return _save(fig, outdir, "sensitivity_heatmap.png")


def before_after_comparison(before: dict, after: dict, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.4))
    items = [
        ("Shortage events", "policy_events"),
        ("Shortage minutes", "policy_shortage_minutes"),
        ("Severity", "policy_severity_cart_records"),
        ("Retrieval trips", "n_replenishment_actions"),
        ("Carts moved", "total_carts_recommended"),
    ]
    for ax, (title, key) in zip(axes, items):
        vals = [before.get(key, np.nan), after.get(key, np.nan)]
        ax.bar(["Original", "Audited"], vals, color=[C_PRED, C_SIM])
        ax.set_title(title, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max([v for v in vals if np.isfinite(v)] + [1]) * 1.3)
    fig.suptitle("Original simulator vs audited simulator (same holdout window)", fontsize=10)
    return _save(fig, outdir, "before_after_comparison.png")


def operational_cost(before: dict, after: dict, scenarios: pd.DataFrame, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = ["Baseline\n(no policy)", "Original\nsimulator", "Audited\nsimulator"]
    vals = [after["baseline_cost_J"], before["policy_cost_J"], after["policy_cost_J"]]
    axes[0].bar(labels, vals, color=["#595959", C_PRED, C_SIM])
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Operational cost J")
    axes[0].set_title("Operational cost comparison")

    axes[1].bar(scenarios["scenario"].str.replace("_", "\n"), scenarios["policy_cost_J"],
                color=["#2a9d8f", "#1f4e79", "#d1495b"])
    for i, v in enumerate(scenarios["policy_cost_J"]):
        axes[1].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].axhline(after["baseline_cost_J"], color="black", ls="--", lw=1,
                    label="Baseline cost")
    axes[1].set_ylabel("Operational cost J")
    axes[1].set_title("Cost by operating scenario")
    axes[1].legend(fontsize=8)
    return _save(fig, outdir, "operational_cost.png")
