"""Early-warning engine: alert state machine and lead-time evaluation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    ALERT_HYSTERESIS,
    RED_THRESHOLD,
)

LOGGER = logging.getLogger(__name__)

NORMAL, WARNING, CRITICAL = "NORMAL", "WARNING", "CRITICAL"


@dataclass
class AlertEngineConfig:
    warning_risk: float = 0.20
    critical_risk: float = 0.50
    cooldown_min: float = 20.0
    hysteresis: float = ALERT_HYSTERESIS
    red_threshold: int = RED_THRESHOLD
    consecutive_to_fire: int = 1


@dataclass
class AlertEngine:
    """Stateful alerting with deduplication, cooldown, hysteresis, escalation.

    Observing ``Count <= RED`` forces ``state = CRITICAL`` and flags a
    *reactive* alert; those are recorded separately, never credited as
    predictive warnings and never start the predictive cooldown.
    """

    cfg: AlertEngineConfig = field(default_factory=AlertEngineConfig)
    state: str = NORMAL
    last_alert_time: pd.Timestamp | None = None
    _streak: int = 0

    def reset(self) -> None:
        self.state = NORMAL
        self.last_alert_time = None
        self._streak = 0

    def step(self, ts: pd.Timestamp, risk: float, observed_cart: float) -> dict:
        cfg = self.cfg
        reactive = observed_cart <= cfg.red_threshold
        prev_state = self.state

        if risk >= cfg.warning_risk:
            self._streak += 1
        else:
            self._streak = 0

        desired = NORMAL
        if self._streak >= cfg.consecutive_to_fire:
            desired = CRITICAL if risk >= cfg.critical_risk else WARNING

        # Hysteresis: stay elevated until risk drops clearly below entry level.
        # Capped at half the entry level so a small band (e.g. 0.066) cannot
        # produce a negative exit level that latches WARNING forever.
        if prev_state in (WARNING, CRITICAL) and desired == NORMAL:
            exit_level = cfg.warning_risk - min(cfg.hysteresis, cfg.warning_risk / 2.0)
            if risk > exit_level:
                desired = WARNING

        # An observed bay at or below RED is a shortage now, whatever the forecast.
        if reactive:
            desired = CRITICAL

        cooldown_active = (
            self.last_alert_time is not None
            and (ts - self.last_alert_time).total_seconds() / 60.0 < cfg.cooldown_min
        )

        escalated = prev_state == WARNING and desired == CRITICAL
        new_alert = desired in (WARNING, CRITICAL) and (
            prev_state == NORMAL or escalated
        )
        if new_alert and cooldown_active and not escalated:
            new_alert = False

        if new_alert and not reactive:
            self.last_alert_time = ts

        self.state = desired
        return {
            "alert_time": ts,
            "state": desired,
            "prev_state": prev_state,
            "risk": float(risk),
            "observed_cart": float(observed_cart),
            "new_predictive_alert": bool(new_alert and not reactive),
            "reactive_critical": bool(reactive),
            "escalation": bool(escalated),
            "recovery": bool(prev_state in (WARNING, CRITICAL) and desired == NORMAL),
            "suppressed_by_cooldown": bool(
                desired in (WARNING, CRITICAL) and cooldown_active and prev_state == NORMAL
            ),
        }


def run_alert_engine(
    frame: pd.DataFrame, risk: np.ndarray, cfg: AlertEngineConfig
) -> pd.DataFrame:
    """Replay telemetry chronologically through the alert state machine."""
    engine = AlertEngine(cfg)
    rows = []
    carts = frame["cart_current"].to_numpy(dtype=float)
    times = pd.to_datetime(frame["decision_time"]).to_numpy()
    for i in range(len(frame)):
        rows.append(engine.step(pd.Timestamp(times[i]), float(risk[i]), carts[i]))
    out = pd.DataFrame(rows)
    out.index = frame.index
    return out


def evaluate_warnings(
    alerts: pd.DataFrame,
    events: pd.DataFrame,
    frame: pd.DataFrame,
    max_lead_min: float = 60.0,
) -> tuple[dict, pd.DataFrame]:
    """Match predictive alerts to independent shortage events.

    A warning counts for an event if it fires strictly before the event onset
    and no earlier than ``max_lead_min`` before it, while the store is not
    already in shortage.
    """
    alert_times = pd.to_datetime(
        alerts.loc[alerts["new_predictive_alert"], "alert_time"]
    ).sort_values()
    rows = []
    for _, ev in events.iterrows():
        onset = pd.Timestamp(ev["onset_time"])
        window_start = onset - pd.Timedelta(minutes=max_lead_min)
        cand = alert_times[(alert_times < onset) & (alert_times >= window_start)]
        if len(cand):
            first = cand.min()
            lead = (onset - first).total_seconds() / 60.0
        else:
            first, lead = pd.NaT, np.nan
        rows.append(
            {
                "event_id": ev["event_id"],
                "onset_time": onset,
                "recovery_time": ev["recovery_time"],
                "min_cart": ev["min_cart"],
                "duration_min": ev["duration_min"],
                "severity_cart_records": ev["severity_cart_records"],
                "detected": bool(len(cand) > 0),
                "first_warning_time": first,
                "lead_time_min": lead,
                "n_warnings_in_window": int(len(cand)),
            }
        )
    matched = pd.DataFrame(rows)

    n_events = len(matched)
    detected = int(matched["detected"].sum()) if n_events else 0
    leads = matched.loc[matched["detected"], "lead_time_min"].to_numpy()

    used = set()
    for _, r in matched.loc[matched["detected"]].iterrows():
        used.add(pd.Timestamp(r["first_warning_time"]))
    true_alerts = 0
    for t in alert_times:
        onset_after = events["onset_time"].astype("datetime64[ns]")
        future = onset_after[onset_after > t]
        if len(future) and (future.min() - t).total_seconds() / 60.0 <= max_lead_min:
            true_alerts += 1
    n_alerts = int(len(alert_times))
    false_alerts = n_alerts - true_alerts

    span_days = max(
        (
            pd.to_datetime(frame["decision_time"]).max()
            - pd.to_datetime(frame["decision_time"]).min()
        ).total_seconds()
        / 86400.0,
        1e-9,
    )

    summary = {
        "n_events": n_events,
        "n_detected": detected,
        "event_recall": detected / n_events if n_events else np.nan,
        "n_predictive_alerts": n_alerts,
        "n_true_alerts": true_alerts,
        "n_false_alerts": false_alerts,
        "event_precision": true_alerts / n_alerts if n_alerts else np.nan,
        "false_alerts_per_day": false_alerts / span_days,
        "alerts_per_day": n_alerts / span_days,
        "mean_lead_time_min": float(np.mean(leads)) if len(leads) else np.nan,
        "median_lead_time_min": float(np.median(leads)) if len(leads) else np.nan,
        "pct_warned_ge_10min": float(np.mean(matched["lead_time_min"].fillna(-1) >= 10) * 100)
        if n_events
        else np.nan,
        "pct_warned_ge_20min": float(np.mean(matched["lead_time_min"].fillna(-1) >= 20) * 100)
        if n_events
        else np.nan,
        "n_reactive_criticals": int(alerts["reactive_critical"].sum()),
        "evaluation_days": span_days,
    }
    return summary, matched
