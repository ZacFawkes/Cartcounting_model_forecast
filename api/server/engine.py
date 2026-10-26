"""Scoring and backtesting engine.

Wraps the validated forecasting models and the AUDITED replenishment decision
logic. The same ``plan_replenishment`` function drives both the online /score
endpoint and the /backtest endpoint, so an online recommendation can never
diverge from what the backtest would have produced.
"""
from __future__ import annotations

import logging
import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from server.settings import BUNDLED_DATA, MIN_HISTORY_RECORDS, MODEL_FILE

# Pipeline imports (sys.path extended in settings).
from alert_engine import AlertEngine, AlertEngineConfig  # noqa: E402
from audit.attribution import attribute_events, attribution_summary  # noqa: E402
from audit.audited_simulator import (  # noqa: E402
    ClosedLoopPredictor,
    PolicyConfig,
    ReplayPredictor,
    plan_replenishment,
    run_simulation,
)
from audit.inventory_model import (  # noqa: E402
    InventoryConfig,
    apply_capacity,
    consume_surplus,
    observed_inventory,
)
from audit.order_book import OrderBook  # noqa: E402
from data_loader import load_raw, prepare  # noqa: E402
from feature_engineering import apply_value_clipper, build_features  # noqa: E402
from forecasting import ConformalCalibrator  # noqa: E402
from target_builder import build_targets  # noqa: E402

LOGGER = logging.getLogger("cart_api.engine")

TARGETS = ["y_cart_3", "y_cart_6", "y_min_3", "y_min_6"]
HORIZON_NOTE = (
    "Horizons are record-based: +3 records is nominally 20 minutes and +6 records "
    "nominally 40 minutes. Actual elapsed time varies with telemetry cadence."
)
BACKTEST_CAVEAT = (
    "Forecast accuracy is measured. Shortage reduction, trips and carts moved are "
    "simulator outputs under documented assumptions, not observed causal effects. "
    "The estimate is highly sensitive to the inventory consumption model."
)


class ModelNotLoaded(RuntimeError):
    pass


class InsufficientHistory(ValueError):
    pass


# --------------------------------------------------------------------------
# Model bundle
# --------------------------------------------------------------------------
@dataclass
class ModelBundle:
    estimators: dict
    selected: dict
    feature_cols: list
    clip_bounds: pd.DataFrame
    conformal: ConformalCalibrator
    cart_ceiling: float
    red_threshold: float
    version: str

    @classmethod
    def load(cls, path=MODEL_FILE, red_threshold: float = 15.0) -> "ModelBundle":
        if not path.exists():
            raise ModelNotLoaded(
                f"Model bundle not found at {path}. Run cart_forecasting/main.py first."
            )
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        conf = ConformalCalibrator(alpha=payload.get("conformal_alpha", 0.10))
        conf.residuals_ = np.asarray(payload["conformal_residuals"], dtype=float)
        mtime = pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y%m%d-%H%M%S")
        LOGGER.info("Loaded model bundle (%d features, version %s)",
                    len(payload["feature_cols"]), mtime)
        return cls(
            estimators=payload["estimators"],
            selected=payload["selected"],
            feature_cols=payload["feature_cols"],
            clip_bounds=payload["clip_bounds"],
            conformal=conf,
            cart_ceiling=float(payload.get("cart_ceiling", 64.0)),
            red_threshold=red_threshold,
            version=f"cart-forecast-{mtime}",
        )

    # ---------------------------------------------------------------- predict
    def predict_row(self, feature_row: pd.DataFrame) -> dict:
        out = {}
        for target, key in (
            ("y_min_6", "pred_min_6"),
            ("y_min_3", "pred_min_3"),
            ("y_cart_3", "pred_cart_3"),
            ("y_cart_6", "pred_cart_6"),
        ):
            raw = float(self.estimators[target].predict(feature_row)[0])
            out[key] = float(np.clip(raw, 0.0, self.cart_ceiling))
        out["lower_min_6"] = float(
            self.conformal.lower_bound(np.array([out["pred_min_6"]]))[0]
        )
        out["risk"] = float(
            self.conformal.exceedance_probability(
                np.array([out["pred_min_6"]]), self.red_threshold
            )[0]
        )
        return out

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.feature_cols]
        out = frame.copy()
        for target in TARGETS:
            raw = self.estimators[target].predict(X)
            out[f"pred_{target}"] = np.clip(raw, 0.0, self.cart_ceiling)
        out["lower_y_min_6"] = self.conformal.lower_bound(out["pred_y_min_6"].to_numpy())
        out["risk"] = self.conformal.exceedance_probability(
            out["pred_y_min_6"].to_numpy(), self.red_threshold
        )
        return out


# --------------------------------------------------------------------------
# Per-session online state
# --------------------------------------------------------------------------
@dataclass
class Session:
    session_id: str
    book: OrderBook = field(default_factory=OrderBook)
    alert: AlertEngine = field(default_factory=AlertEngine)
    surplus: float = 0.0
    last_record_time: pd.Timestamp | None = None
    last_decision_time: pd.Timestamp | None = None
    reserve_left: float = np.inf
    n_calls: int = 0


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str, alert_cfg: AlertEngineConfig) -> Session:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                s = Session(session_id=session_id, alert=AlertEngine(alert_cfg))
                self._sessions[session_id] = s
            else:
                s.alert.cfg = alert_cfg
            return s

    def reset(self, session_id: str | None = None) -> int:
        with self._lock:
            if session_id is None:
                n = len(self._sessions)
                self._sessions.clear()
                return n
            return 1 if self._sessions.pop(session_id, None) is not None else 0

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "calls": s.n_calls,
                    "surplus": round(s.surplus, 2),
                    "pending_orders": len(s.book.pending),
                    "pending_carts": s.book.pending_quantity(),
                    "completed_orders": len(s.book.completed),
                    "alert_state": s.alert.state,
                    "last_record_time": str(s.last_record_time),
                }
                for s in self._sessions.values()
            ]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def records_to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for c in ("DVRDate", "next_DVRDate"):
        df[c] = pd.to_datetime(df[c])
    for c in ("Count", "Count_IN", "Count_OUT"):
        df[c] = pd.to_numeric(df[c])
    n = len(df)
    df["CountID"] = range(n)
    df["CNN_Index"] = range(n)
    for c, v in (("KDVR", 215), ("ChanNo", 0), ("CatID", 4000), ("AreaID", 69)):
        df[c] = v
    if "siteKey" not in df or df["siteKey"].isna().all():
        df["siteKey"] = 163
    return prepare(df)


def _policy_from(d: dict) -> PolicyConfig:
    return PolicyConfig(
        red_threshold=d["red_threshold"],
        amber_threshold=d["amber_threshold"],
        target_stock=d["target_stock"],
        risk_threshold=d["risk_threshold"],
        lead_time_min=d["lead_time_min"],
        cooldown_min=d["cooldown_min"],
        min_carts_per_trip=d["min_carts_per_trip"],
        max_carts_per_trip=d["max_carts_per_trip"],
        quantity_rule=d["quantity_rule"],
        planning_horizon=d.get("planning_horizon", 6),
    )


def _inventory_from(d: dict) -> InventoryConfig:
    return InventoryConfig(
        bay_capacity=d["bay_capacity"],
        usage=d["usage"],
        reserve=d["reserve"],
        consumption_model=d["consumption_model"],
        retention_halflife_min=d["retention_halflife_min"],
    )


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.bundle: ModelBundle | None = None
        self.sessions = SessionStore()
        self.started_at = time.time()
        self._bundled: pd.DataFrame | None = None

    def load(self) -> None:
        self.bundle = ModelBundle.load()

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def _require(self) -> ModelBundle:
        if self.bundle is None:
            raise ModelNotLoaded("Model bundle is not loaded")
        return self.bundle

    def bundled_data(self) -> pd.DataFrame:
        if self._bundled is None:
            self._bundled = prepare(load_raw(BUNDLED_DATA))
        return self._bundled

    # ------------------------------------------------------------------ score
    def score(self, req: dict) -> dict:
        bundle = self._require()
        policy_d = req["policy"]
        inv_d = req["inventory"]
        policy = _policy_from(policy_d)
        inventory = _inventory_from(inv_d)
        bundle.red_threshold = policy.red_threshold

        raw = records_to_frame([r for r in req["history"]])
        if len(raw) < MIN_HISTORY_RECORDS:
            raise InsufficientHistory(
                f"{len(raw)} usable records supplied; at least {MIN_HISTORY_RECORDS} "
                f"consecutive records are required to build the feature window."
            )
        feats = apply_value_clipper(build_features(raw), bundle.clip_bounds)
        row = feats.iloc[[-1]]
        X = row[bundle.feature_cols]
        if X.isna().any(axis=None):
            missing = [c for c in bundle.feature_cols if X[c].isna().any()]
            raise InsufficientHistory(
                f"Feature window incomplete ({len(missing)} features unavailable, e.g. "
                f"{missing[:3]}). Supply more consecutive history."
            )

        ts = pd.Timestamp(row["decision_time"].iloc[0])
        ts_start = pd.Timestamp(row["DVRDate"].iloc[0])
        baseline = float(row["Count"].iloc[0])
        traffic_in = float(row["Count_IN"].iloc[0])
        duration = float(row["interval_duration"].iloc[0])

        session_id = req.get("session_id") or f"site-{req['site_key']}"
        alert_cfg = AlertEngineConfig(
            warning_risk=policy.risk_threshold,
            critical_risk=min(0.95, policy.risk_threshold + 0.25),
            cooldown_min=policy.cooldown_min,
            red_threshold=policy.red_threshold,
        )
        session = self.sessions.get(session_id, alert_cfg)
        reserve = inventory.reserve_scenario
        effective_lead = policy.lead_time_min + reserve.availability_delay_min

        advance = session.last_record_time is None or ts > session.last_record_time
        if advance:
            for o in session.book.deliver_due(record_time=ts, record_start_time=ts_start):
                session.surplus += o.delivered_quantity
            session.surplus, _absorbed = apply_capacity(session.surplus, baseline, inventory)
            session.surplus, _consumed = consume_surplus(
                session.surplus, baseline, traffic_in, duration, inventory
            )
            session.last_record_time = ts
            session.n_calls += 1

        current_sim = observed_inventory(baseline, session.surplus, inventory)

        # Project the surviving surplus over the planning horizon using recent
        # traffic only, then forecast on the surplus-adjusted state.
        recent_in = float(feats["Count_IN"].tail(3).mean())
        projected = session.surplus
        for _ in range(policy.planning_horizon):
            projected, _c = consume_surplus(projected, baseline, recent_in, duration, inventory)

        pred = bundle.predict_row(X)
        for k in ("pred_min_6", "pred_min_3", "pred_cart_3", "pred_cart_6"):
            pred[k] += projected
        pred["lower_min_6"] = float(
            bundle.conformal.lower_bound(np.array([pred["pred_min_6"]]))[0]
        )
        pred["risk"] = float(
            bundle.conformal.exceedance_probability(
                np.array([pred["pred_min_6"]]), policy.red_threshold
            )[0]
        )

        alert = session.alert.step(ts, pred["risk"], current_sim) if advance else {
            "state": session.alert.state,
            "new_predictive_alert": False,
            "reactive_critical": current_sim < policy.red_threshold,
        }

        cooldown_active = (
            session.last_decision_time is not None
            and (ts - session.last_decision_time).total_seconds() / 60.0 < policy.cooldown_min
        )
        plan, order = None, None
        if advance and not cooldown_active:
            plan = plan_replenishment(
                pred=pred,
                current_sim=current_sim,
                observed_cart=baseline,
                pending_qty=session.book.pending_quantity(),
                policy=policy,
                inventory=inventory,
                reserve_left=session.reserve_left,
                effective_lead=effective_lead,
            )
        if plan and "skip_reason" not in plan:
            order = session.book.place(
                decision_time=ts,
                quantity=plan["quantity"],
                lead_time_min=effective_lead,
                urgency=plan["urgency"],
                risk_at_decision=pred["risk"],
                predicted_min_at_decision=pred["pred_min_6"],
                capacity_rejected=plan["capacity_rejected"],
            )
            session.last_decision_time = ts

        if order is not None:
            reason = (
                f"risk {pred['risk']:.2f} >= {policy.risk_threshold:.2f}; predicted 40-minute "
                f"minimum {pred['pred_min_6']:.1f} carts against a target of "
                f"{policy.target_stock:.0f}; {plan['quantity']} carts ordered after netting "
                f"{session.book.pending_quantity() - plan['quantity']:.0f} already in transit"
            )
        elif plan and "skip_reason" in plan:
            reason = f"No action: {plan['skip_reason']}"
        elif cooldown_active:
            reason = f"No action: within the {policy.cooldown_min:.0f}-minute cooldown"
        elif not advance:
            reason = "No action: record already processed for this session"
        else:
            reason = (
                f"No action: risk {pred['risk']:.2f} below the {policy.risk_threshold:.2f} "
                f"threshold"
            )

        return {
            "site_key": req["site_key"],
            "session_id": session_id,
            "decision_time": ts.isoformat(),
            "current_cart": baseline,
            "simulated_cart": round(current_sim, 2),
            "forecast": {
                "cart_at_20min": round(pred["pred_cart_3"], 2),
                "cart_at_40min": round(pred["pred_cart_6"], 2),
                "min_within_20min": round(pred["pred_min_3"], 2),
                "min_within_40min": round(pred["pred_min_6"], 2),
                "lower_bound_min_within_40min": round(pred["lower_min_6"], 2),
                "horizon_note": HORIZON_NOTE,
            },
            "shortage_risk": round(pred["risk"], 4),
            "alert_state": alert["state"],
            "new_predictive_alert": bool(alert["new_predictive_alert"]),
            "reactive_critical": bool(alert["reactive_critical"]),
            "action": {
                "should_replenish": order is not None,
                "quantity": int(order.quantity) if order else 0,
                "urgency": plan["urgency"] if plan and "urgency" in plan else "NONE",
                "order_id": order.order_id if order else None,
                "decision_time": ts.isoformat() if order else None,
                "expected_completion_time": (
                    order.expected_completion_time.isoformat() if order else None
                ),
                "actionable_before_breach": (
                    plan["actionable_before_breach"] if plan and "urgency" in plan else None
                ),
                "minutes_to_predicted_breach": (
                    None
                    if not plan or "urgency" not in plan
                    or not np.isfinite(plan["minutes_to_predicted_breach"])
                    else round(plan["minutes_to_predicted_breach"], 1)
                ),
                "reason": reason,
            },
            "pending_carts": session.book.pending_quantity(),
            "pending_orders": [
                {
                    "order_id": o.order_id,
                    "quantity": o.quantity,
                    "expected_completion_time": o.expected_completion_time.isoformat(),
                    "status": o.status,
                }
                for o in session.book.pending
            ],
            "model_version": bundle.version,
            "assumptions": {"policy": policy_d, "inventory": inv_d},
        }

    # --------------------------------------------------------------- backtest
    def backtest(self, req: dict) -> tuple[dict, dict]:
        """Return ``(response_payload, artefact_frames)``."""
        bundle = self._require()
        policy_d, inv_d = req["policy"], req["inventory"]
        policy = _policy_from(policy_d)
        inventory = _inventory_from(inv_d)
        bundle.red_threshold = policy.red_threshold

        if req.get("use_bundled_data") or not req.get("records"):
            raw = self.bundled_data().copy()
            if req.get("start"):
                raw = raw[raw["decision_time"] >= pd.Timestamp(req["start"])]
            if req.get("end"):
                raw = raw[raw["decision_time"] <= pd.Timestamp(req["end"])]
            raw = raw.reset_index(drop=True)
        else:
            raw = records_to_frame(req["records"])

        if len(raw) < MIN_HISTORY_RECORDS + policy.planning_horizon:
            raise InsufficientHistory(
                f"{len(raw)} records supplied; a backtest needs at least "
                f"{MIN_HISTORY_RECORDS + policy.planning_horizon}."
            )

        frame = build_targets(build_features(raw))
        frame = apply_value_clipper(frame, bundle.clip_bounds)
        frame = frame.reset_index(drop=True)
        frame["global_index"] = frame.index.to_numpy()

        usable = frame[bundle.feature_cols].notna().all(axis=1)
        origins = frame[usable].copy()
        if len(origins) == 0:
            raise InsufficientHistory("No record had a complete feature window.")
        origins = bundle.predict_frame(origins)

        pos = {int(gi): k for k, gi in enumerate(origins["global_index"])}
        origin_positions = {
            i: pos[int(gi)] for i, gi in enumerate(frame["global_index"]) if int(gi) in pos
        }

        if req.get("mode") == "closed_loop":
            predictor = ClosedLoopPredictor(
                full_frame=frame,
                feature_cols=bundle.feature_cols,
                clip_bounds=bundle.clip_bounds,
                estimators=bundle.estimators,
                conformal=bundle.conformal,
                red=policy.red_threshold,
                cart_bounds=(0.0, bundle.cart_ceiling),
            )
        else:
            predictor = ReplayPredictor(
                pred_min_6=origins["pred_y_min_6"].to_numpy(),
                pred_min_3=origins["pred_y_min_3"].to_numpy(),
                pred_cart_3=origins["pred_y_cart_3"].to_numpy(),
                pred_cart_6=origins["pred_y_cart_6"].to_numpy(),
                conformal=bundle.conformal,
                red=policy.red_threshold,
            )

        result = run_simulation(frame, origin_positions, predictor, policy, inventory)
        orders = result["orders"].to_frame()
        att = attribute_events(
            result["series"], result["baseline_events"], result["policy_events"],
            orders, None, policy.red_threshold,
        )

        run_id = f"bt-{pd.Timestamp.utcnow():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        payload = {
            "run_id": run_id,
            "label": req.get("label"),
            "mode": req.get("mode", "replay"),
            "window_start": str(frame["decision_time"].min()),
            "window_end": str(frame["decision_time"].max()),
            "n_records": int(len(frame)),
            "scored_origins": int(len(origins)),
            "metrics": _clean(result["metrics"]),
            "forecast_accuracy": _forecast_accuracy(origins),
            "attribution_summary": _clean(attribution_summary(att)) if len(att) else {},
            "assumptions": {"policy": policy_d, "inventory": inv_d},
            "caveat": BACKTEST_CAVEAT,
        }
        if req.get("include_decisions", True):
            payload["decisions"] = _records(result["decisions"])
        if req.get("include_events", True):
            payload["shortage_events"] = _records(result["baseline_events"])
        if req.get("include_series", False):
            payload["series"] = _records(
                result["series"][
                    ["decision_time", "Count", "sim_cart", "surplus", "pending_quantity"]
                ]
            )

        frames = {
            "decisions": result["decisions"],
            "order_book": orders,
            "baseline_events": result["baseline_events"],
            "policy_events": result["policy_events"],
            "event_attribution": att,
            "series": result["series"][
                ["DVRDate", "decision_time", "Count", "sim_cart", "surplus",
                 "pending_quantity", "carts_consumed"]
            ],
            "forecasts": origins[
                ["decision_time", "Count"] + [f"pred_{t}" for t in TARGETS]
                + ["lower_y_min_6", "risk"]
            ],
        }
        return payload, frames


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else round(float(v), 4)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        elif isinstance(v, pd.DataFrame):
            continue
        else:
            out[k] = v
    return out


def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or not len(df):
        return []
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out.replace({np.nan: None}).to_dict(orient="records")


def _forecast_accuracy(origins: pd.DataFrame) -> dict:
    out = {}
    pairs = [
        ("y_cart_3", "cart_at_20min"),
        ("y_cart_6", "cart_at_40min"),
        ("y_min_3", "min_within_20min"),
        ("y_min_6", "min_within_40min"),
    ]
    for target, label in pairs:
        if target not in origins:
            continue
        m = origins[target].notna()
        if m.sum() == 0:
            continue
        err = origins.loc[m, f"pred_{target}"] - origins.loc[m, target]
        out[f"mae_{label}"] = round(float(err.abs().mean()), 4)
        out[f"rmse_{label}"] = round(float(np.sqrt((err**2).mean())), 4)
        out[f"bias_{label}"] = round(float(err.mean()), 4)
    m6 = origins["y_min_6"].notna() if "y_min_6" in origins else None
    if m6 is not None and m6.sum() > 0:
        covered = origins.loc[m6, "y_min_6"] >= origins.loc[m6, "lower_y_min_6"]
        out["lower_bound_coverage"] = round(float(covered.mean()), 4)
        out["n_scored"] = int(m6.sum())
    return out
