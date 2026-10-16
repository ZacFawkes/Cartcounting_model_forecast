"""Fake live caller: replays historical telemetry to /score record by record.

Simulates the store controller that would call the API roughly every 6-8
minutes as each telemetry record completes. Each call sends only the history
available at that moment — never a future record — so the stream is a faithful
rehearsal of online operation.

Usage
-----
    python client/fake_score_caller.py --limit 120
    python client/fake_score_caller.py --start "2026-10-11 09:00" --end "2026-10-11 20:00"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from client.api_client import (  # noqa: E402
    CLIENT_LOG_DIR,
    DEFAULT_BASE_URL,
    LOGGER,
    CartApiClient,
    configure_logging,
    load_telemetry,
    slice_window,
    to_records,
)

HISTORY_WINDOW = 20  # records sent with every call


def run(
    base_url: str = DEFAULT_BASE_URL,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    history_window: int = HISTORY_WINDOW,
    delay: float = 0.0,
    session_id: str = "fake-live-stream",
    policy: dict | None = None,
    reset: bool = True,
) -> pd.DataFrame:
    configure_logging()
    client = CartApiClient(base_url, name="fake_score_caller")
    if not client.wait_until_ready():
        raise SystemExit("Server is not ready — start server/run_server.sh first")

    df = slice_window(load_telemetry(), start, end)
    if limit:
        df = df.head(limit + history_window)
    if len(df) <= history_window:
        raise SystemExit(f"Need more than {history_window} records to stream")

    if reset:
        client.reset_sessions(session_id)

    rows = []
    n_calls = 0
    LOGGER.info("Streaming %d records to /score (history window %d)",
                len(df) - history_window, history_window)

    for i in range(history_window, len(df)):
        history = df.iloc[i - history_window + 1 : i + 1]
        payload = {
            "site_key": int(history["siteKey"].iloc[-1]),
            "history": to_records(history),
            "session_id": session_id,
            "client": "fake_score_caller",
        }
        if policy:
            payload["policy"] = policy

        resp = client.score(payload)
        n_calls += 1
        f = resp["forecast"]
        a = resp["action"]
        rows.append(
            {
                "decision_time": resp["decision_time"],
                "request_id": resp["request_id"],
                "current_cart": resp["current_cart"],
                "simulated_cart": resp["simulated_cart"],
                "pred_cart_20min": f["cart_at_20min"],
                "pred_cart_40min": f["cart_at_40min"],
                "pred_min_40min": f["min_within_40min"],
                "lower_bound_min_40min": f["lower_bound_min_within_40min"],
                "risk": resp["shortage_risk"],
                "alert_state": resp["alert_state"],
                "new_alert": resp["new_predictive_alert"],
                "reactive_critical": resp["reactive_critical"],
                "should_replenish": a["should_replenish"],
                "quantity": a["quantity"],
                "urgency": a["urgency"],
                "order_id": a["order_id"],
                "deliver_by": a["expected_completion_time"],
                "pending_carts": resp["pending_carts"],
                "reason": a["reason"],
            }
        )
        if a["should_replenish"]:
            LOGGER.info(
                "%s | cart=%.0f risk=%.2f -> RETRIEVE %d carts by %s [%s]",
                resp["decision_time"][:16], resp["current_cart"], resp["shortage_risk"],
                a["quantity"], (a["expected_completion_time"] or "")[11:16], a["urgency"],
            )
        elif resp["new_predictive_alert"]:
            LOGGER.info("%s | cart=%.0f risk=%.2f -> %s alert",
                        resp["decision_time"][:16], resp["current_cart"],
                        resp["shortage_risk"], resp["alert_state"])
        if delay:
            time.sleep(delay)

    out = pd.DataFrame(rows)
    path = CLIENT_LOG_DIR / "fake_score_stream.csv"
    out.to_csv(path, index=False)

    n_orders = int(out["should_replenish"].sum())
    LOGGER.info("-" * 72)
    LOGGER.info("Calls: %d | replenishment orders: %d | carts: %d | alerts: %d",
                n_calls, n_orders, int(out.loc[out['should_replenish'], 'quantity'].sum()),
                int(out["new_alert"].sum()))
    LOGGER.info("Records observed below RED: %d", int(out["reactive_critical"].sum()))
    LOGGER.info("Client results written to %s", path)
    LOGGER.info("Server-side log: api/logs/score_results.csv")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Replay telemetry to the /score endpoint")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--start", default="2026-10-11 08:00")
    p.add_argument("--end", default="2026-10-11 20:00")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--history-window", type=int, default=HISTORY_WINDOW)
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds to sleep between calls (use 0 for a fast replay)")
    p.add_argument("--session-id", default="fake-live-stream")
    p.add_argument("--no-reset", action="store_true")
    args = p.parse_args()
    run(
        base_url=args.base_url,
        start=args.start,
        end=args.end,
        limit=args.limit,
        history_window=args.history_window,
        delay=args.delay,
        session_id=args.session_id,
        reset=not args.no_reset,
    )


if __name__ == "__main__":
    main()
