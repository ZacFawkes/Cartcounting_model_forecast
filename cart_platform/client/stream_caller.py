"""Fake streaming caller: pushes each new telemetry record to /v1/score.

Simulates the collector that fires whenever a new log row lands in the
database. Each call sends only the history available at that moment, so the
stream is a faithful rehearsal of online operation.

Usage
-----
    python client/stream_caller.py --entity 215:0 --window 20
    python client/stream_caller.py --entity 215:0 --start "2026-10-11 08:00" \
                                   --end "2026-10-11 20:00"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from client.platform_client import (  # noqa: E402
    CLIENT_LOG_DIR,
    DEFAULT_BASE_URL,
    LOGGER,
    PlatformClient,
    configure_logging,
    load_telemetry,
    slice_window,
    to_records,
)

HISTORY_WINDOW = 20  # 10 is the hard floor; 20 gives margin against gaps


def run(
    base_url: str = DEFAULT_BASE_URL,
    entity_id: str = "215:0",
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    window: int = HISTORY_WINDOW,
    delay: float = 0.0,
    params: dict | None = None,
    reset: bool = True,
) -> pd.DataFrame:
    configure_logging()
    client = PlatformClient(base_url, name="stream_caller")
    if not client.wait_until_ready():
        raise SystemExit("Server not ready — start server/run_server.sh first")

    df = load_telemetry()
    df = df[df["entity_id"] == entity_id]
    if df.empty:
        raise SystemExit(f"no telemetry for entity {entity_id}")
    df = slice_window(df, start, end)
    if limit:
        df = df.head(limit + window)
    if len(df) <= window:
        raise SystemExit(f"need more than {window} records to stream")

    if reset:
        client.reset(entity_id)

    rows = []
    LOGGER.info("Streaming %d records for %s (history window %d)",
                len(df) - window, entity_id, window)

    for i in range(window, len(df)):
        payload = {
            "entity_id": entity_id,
            "records": to_records(df.iloc[i - window + 1 : i + 1]),
            "write_to_db": True,
            "client": "stream_caller",
        }
        if params:
            payload["params"] = params

        resp = client.score(payload)
        f, a = resp["forecast"], resp["action"]
        rows.append({
            "entity_id": resp["entity_id"],
            "decision_time": resp["decision_time"],
            "request_id": resp["request_id"],
            "current_cart": resp["current_cart"],
            "simulated_cart": resp["simulated_cart"],
            "pred_cart_20min": f["cart_at_20min"],
            "pred_min_40min": f["min_within_40min"],
            "lower_bound_40min": f["lower_bound_min_within_40min"],
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
            "persisted": resp.get("persisted"),
            "reason": a["reason"],
        })
        if a["should_replenish"]:
            LOGGER.info("%s | %s cart=%.0f risk=%.2f -> RETRIEVE %d by %s [%s]",
                        entity_id, resp["decision_time"][:16], resp["current_cart"],
                        resp["shortage_risk"], a["quantity"],
                        (a["expected_completion_time"] or "")[11:16], a["urgency"])
        elif resp["new_predictive_alert"]:
            LOGGER.info("%s | %s cart=%.0f risk=%.2f -> %s alert",
                        entity_id, resp["decision_time"][:16], resp["current_cart"],
                        resp["shortage_risk"], resp["alert_state"])
        if delay:
            time.sleep(delay)

    out = pd.DataFrame(rows)
    path = CLIENT_LOG_DIR / f"stream_{entity_id.replace(':', '_')}.csv"
    out.to_csv(path, index=False)

    LOGGER.info("-" * 74)
    LOGGER.info("Calls %d | orders %d | carts %d | alerts %d | below RED %d",
                len(out), int(out["should_replenish"].sum()),
                int(out.loc[out["should_replenish"], "quantity"].sum()),
                int(out["new_alert"].sum()), int(out["reactive_critical"].sum()))
    LOGGER.info("Client results: %s", path)
    LOGGER.info("Server write-back: scoring_results table (GET /v1/logs/scores)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Stream telemetry to /v1/score")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--entity", default="215:0")
    p.add_argument("--start", default="2026-10-11 08:00")
    p.add_argument("--end", default="2026-10-11 20:00")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--window", type=int, default=HISTORY_WINDOW)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--no-reset", action="store_true")
    a = p.parse_args()
    run(a.base_url, a.entity, a.start, a.end, a.limit, a.window, a.delay,
        reset=not a.no_reset)


if __name__ == "__main__":
    main()
