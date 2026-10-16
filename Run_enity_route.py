"""Full 215:0 run through the canonical entity route.

POST /v1/entities/215/0/score — writes one immutable CSV per decision and
publishes each result to the standalone receiver via the outbox.
"""
import sys
import time
from pathlib import Path

ROOT = Path("/Users/Zac/PycharmProjects/Cartcounting_model_forecast/cart_platform")
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from client.platform_client import (
    CLIENT_LOG_DIR,
    LOGGER,
    configure_logging,
    load_telemetry,
    to_records,
)

DATASET = '/Users/Zac/PycharmProjects/Cartcounting_model_forecast/cart_platform/data/real_data.csv'
BASE = "http://127.0.0.1:8100"
ENTITY = "215:0"
MODEL_ID = "mdl_215_0_20261013113343_8b835e"
WINDOW = 20


def main() -> None:
    configure_logging()
    df = load_telemetry(DATASET)
    df = df[df["entity_id"] == ENTITY].reset_index(drop=True)
    LOGGER.info("Entity route run | %s | %d records | model %s",
                ENTITY, len(df), MODEL_ID)

    sess = requests.Session()
    rows, t0 = [], time.perf_counter()

    for i in range(WINDOW, len(df)):
        payload = {
            "records": to_records(df.iloc[i - WINDOW + 1 : i + 1]),
            "write_to_db": True,
            "client": "entity_route_runner",
        }
        r = sess.post(f"{BASE}/v1/entities/215/0/score", json=payload, timeout=180)
        r.raise_for_status()
        resp = r.json()
        f, a = resp["forecast"], resp["action"]
        rows.append({
            "decision_time": resp["decision_time"],
            "request_id": resp["request_id"],
            "model_id": resp["model_id"],
            "current_cart": resp["current_cart"],
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
            "artifact_path": resp.get("artifact_path"),
            "persisted": resp.get("persisted"),
            "idempotent_replay": resp.get("idempotent_replay"),
            "reason": a["reason"],
        })
        if a["should_replenish"]:
            LOGGER.info("%s | %s cart=%.0f risk=%.2f -> RETRIEVE %d by %s [%s]",
                        ENTITY, resp["decision_time"][:16], resp["current_cart"],
                        resp["shortage_risk"], a["quantity"],
                        (a["expected_completion_time"] or "")[11:16], a["urgency"])

    out = pd.DataFrame(rows)
    path = CLIENT_LOG_DIR / "entity_route_215_0_all.csv"
    out.to_csv(path, index=False)

    LOGGER.info("-" * 74)
    LOGGER.info("Calls %d in %.1fs | orders %d | carts %d | alerts %d | below RED %d",
                len(out), time.perf_counter() - t0,
                int(out["should_replenish"].sum()),
                int(out.loc[out["should_replenish"], "quantity"].sum()),
                int(out["new_alert"].sum()), int(out["reactive_critical"].sum()))
    LOGGER.info("Artifacts written: %d | replays: %d",
                int(out["artifact_path"].notna().sum()),
                int(out["idempotent_replay"].fillna(False).sum()))
    LOGGER.info("Models used: %s", out["model_id"].unique().tolist())
    LOGGER.info("Client results: %s", path)


if __name__ == "__main__":
    main()
