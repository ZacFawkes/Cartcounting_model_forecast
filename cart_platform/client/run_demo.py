"""End-to-end walkthrough of the platform.

Start the server first, then:

    python client/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import backtest_caller, stream_caller  # noqa: E402
from client.platform_client import (  # noqa: E402
    CLIENT_LOG_DIR,
    DEFAULT_BASE_URL,
    LOGGER,
    PlatformClient,
    configure_logging,
)

BAR = "=" * 86


def main(base_url: str = DEFAULT_BASE_URL) -> None:
    configure_logging()
    client = PlatformClient(base_url, name="run_demo")

    LOGGER.info(BAR)
    LOGGER.info("STEP 1 — health and bootstrap")
    if not client.wait_until_ready():
        raise SystemExit("Server not ready — start server/run_server.sh first")
    LOGGER.info("%s", client.health())

    LOGGER.info(BAR)
    LOGGER.info("STEP 2 — model provenance")
    info = client.model_info()
    LOGGER.info("model %s v%s | %s features | grain %s",
                info["model_id"], info["version"], info["n_features"], info["entity_grain"])
    LOGGER.info("feature-time thresholds (frozen): red=%s amber=%s",
                info["feature_time_thresholds"]["red_model"],
                info["feature_time_thresholds"]["amber_model"])
    LOGGER.info("history: floor %s, enforced %s, recommended %s",
                info["history_requirements"]["absolute_minimum"],
                info["history_requirements"]["enforced_minimum"],
                info["history_requirements"]["recommended"])

    LOGGER.info(BAR)
    LOGGER.info("STEP 3 — entity registry")
    ents = client.entities()["entities"]
    for e in ents:
        LOGGER.info("  %s  site=%s  status=%-7s red=%s amber=%s target=%s cap=%s (%s)",
                    e["entity_id"], e["site_key"], e["status"], e["red_threshold"],
                    e["amber_threshold"], e["target_stock"], e["bay_capacity"],
                    e["threshold_source"])
    entity_id = ents[0]["entity_id"]

    LOGGER.info(BAR)
    LOGGER.info("STEP 4 — resolved configuration for %s", entity_id)
    detail = client.entity(entity_id)
    for k, v in sorted(detail["resolved_config"].items()):
        LOGGER.info("  %-24s %-14s <- %s", k, v["value"], v["source"])
    for w in detail["warnings"]:
        LOGGER.warning("  ! %s", w)

    LOGGER.info(BAR)
    LOGGER.info("STEP 5 — backtest scenarios")
    bt = backtest_caller.run(base_url, entity_id=entity_id)

    LOGGER.info(BAR)
    LOGGER.info("STEP 6 — live scoring stream")
    stream = stream_caller.run(base_url, entity_id=entity_id,
                               start="2026-10-11 08:00", end="2026-10-11 20:00",
                               params=backtest_caller.AUDITED)

    LOGGER.info(BAR)
    LOGGER.info("STEP 7 — durable state and order ledger")
    LOGGER.info("sessions: %s", client.sessions()["sessions"])
    orders = client.orders(entity_id=entity_id, limit=5)["orders"]
    for o in orders:
        LOGGER.info("  %s  %s  qty=%s  status=%s  executed_by=%s",
                    o["order_id"], o["decision_time"][:16], o["quantity"],
                    o["status"], o["executed_by"])

    if orders:
        LOGGER.info(BAR)
        LOGGER.info("STEP 8 — log what staff actually did (the audit's blocking gap)")
        done = client.execute_order(orders[0]["order_id"], "demo-staff",
                                    orders[0]["quantity"], "demo execution log")
        LOGGER.info("  %s executed_by=%s actual_carts=%s",
                    done["order"]["order_id"], done["order"]["executed_by"],
                    done["order"]["actual_carts"])

    LOGGER.info(BAR)
    LOGGER.info("STEP 9 — entity-keyed logs")
    for kind in ("requests", "scores", "backtests"):
        n = len(client.logs(kind, entity_id=entity_id, limit=100)["entries"])
        LOGGER.info("  %-10s %d entries for %s", kind, n, entity_id)

    LOGGER.info(BAR)
    LOGGER.info("Demo complete.")
    LOGGER.info("  backtest runs   : %d", len(bt))
    LOGGER.info("  scoring calls   : %d", len(stream))
    LOGGER.info("  orders issued   : %d", int(stream["should_replenish"].sum()))
    LOGGER.info("  client logs     : %s", CLIENT_LOG_DIR)
    LOGGER.info("  server database : cart_platform/data/platform.db")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL)
