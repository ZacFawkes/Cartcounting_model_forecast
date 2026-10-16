"""End-to-end demo: health -> model info -> backtests -> live stream -> logs.

Run the server first, then:

    python client/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import fake_backtest_caller, fake_score_caller  # noqa: E402
from client.api_client import (  # noqa: E402
    CLIENT_LOG_DIR,
    DEFAULT_BASE_URL,
    LOGGER,
    CartApiClient,
    configure_logging,
)


def main(base_url: str = DEFAULT_BASE_URL) -> None:
    configure_logging()
    client = CartApiClient(base_url, name="run_demo")

    LOGGER.info("=" * 78)
    LOGGER.info("STEP 1 — health check")
    if not client.wait_until_ready():
        raise SystemExit("Server not ready — start server/run_server.sh first")
    LOGGER.info("%s", client.health())

    LOGGER.info("=" * 78)
    LOGGER.info("STEP 2 — model info")
    info = client.model_info()
    LOGGER.info("model=%s features=%s", info["model_version"], info["n_features"])
    LOGGER.info("selected models: %s", info["selected_models"])
    LOGGER.info("served policy: %s", info["default_policy"])

    LOGGER.info("=" * 78)
    LOGGER.info("STEP 3 — backtest scenarios")
    bt = fake_backtest_caller.run(base_url)

    LOGGER.info("=" * 78)
    LOGGER.info("STEP 4 — live scoring stream")
    stream = fake_score_caller.run(base_url, start="2026-10-11 08:00",
                                   end="2026-10-11 20:00")

    LOGGER.info("=" * 78)
    LOGGER.info("STEP 5 — online session state")
    LOGGER.info("%s", client.sessions())

    LOGGER.info("=" * 78)
    LOGGER.info("STEP 6 — server-side logs")
    for kind in ("requests", "scores", "backtests"):
        entries = client.logs(kind, limit=5).get("entries", [])
        LOGGER.info("%-10s last %d entries recorded server-side", kind, len(entries))

    LOGGER.info("=" * 78)
    LOGGER.info("Demo complete.")
    LOGGER.info("  backtest scenarios run : %d", len(bt))
    LOGGER.info("  scoring calls made     : %d", len(stream))
    LOGGER.info("  replenishment orders   : %d", int(stream["should_replenish"].sum()))
    LOGGER.info("  client logs            : %s", CLIENT_LOG_DIR)
    LOGGER.info("  server logs            : api/logs/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL)
