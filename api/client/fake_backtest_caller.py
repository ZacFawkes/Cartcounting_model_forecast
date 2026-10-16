"""Fake backtest caller: submits telemetry windows to /backtest.

Runs the holdout window under several policy and assumption scenarios so the
sensitivity of the simulated benefit is visible from the API itself, exactly as
reported by the simulation audit.

Usage
-----
    python client/fake_backtest_caller.py
    python client/fake_backtest_caller.py --scenario base_case
    python client/fake_backtest_caller.py --send-records     # post raw telemetry
"""
from __future__ import annotations

import argparse
import sys
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

HOLDOUT_START = "2026-10-10 00:00"
HOLDOUT_END = "2026-10-12 23:59"

# Mirrors the audit scenarios so API results and the audit report agree.
SCENARIOS: dict[str, dict] = {
    "audited_default": {
        "label": "Audited re-optimised policy, base-case assumptions",
        "mode": "replay",
        "policy": {},
        "inventory": {},
    },
    "pre_audit_policy": {
        "label": "Original pre-audit policy (risk 0.40, cooldown 10 min)",
        "mode": "replay",
        "policy": {"risk_threshold": 0.40, "cooldown_min": 10.0},
        "inventory": {},
    },
    "closed_loop": {
        "label": "Audited policy, closed-loop simulation",
        "mode": "closed_loop",
        "policy": {},
        "inventory": {},
    },
    "best_case": {
        "label": "Best case: 5-min lead, unlimited reserve, persistent surplus",
        "mode": "replay",
        "policy": {"lead_time_min": 5.0},
        "inventory": {"usage": "LOW", "reserve": "UNLIMITED",
                      "consumption_model": "persistent"},
    },
    "conservative": {
        "label": "Conservative: 20-min lead, delayed reserve, full turnover",
        "mode": "replay",
        "policy": {"lead_time_min": 20.0},
        "inventory": {"usage": "HIGH", "reserve": "DELAYED",
                      "consumption_model": "turnover"},
    },
    "low_workload": {
        "label": "Workload-light policy (risk 0.70, 45-min cooldown)",
        "mode": "replay",
        "policy": {"risk_threshold": 0.70, "cooldown_min": 45.0},
        "inventory": {},
    },
}


def run(
    base_url: str = DEFAULT_BASE_URL,
    scenarios: list[str] | None = None,
    start: str = HOLDOUT_START,
    end: str = HOLDOUT_END,
    send_records: bool = False,
) -> pd.DataFrame:
    configure_logging()
    client = CartApiClient(base_url, name="fake_backtest_caller")
    if not client.wait_until_ready():
        raise SystemExit("Server is not ready — start server/run_server.sh first")

    names = scenarios or list(SCENARIOS)
    records = None
    if send_records:
        df = slice_window(load_telemetry(), start, end)
        records = to_records(df)
        LOGGER.info("Posting %d telemetry records per request", len(records))

    rows = []
    for name in names:
        spec = SCENARIOS[name]
        payload = {
            "mode": spec["mode"],
            "policy": spec["policy"],
            "inventory": spec["inventory"],
            "label": spec["label"],
            "include_decisions": True,
            "include_events": True,
            "client": "fake_backtest_caller",
        }
        if records is not None:
            payload["records"] = records
        else:
            payload.update({"use_bundled_data": True, "start": start, "end": end})

        LOGGER.info("POST /backtest  [%s]", name)
        resp = client.backtest(payload)
        m = resp["metrics"]
        acc = resp["forecast_accuracy"]
        att = resp.get("attribution_summary", {})
        rows.append(
            {
                "scenario": name,
                "label": spec["label"],
                "run_id": resp["run_id"],
                "mode": resp["mode"],
                "records": resp["n_records"],
                "scored_origins": resp["scored_origins"],
                "baseline_events": m.get("baseline_events"),
                "policy_events": m.get("policy_events"),
                "event_reduction_pct": m.get("event_reduction_pct"),
                "duration_reduction_pct": m.get("duration_reduction_pct"),
                "severity_reduction_pct": m.get("severity_reduction_pct"),
                "actions": m.get("n_replenishment_actions"),
                "carts_moved": m.get("total_carts_recommended"),
                "trips_per_day": m.get("trips_per_day"),
                "policy_cost_J": m.get("policy_cost_J"),
                "baseline_cost_J": m.get("baseline_cost_J"),
                "mae_min_40min": acc.get("mae_min_within_40min"),
                "mae_cart_20min": acc.get("mae_cart_at_20min"),
                "lower_bound_coverage": acc.get("lower_bound_coverage"),
                "prevented_onset": att.get("prevented_onset"),
                "reduced_duration_only": att.get("reduced_duration_only"),
                "conserved": m.get("ledger_conserved"),
                "artifact_dir": resp["artifact_dir"],
            }
        )
        LOGGER.info(
            "  -> %s events %s->%s | severity -%.1f%% | %s trips, %s carts | run %s",
            name, m.get("baseline_events"), m.get("policy_events"),
            m.get("severity_reduction_pct") or 0.0, m.get("n_replenishment_actions"),
            m.get("total_carts_recommended"), resp["run_id"],
        )

    out = pd.DataFrame(rows)
    path = CLIENT_LOG_DIR / "fake_backtest_results.csv"
    out.to_csv(path, index=False)

    LOGGER.info("-" * 78)
    cols = ["scenario", "baseline_events", "policy_events", "severity_reduction_pct",
            "actions", "carts_moved", "mae_min_40min"]
    LOGGER.info("\n%s", out[cols].to_string(index=False))
    LOGGER.info("Client results written to %s", path)
    LOGGER.info("Server-side log: api/logs/backtest_results.csv (+ logs/runs/<run_id>/)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Submit backtest windows to the API")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--scenario", action="append", choices=list(SCENARIOS),
                   help="Repeatable; defaults to every scenario")
    p.add_argument("--start", default=HOLDOUT_START)
    p.add_argument("--end", default=HOLDOUT_END)
    p.add_argument("--send-records", action="store_true",
                   help="POST raw telemetry instead of using the bundled dataset")
    args = p.parse_args()
    run(args.base_url, args.scenario, args.start, args.end, args.send_records)


if __name__ == "__main__":
    main()
