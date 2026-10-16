"""Fake backtest caller: submits windows to /v1/backtest under several scenarios.

Mirrors the simulation-audit scenarios so API output and the audit report agree,
and makes the assumption-sensitivity visible from the API itself.

Usage
-----
    python client/backtest_caller.py --entity 215:0
    python client/backtest_caller.py --site 163 --scenario audited_default
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from client.platform_client import (  # noqa: E402
    CLIENT_LOG_DIR,
    DEFAULT_BASE_URL,
    LOGGER,
    PlatformClient,
    configure_logging,
)

HOLDOUT_START = "2026-10-10 00:00"
HOLDOUT_END = "2026-10-12 23:59"

# Audited policy, expressed as explicit overrides so a run is reproducible even
# if an entity's registry defaults change later.
AUDITED = {
    "red_threshold": 15, "amber_threshold": 20, "target_stock": 28,
    "cooldown_min": 30.0, "lead_time_min": 10.0,
}

SCENARIOS: dict[str, dict] = {
    "entity_defaults": {
        "label": "Entity registry defaults (no overrides)",
        "mode": "replay", "params": {},
    },
    "audited_default": {
        "label": "Audited re-optimised policy, base-case assumptions",
        "mode": "replay", "params": AUDITED,
    },
    "pre_audit_policy": {
        "label": "Original pre-audit policy (risk 0.40, cooldown 10 min)",
        "mode": "replay",
        "params": {**AUDITED, "cooldown_min": 10.0},
    },
    "closed_loop": {
        "label": "Audited policy, closed-loop simulation",
        "mode": "closed_loop", "params": AUDITED,
    },
    "best_case": {
        "label": "Best case: 5-min lead, persistent surplus, low usage",
        "mode": "replay",
        "params": {**AUDITED, "lead_time_min": 5.0, "usage": "LOW",
                   "consumption_model": "persistent"},
    },
    "conservative": {
        "label": "Conservative: 20-min lead, full turnover, high usage",
        "mode": "replay",
        "params": {**AUDITED, "lead_time_min": 20.0, "usage": "HIGH",
                   "consumption_model": "turnover"},
    },
}


def run(
    base_url: str = DEFAULT_BASE_URL,
    entity_id: str | None = "215:0",
    site_key: int | None = None,
    scenarios: list[str] | None = None,
    start: str = HOLDOUT_START,
    end: str = HOLDOUT_END,
) -> pd.DataFrame:
    configure_logging()
    client = PlatformClient(base_url, name="backtest_caller")
    if not client.wait_until_ready():
        raise SystemExit("Server not ready — start server/run_server.sh first")

    rows = []
    for name in scenarios or list(SCENARIOS):
        spec = SCENARIOS[name]
        payload = {
            "start": start, "end": end, "mode": spec["mode"],
            "params": spec["params"], "label": spec["label"],
            "client": "backtest_caller",
        }
        if site_key is not None:
            payload["site_key"] = site_key
        else:
            payload["entity_id"] = entity_id

        LOGGER.info("POST /v1/backtest [%s]", name)
        resp = client.backtest(payload)
        for run_ in resp["runs"]:
            m, acc = run_["metrics"], run_["forecast_accuracy"]
            att = run_.get("attribution_summary", {})
            rows.append({
                "scenario": name, "label": spec["label"],
                "entity_id": run_["entity_id"], "run_id": run_["run_id"],
                "mode": run_["mode"], "records": run_["n_records"],
                "baseline_events": m.get("baseline_events"),
                "policy_events": m.get("policy_events"),
                "event_reduction_pct": m.get("event_reduction_pct"),
                "duration_reduction_pct": m.get("duration_reduction_pct"),
                "severity_reduction_pct": m.get("severity_reduction_pct"),
                "actions": m.get("n_replenishment_actions"),
                "carts_moved": m.get("total_carts_recommended"),
                "trips_per_day": m.get("trips_per_day"),
                "policy_cost_J": m.get("policy_cost_J"),
                "mae_min_40min": acc.get("mae_min_within_40min"),
                "lower_bound_coverage": acc.get("lower_bound_coverage"),
                "prevented_onset": att.get("prevented_onset"),
                "reduced_duration_only": att.get("reduced_duration_only"),
                "conserved": m.get("ledger_conserved"),
                "n_warnings": len(run_.get("warnings", [])),
            })
            LOGGER.info("  %s %s: events %s->%s | severity -%.1f%% | %s trips, %s carts",
                        name, run_["entity_id"], m.get("baseline_events"),
                        m.get("policy_events"), m.get("severity_reduction_pct") or 0.0,
                        m.get("n_replenishment_actions"), m.get("total_carts_recommended"))

    out = pd.DataFrame(rows)
    path = CLIENT_LOG_DIR / "backtest_results.csv"
    out.to_csv(path, index=False)
    LOGGER.info("-" * 92)
    cols = ["scenario", "entity_id", "baseline_events", "policy_events",
            "severity_reduction_pct", "actions", "carts_moved", "mae_min_40min"]
    LOGGER.info("\n%s", out[cols].to_string(index=False))
    LOGGER.info("Client results: %s", path)
    LOGGER.info("Server-side: backtest_runs table + logs/runs/<run_id>/")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Submit backtests to the platform")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--entity", default="215:0")
    p.add_argument("--site", type=int, default=None)
    p.add_argument("--scenario", action="append", choices=list(SCENARIOS))
    p.add_argument("--start", default=HOLDOUT_START)
    p.add_argument("--end", default=HOLDOUT_END)
    a = p.parse_args()
    run(a.base_url, a.entity, a.site, a.scenario, a.start, a.end)


if __name__ == "__main__":
    main()
