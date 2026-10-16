"""Server configuration and shared paths.

The API serves the AUDITED decision engine. The forecasting models come from
``cart_forecasting/outputs/models`` (validated by the audit); the replenishment
policy defaults come from the audited re-optimised policy, not from the
original simulator that the audit found defective.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = API_ROOT.parent
PIPELINE_ROOT = PROJECT_ROOT / "cart_forecasting"

# Make the modelling package importable.
for _p in (PIPELINE_ROOT, PIPELINE_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MODEL_FILE = PIPELINE_ROOT / "outputs" / "models" / "cart_forecasting_models.pkl"
AUDIT_SUMMARY = PIPELINE_ROOT / "simulation_audit" / "audit_summary.json"
BUNDLED_DATA = PIPELINE_ROOT / "data" / "real_data.csv"

LOG_DIR = Path(os.getenv("CART_API_LOG_DIR", API_ROOT / "logs"))
RUN_DIR = LOG_DIR / "runs"
REQUEST_LOG = LOG_DIR / "requests.jsonl"
SCORE_LOG = LOG_DIR / "score_results.csv"
BACKTEST_LOG = LOG_DIR / "backtest_results.csv"
ERROR_LOG = LOG_DIR / "errors.jsonl"
SERVER_LOG = LOG_DIR / "server.log"

for _d in (LOG_DIR, RUN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HOST = os.getenv("CART_API_HOST", "127.0.0.1")
PORT = int(os.getenv("CART_API_PORT", "8000"))

# Minimum records of history required before features can be built.
MIN_HISTORY_RECORDS = 12

API_VERSION = "1.0.0"
API_TITLE = "Cart Availability Forecasting & Replenishment API"

# Audited re-optimised policy (frozen on validation data).
DEFAULT_POLICY = {
    "red_threshold": 15.0,
    "amber_threshold": 20.0,
    "target_stock": 28.0,
    "risk_threshold": 0.55,
    "lead_time_min": 10.0,
    "cooldown_min": 30.0,
    "min_carts_per_trip": 3,
    "max_carts_per_trip": 20,
    "quantity_rule": "point",
    "planning_horizon": 6,
}

# Audited base-case inventory assumptions.
DEFAULT_INVENTORY = {
    "bay_capacity": 60.0,
    "usage": "MEDIUM",
    "reserve": "LIMITED",
    "consumption_model": "unmet_demand",
    "retention_halflife_min": 45.0,
}
