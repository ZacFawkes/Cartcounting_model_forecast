"""System-level defaults — resolution level 3 (lowest precedence).

Level 1 = API request override, level 2 = entity/site config, level 3 = here.
A value falling through to this file is recorded in every response as
``"source": "system_default"`` so a fallback is never mistaken for a decision.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLATFORM_ROOT.parent
PIPELINE_ROOT = PROJECT_ROOT / "cart_forecasting"

for _p in (PIPELINE_ROOT, PIPELINE_ROOT / "src", PLATFORM_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_DIR = PLATFORM_ROOT / "data"
UPLOAD_DIR = Path(os.getenv("CART_PLATFORM_UPLOAD_DIR", DATA_DIR / "uploads"))
ARTIFACT_DIR = Path(os.getenv("CART_PLATFORM_ARTIFACT_DIR", PLATFORM_ROOT / "logs"))
LOG_DIR = Path(os.getenv("CART_PLATFORM_LOG_DIR", PLATFORM_ROOT / "logs"))
RUN_DIR = LOG_DIR / "runs"
MODEL_DIR = PLATFORM_ROOT / "models"
DB_PATH = Path(os.getenv("CART_PLATFORM_DB", DATA_DIR / "platform.db"))
SERVER_LOG = LOG_DIR / "server.log"

SCORE_ARTIFACT_DIR = LOG_DIR / "scores"
TRAINING_ARTIFACT_DIR = LOG_DIR / "training"
PER_ENTITY_MODEL_DIR = MODEL_DIR / "per_entity"

for _d in (DATA_DIR, LOG_DIR, RUN_DIR, MODEL_DIR, UPLOAD_DIR,
           SCORE_ARTIFACT_DIR, TRAINING_ARTIFACT_DIR, PER_ENTITY_MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Source model bundle produced by cart_forecasting/main.py
SOURCE_MODEL_BUNDLE = PIPELINE_ROOT / "outputs" / "models" / "cart_forecasting_models.pkl"
BUNDLED_TELEMETRY = PIPELINE_ROOT / "data" / "real_data.csv"

HOST = os.getenv("CART_PLATFORM_HOST", "127.0.0.1")
PORT = int(os.getenv("CART_PLATFORM_PORT", "8100"))

API_TITLE = "Cart Forecasting Platform"
API_VERSION = "2.0.0"

# D9b — set True when DVR ids are only unique within a store.
# Entity identity includes the site: "163:215:0". DVR ids are not guaranteed
# unique across stores, so the site is part of the key, not an attribute of it.
# Set CART_ENTITY_INCLUDE_SITE=0 only for a single-site deployment.
INCLUDE_SITE_IN_ENTITY_ID = os.getenv("CART_ENTITY_INCLUDE_SITE", "1") == "1"

# D9c — one channel can watch more than one physical cart bay, so the bay is
# part of the identity: "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8".
# The dataset must then carry the bay column (data_loader.BAY_COL, "BayID"),
# which column_mapping can map from any source column name.
# Set CART_ENTITY_INCLUDE_BAY=0 for a deployment with one bay per channel.
INCLUDE_BAY_IN_ENTITY_ID = os.getenv("CART_ENTITY_INCLUDE_BAY", "1") == "1"

# Feature window: 10 records is the hard floor (cart_lag_9, cart_momentum_9,
# cart_diff_volatility_9 all reach back to t-9). 12 is enforced for margin.
MIN_HISTORY_RECORDS = int(os.getenv("CART_MIN_HISTORY", "12"))
ABSOLUTE_MIN_HISTORY = 10
RECOMMENDED_HISTORY = 20

# Entity lifecycle
MIN_DAYS_FOR_ACTIVE = 14          # below this an entity stays in shadow (D7)

# ---------------------------------------------------------------------------
# Level-3 parameter defaults
# ---------------------------------------------------------------------------
SYSTEM_DEFAULTS: dict = {
    # thresholds (evaluation / policy side — see red_model vs red_eval, D10)
    "red_threshold": 15,
    "amber_threshold": 20,
    "target_stock": 25,
    # policy
    # NOTE: risk_threshold and critical_risk_threshold are deliberately absent.
    # They are clustered from the entity's own calibrated risk at training time
    # and frozen into the model — see training/alert_bands.py.
    "lead_time_min": 10.0,
    "cooldown_min": 20.0,
    "min_carts_per_trip": 3,
    "max_carts_per_trip": 20,
    "quantity_rule": "point",
    # physical
    "bay_capacity": 60,
    "reserve_per_day": 60,
    # simulation assumptions (backtest only)
    "consumption_model": "unmet_demand",
    "usage": "MEDIUM",
    "retention_halflife_min": 45.0,
    # data handling
    "max_sequence_gap_min": 15.0,
}

# Which level-2 table owns each parameter. Entity values always win over site.
SITE_LEVEL_PARAMS = {
    "lead_time_min", "reserve_per_day", "min_carts_per_trip",
    "max_carts_per_trip", "cooldown_min",
}
ENTITY_LEVEL_PARAMS = {
    "red_threshold", "amber_threshold", "target_stock", "bay_capacity",
    "quantity_rule", "consumption_model", "usage", "retention_halflife_min",
    "max_sequence_gap_min",
} | SITE_LEVEL_PARAMS

# Parameters a request may override freely.
REQUEST_OVERRIDABLE = set(SYSTEM_DEFAULTS)

# Parameters that also shape the FEATURE VECTOR and therefore cannot be changed
# at request time without invalidating the model (decision D10).
FEATURE_COUPLED_PARAMS = {"red_threshold", "amber_threshold"}

# Threshold derivation when no business value is supplied (D3).
THRESHOLD_PERCENTILES = {"red": 10, "amber": 25, "target": 60}
BAY_CAPACITY_MARGIN = 1.05

# ---------------------------------------------------------------------------
# Alert bands discovered by K-Means on the calibrated risk distribution
# ---------------------------------------------------------------------------
# NORMAL < band_warning <= WARNING < band_critical <= CRITICAL.
# band_warning also gates replenishment, so alerting and ordering share one
# measured operating point instead of two hand-set constants.
ALERT_BAND_SEED = 0               # K-Means is deterministic for a fixed seed

# Guards against a degenerate clustering. Measured: on a short window with no
# high-risk records K-Means returns bands at 0.006 / 0.042, which would mark
# nearly every record elevated.
ALERT_BAND_MIN_SAMPLES = 100      # calibration records needed to band at all
ALERT_BAND_MIN_EVENTS = 10        # records at or above EVENT_RISK_FLOOR
EVENT_RISK_FLOOR = 0.10           # what counts as a non-trivial risk
ALERT_BAND_MIN_WARNING = 0.02     # a lower band than this is rejected

# Used only when banding is impossible; the source is reported as
# "system_default" so it is never mistaken for a measured value.
DEFAULT_BAND_WARNING = 0.55
DEFAULT_BAND_CRITICAL = 0.80


# Cost coefficients for the operational objective J.
COST_LAMBDAS = {"shortage": 10.0, "trip": 5.0, "cart": 0.25, "excess": 0.10}

CONFORMAL_ALPHA = 0.10

CAVEAT = (
    "Forecast accuracy is measured on held-out telemetry. Shortage-reduction "
    "figures are simulator output under documented assumptions, not observed "
    "causal effects; the audited range across defensible assumptions is wide."
)


# ---------------------------------------------------------------------------
# On-demand upload / training module
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("CART_MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
ALLOWED_UPLOAD_SUFFIXES = {".csv", ".tsv", ".txt"}

REQUIRED_DATASET_COLUMNS = [
    "KDVR", "ChanNo", "DVRDate", "next_DVRDate", "Count", "Count_IN", "Count_OUT",
]

# Minimum data for a trustworthy per-entity model. Below these an entity is
# reported INSUFFICIENT_TRAINING_DATA and stays shadow — never fabricated.
MIN_TRAIN_RECORDS = int(os.getenv("CART_MIN_TRAIN_RECORDS", "200"))
MIN_TRAIN_DAYS = int(os.getenv("CART_MIN_TRAIN_DAYS", "2"))
MIN_HOLDOUT_RECORDS = int(os.getenv("CART_MIN_HOLDOUT_RECORDS", "40"))
MIN_CALIBRATION_SAMPLES = int(os.getenv("CART_MIN_CALIBRATION_SAMPLES", "500"))
MIN_CALIBRATION_POSITIVES = int(os.getenv("CART_MIN_CALIBRATION_POSITIVES", "20"))

# Chronological split fractions for an upload-scoped training run.
TRAIN_FRACTION = float(os.getenv("CART_TRAIN_FRACTION", "0.6"))
CALIBRATION_FRACTION = float(os.getenv("CART_CALIBRATION_FRACTION", "0.2"))
# Remainder is the final holdout used only by the three backtests.

BACKTEST_TYPES = ("forecast_accuracy", "policy_replay", "closed_loop_robustness")

# ---------------------------------------------------------------------------
# API authentication (applies to upload/train/promote/score/artifact routes)
# ---------------------------------------------------------------------------
API_TOKEN = os.getenv("CART_PLATFORM_API_TOKEN")  # None => auth disabled (dev)
AUTH_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc",
                     "/docs/oauth2-redirect"}

# ---------------------------------------------------------------------------
# P7 — standalone result receiver
# ---------------------------------------------------------------------------
RECEIVER_DB_PATH = Path(os.getenv("CART_RECEIVER_DB", DATA_DIR / "results_receiver.db"))
RECEIVER_HOST = os.getenv("CART_RECEIVER_HOST", "127.0.0.1")
RECEIVER_PORT = int(os.getenv("CART_RECEIVER_PORT", "8200"))
RECEIVER_TOKEN = os.getenv("CART_RESULT_RECEIVER_TOKEN")

RESULT_RECEIVER_URL = os.getenv("CART_RESULT_RECEIVER_URL", "http://127.0.0.1:8200")
RESULT_RECEIVER_ENABLED = os.getenv("CART_RESULT_RECEIVER_ENABLED", "1") == "1"
PUBLISH_TIMEOUT_SEC = float(os.getenv("CART_RESULT_PUBLISH_TIMEOUT_SEC", "10"))
PUBLISH_MAX_ATTEMPTS = int(os.getenv("CART_RESULT_PUBLISH_MAX_ATTEMPTS", "8"))
PUBLISH_BASE_BACKOFF_SEC = float(os.getenv("CART_RESULT_PUBLISH_BACKOFF_SEC", "2"))
PUBLISH_MAX_BACKOFF_SEC = float(os.getenv("CART_RESULT_PUBLISH_MAX_BACKOFF_SEC", "300"))
PUBLISH_WORKER_INTERVAL_SEC = float(os.getenv("CART_RESULT_WORKER_INTERVAL_SEC", "5"))
EVENT_SCHEMA_VERSION = 1
