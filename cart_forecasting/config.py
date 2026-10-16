"""Central configuration for the cart availability forecasting project.

Every operational constant is declared here so that assumptions are explicit,
auditable and easy to override in sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

RAW_DATA_FILE = DATA_DIR / "real_data.csv"
REPORT_FILE = OUTPUT_DIR / "Cart_Counting_Forecasting_Replenishment_Backtest_Report.docx"

for _d in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Data semantics
# --------------------------------------------------------------------------
CSV_SEPARATOR = "\t"
CSV_ENCODING = "utf-8-sig"

TIME_COL = "DVRDate"
NEXT_TIME_COL = "next_DVRDate"
CART_COL = "Count"
IN_COL = "Count_IN"
OUT_COL = "Count_OUT"

# A record is the atomic scoring unit. 3 records ~ 20 minutes (assumption).
RECORDS_PER_20MIN = 3
RECORDS_PER_40MIN = 6
RECORDS_PER_60MIN = 9
NOMINAL_RECORD_MINUTES = 20.0 / RECORDS_PER_20MIN

# Feature windows (in records)
FEATURE_WINDOWS = [3, 6, 9]

# A sequence break is declared when the gap between next_DVRDate and the
# following DVRDate exceeds this many minutes.
MAX_SEQUENCE_GAP_MIN = 15.0
# Horizons whose realised elapsed time deviates more than this ratio from the
# nominal horizon are flagged unreliable.
HORIZON_DEVIATION_TOLERANCE = 0.50
# Individual intervals longer than this are flagged as long-interval records.
LONG_INTERVAL_MIN = 15.0

# --------------------------------------------------------------------------
# Shortage thresholds
# --------------------------------------------------------------------------
RED_THRESHOLD = 15
AMBER_THRESHOLD = 20
TARGET_STOCK = 25

PRIMARY_HORIZON = 3    # records (~20 min)
PLANNING_HORIZON = 6   # records (~40 min)
HORIZONS = [PRIMARY_HORIZON, PLANNING_HORIZON]

# Consecutive records at/above RED required to declare shortage recovery.
EVENT_RECOVERY_RECORDS = 2

# --------------------------------------------------------------------------
# Replenishment operations
# --------------------------------------------------------------------------
REPLENISHMENT_LEAD_TIME_MIN = 10.0
# Extra carts placed in the bay are consumed by customers, so the simulated
# boost decays back towards the observed baseline dynamics. Modelled as an
# exponential half-life; a documented assumption, not a measured value.
REPLENISHMENT_RETENTION_HALFLIFE_MIN = 45.0
MIN_CARTS_PER_TRIP = 3
MAX_CARTS_PER_TRIP = 20          # None => unconstrained
BAY_CAPACITY = 60                # observed historical max Count is 58
RESERVE_CART_INVENTORY = None    # unknown -> unconstrained, documented as such
ALERT_COOLDOWN_MIN = 20.0

# --------------------------------------------------------------------------
# Cost coefficients for the replenishment objective J
# --------------------------------------------------------------------------
COST_SHORTAGE = 10.0        # lambda_1 : per cart-record of shortage severity
COST_TRIP = 5.0             # lambda_2 : per replenishment trip
COST_CART_MOVED = 0.25      # lambda_3 : per cart retrieved
COST_EXCESS = 0.10          # lambda_4 : per cart-record above target stock

COST_SENSITIVITY_GRID = {
    "COST_SHORTAGE": [5.0, 10.0, 20.0],
    "COST_TRIP": [2.0, 5.0, 10.0],
}

# --------------------------------------------------------------------------
# Chronological split (fractions of complete days)
# --------------------------------------------------------------------------
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
# Remainder is the final holdout.
MIN_RECORDS_FOR_COMPLETE_DAY = 120

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]

LGBM_PARAM_GRID = {
    "num_leaves": [7, 15, 31],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.05],
    "min_child_samples": [20, 40],
    "n_estimators": [200, 400],
}
LGBM_FIXED_PARAMS = {
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": 1,
}

# Conformal / uncertainty
CONFORMAL_ALPHA = 0.10  # -> 90% one-sided lower bound target coverage

# --------------------------------------------------------------------------
# Policy search space (validated, then frozen before holdout)
# --------------------------------------------------------------------------
POLICY_GRID = {
    "risk_threshold": [0.20, 0.30, 0.40, 0.50],
    "target_stock": [22, 25, 28],
    "quantity_rule": ["point", "risk"],
    "cooldown_min": [10.0, 20.0, 30.0],
}

ALERT_WARNING_RISK = 0.20   # default, overridden by validation calibration
ALERT_CRITICAL_RISK = 0.50
ALERT_HYSTERESIS = 0.10     # risk must fall this far below the entry threshold

ROBUSTNESS_GRID = {
    "lead_time_min": [5.0, 10.0, 15.0, 20.0],
    "target_stock": [20, 22, 25, 28, 30],
    "risk_threshold": [0.15, 0.25, 0.35, 0.45, 0.55],
    "max_carts_per_trip": [10, 15, 20, 30],
    "retention_halflife_min": [20.0, 45.0, 90.0, 180.0],
}

EPS = 1e-6


@dataclass
class RunConfig:
    """Runtime-overridable view of the configuration."""

    red_threshold: int = RED_THRESHOLD
    amber_threshold: int = AMBER_THRESHOLD
    target_stock: int = TARGET_STOCK
    primary_horizon: int = PRIMARY_HORIZON
    planning_horizon: int = PLANNING_HORIZON
    lead_time_min: float = REPLENISHMENT_LEAD_TIME_MIN
    retention_halflife_min: float = REPLENISHMENT_RETENTION_HALFLIFE_MIN
    risk_threshold: float = ALERT_WARNING_RISK
    cooldown_min: float = ALERT_COOLDOWN_MIN
    min_carts_per_trip: int = MIN_CARTS_PER_TRIP
    max_carts_per_trip: int | None = MAX_CARTS_PER_TRIP
    bay_capacity: int | None = BAY_CAPACITY
    reserve_cart_inventory: int | None = RESERVE_CART_INVENTORY
    quantity_rule: str = "point"
    windows: list = field(default_factory=lambda: list(FEATURE_WINDOWS))

    def replace(self, **kwargs) -> "RunConfig":
        data = {**self.__dict__, **kwargs}
        return RunConfig(**data)
