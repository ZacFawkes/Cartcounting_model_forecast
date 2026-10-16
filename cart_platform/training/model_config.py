"""Frozen model configuration and immutable training snapshots.

Two kinds of configuration must never be confused:

**A. Model-dependent** — parameters that shape the feature matrix or the
targets, so changing them invalidates a trained model. These are *frozen* at
training start and travel with the artifact forever:
``red_model``, ``amber_model``, ``feature_windows``, ``primary_horizon``,
``planning_horizon``, ``calibration_red``.

**B. Operational** — parameters that drive alerts, replenishment quantity and
constraints. These may change after training; each backtest and scoring
decision records the effective values it used:
``target_stock``, ``bay_capacity``, ``risk_threshold``, ``lead_time_min``,
``cooldown_min``, ``min/max_carts_per_trip``, ``reserve_per_day``,
``quantity_rule``, ``consumption_model``, ``usage``, ``retention_halflife_min``.

A snapshot is immutable, hashed and persisted per (job, entity), which is what
makes a historical backtest reproducible after the entity config changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from registry.database import as_json, execute, from_json, query, query_one, utc_now

LOGGER = logging.getLogger("platform.model_config")

# Parameters that shape features/targets. Frozen with the model.
MODEL_DEPENDENT_PARAMS = (
    "red_model", "amber_model", "feature_windows",
    "primary_horizon", "planning_horizon", "calibration_red",
)

# Parameters that may evolve independently of the model.
OPERATIONAL_PARAMS = (
    "red_threshold", "amber_threshold", "target_stock", "bay_capacity",
    "lead_time_min", "cooldown_min",
    "min_carts_per_trip", "max_carts_per_trip", "quantity_rule", "reserve_per_day",
    "consumption_model", "usage", "retention_halflife_min",
    "max_sequence_gap_min",
)


class InvalidEntityConfig(ValueError):
    """Raised instead of silently falling back to unrelated thresholds."""

    def __init__(self, entity_id: str, problems: list[str]):
        self.entity_id = entity_id
        self.problems = problems
        super().__init__(
            f"INVALID_ENTITY_CONFIG for '{entity_id}': " + "; ".join(problems)
        )


@dataclass(frozen=True)
class FrozenModelConfig:
    """Model-dependent parameters, fixed for the life of one model."""

    red_model: float
    amber_model: float
    feature_windows: tuple = (3, 6, 9)
    primary_horizon: int = 3
    planning_horizon: int = 6
    calibration_red: float | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["feature_windows"] = list(self.feature_windows)
        return d

    @classmethod
    def from_mapping(cls, data: dict | None, fallback_red: float,
                     fallback_amber: float) -> "FrozenModelConfig":
        """Rebuild from a bundle/registry row, tolerating legacy artifacts."""
        data = data or {}
        windows = data.get("feature_windows") or (3, 6, 9)
        return cls(
            red_model=float(data.get("red_model", fallback_red)),
            amber_model=float(data.get("amber_model", fallback_amber)),
            feature_windows=tuple(windows),
            primary_horizon=int(data.get("primary_horizon", 3)),
            planning_horizon=int(data.get("planning_horizon", 6)),
            calibration_red=(
                None if data.get("calibration_red") is None
                else float(data["calibration_red"])
            ),
        )


@dataclass
class ConfigSnapshot:
    """Immutable record of the configuration one training run resolved."""

    entity_id: str
    job_id: str
    frozen: FrozenModelConfig
    operational: dict
    sources: dict
    warnings: list = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    config_hash: str = ""

    def __post_init__(self):
        if not self.config_hash:
            self.config_hash = self.compute_hash()

    def compute_hash(self) -> str:
        payload = json.dumps(
            {"entity_id": self.entity_id, "frozen": self.frozen.as_dict(),
             "operational": self.operational},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_payload(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "job_id": self.job_id,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "frozen_thresholds": {
                "red_model": self.frozen.red_model,
                "amber_model": self.frozen.amber_model,
            },
            "model_dependent": self.frozen.as_dict(),
            "effective_config": dict(self.operational),
            "sources": dict(self.sources),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
def _coerce_number(value, name: str, problems: list, entity_id: str,
                   integer: bool = False):
    if value is None:
        return None
    try:
        out = int(value) if integer else float(value)
    except (TypeError, ValueError):
        problems.append(f"'{name}' is not numeric (got {value!r})")
        return None
    return out


# Physical quantities that cannot be guessed. A system-default fallback for
# these is treated as "not configured", never as a usable value.
REQUIRED_PHYSICAL_PARAMS = ("bay_capacity",)


def validate(entity_id: str, red: float | None, amber: float | None,
             operational: dict, sources: dict | None = None) -> list[str]:
    """Structural checks. Anything returned here fails the entity explicitly."""
    problems: list[str] = []
    sources = sources or {}
    for name in REQUIRED_PHYSICAL_PARAMS:
        if sources.get(name) == "system_default":
            problems.append(
                f"'{name}' is not configured for this entity (resolved to the "
                f"system default); physical values are never invented"
            )
            operational.pop(name, None)
    if red is None:
        problems.append("red_threshold is not configured")
    if amber is None:
        problems.append("amber_threshold is not configured")
    if red is not None and red <= 0:
        problems.append(f"red_threshold must be positive (got {red})")
    if red is not None and amber is not None and amber <= red:
        problems.append(
            f"amber_threshold ({amber}) must be above red_threshold ({red})"
        )
    target = operational.get("target_stock")
    if target is None:
        problems.append("target_stock is not configured")
    elif amber is not None and target <= amber:
        problems.append(
            f"target_stock ({target}) must be above amber_threshold ({amber})"
        )
    capacity = operational.get("bay_capacity")
    if capacity is None:
        problems.append("bay_capacity is not configured (physical value, never invented)")
    elif target is not None and capacity < target:
        problems.append(
            f"bay_capacity ({capacity}) is below target_stock ({target})"
        )
    lo, hi = operational.get("min_carts_per_trip"), operational.get("max_carts_per_trip")
    if lo is not None and hi is not None and hi < lo:
        problems.append(f"max_carts_per_trip ({hi}) is below min_carts_per_trip ({lo})")
    return problems


def resolve_for_training(entity_id: str, job_id: str,
                         overrides: dict | None = None) -> ConfigSnapshot:
    """Resolve this entity's configuration for a NEW training run.

    Precedence is entity config -> site config -> system default. An existing
    model's frozen configuration is deliberately **not** consulted: a new
    candidate must freeze the configuration that is current now.
    """
    from registry.resolver import resolve

    # model_meta=None => the resolver does not inherit any previous model's
    # frozen thresholds; red_model/amber_model are derived below instead.
    resolved = resolve(entity_id, overrides, model_meta=None)

    problems: list[str] = []
    red = _coerce_number(resolved.values.get("red_threshold"), "red_threshold",
                         problems, entity_id)
    amber = _coerce_number(resolved.values.get("amber_threshold"), "amber_threshold",
                           problems, entity_id)

    operational = {}
    for name in OPERATIONAL_PARAMS:
        if name in resolved.values:
            operational[name] = resolved.values[name]
    problems += validate(entity_id, red, amber, operational, resolved.sources)
    if problems:
        raise InvalidEntityConfig(entity_id, problems)

    frozen = FrozenModelConfig(
        red_model=float(red),
        amber_model=float(amber),
        feature_windows=tuple(_feature_windows()),
        primary_horizon=3,
        planning_horizon=6,
        # Risk is defined as P(min < RED); the calibrator is fitted against the
        # operational RED in force at training time, recorded for reproducibility.
        calibration_red=float(red),
    )
    defaulted = [k for k in ("red_threshold", "amber_threshold", "target_stock")
                 if resolved.sources.get(k) == "system_default"]
    extra_warnings = []
    if defaulted:
        extra_warnings.append(
            f"{', '.join(defaulted)} resolved to system defaults for "
            f"'{entity_id}'; these values are frozen into the model as-is"
        )
    sources = {k: resolved.sources[k] for k in operational if k in resolved.sources}
    sources["red_model"] = f"derived_from:{resolved.sources.get('red_threshold')}"
    sources["amber_model"] = f"derived_from:{resolved.sources.get('amber_threshold')}"

    snapshot = ConfigSnapshot(
        entity_id=entity_id, job_id=job_id, frozen=frozen,
        operational=operational, sources=sources,
        warnings=[w for w in resolved.warnings if "differs from" not in w]
        + extra_warnings,
    )
    LOGGER.info(
        "Resolved training config for %s: red_model=%s amber_model=%s "
        "target_stock=%s bay_capacity=%s (hash %s)",
        entity_id, frozen.red_model, frozen.amber_model,
        operational.get("target_stock"), operational.get("bay_capacity"),
        snapshot.config_hash,
    )
    return snapshot


def _feature_windows() -> list:
    from config import FEATURE_WINDOWS

    return list(FEATURE_WINDOWS)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def persist(snapshot: ConfigSnapshot, model_id: str | None = None) -> None:
    execute(
        """INSERT OR REPLACE INTO config_snapshots
           (config_hash, job_id, entity_id, model_id, red_model, amber_model,
            model_dependent_json, operational_json, sources_json, warnings_json,
            created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (snapshot.config_hash, snapshot.job_id, snapshot.entity_id, model_id,
         snapshot.frozen.red_model, snapshot.frozen.amber_model,
         as_json(snapshot.frozen.as_dict()), as_json(snapshot.operational),
         as_json(snapshot.sources), as_json(snapshot.warnings),
         snapshot.created_at),
    )


def get_snapshot(job_id: str, entity_id: str) -> dict | None:
    row = query_one(
        "SELECT * FROM config_snapshots WHERE job_id=? AND entity_id=?",
        (job_id, entity_id),
    )
    return _expand(row)


def snapshot_for_model(model_id: str) -> dict | None:
    return _expand(
        query_one("SELECT * FROM config_snapshots WHERE model_id=?", (model_id,))
    )


def list_snapshots(entity_id: str | None = None, limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM config_snapshots WHERE 1=1"
    params: list = []
    if entity_id:
        sql += " AND entity_id=?"
        params.append(entity_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return [_expand(r) for r in query(sql, tuple(params))]


def _expand(row: dict | None) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    out["model_dependent"] = from_json(row.get("model_dependent_json"), {})
    out["effective_config"] = from_json(row.get("operational_json"), {})
    out["sources"] = from_json(row.get("sources_json"), {})
    out["warnings"] = from_json(row.get("warnings_json"), [])
    out["frozen_thresholds"] = {"red_model": row.get("red_model"),
                                "amber_model": row.get("amber_model")}
    for k in ("model_dependent_json", "operational_json", "sources_json",
              "warnings_json"):
        out.pop(k, None)
    return out


# --------------------------------------------------------------------------
# Compatibility between a frozen model and current operational config
# --------------------------------------------------------------------------
def compatibility_warnings(frozen: FrozenModelConfig, operational: dict) -> list[str]:
    """Explicit warnings when operational config drifts from the frozen model."""
    out: list[str] = []
    red_op = operational.get("red_threshold")
    amber_op = operational.get("amber_threshold")
    if red_op is not None and float(red_op) != float(frozen.red_model):
        out.append(
            f"operational red_threshold={red_op} differs from the model's frozen "
            f"red_model={frozen.red_model}. Features are built with "
            f"{frozen.red_model} (the trained contract); alerts, quantity and "
            f"replenishment use {red_op}. Retrain to move the feature-time "
            f"threshold."
        )
    if amber_op is not None and float(amber_op) != float(frozen.amber_model):
        out.append(
            f"operational amber_threshold={amber_op} differs from the model's "
            f"frozen amber_model={frozen.amber_model}; features use "
            f"{frozen.amber_model}, operational logic uses {amber_op}."
        )
    if frozen.calibration_red is not None and red_op is not None:
        if float(red_op) != float(frozen.calibration_red):
            out.append(
                f"shortage probabilities were calibrated against RED="
                f"{frozen.calibration_red}; the current operational RED is "
                f"{red_op}, so risk values are not calibrated for that definition."
            )
    return out
