"""Three-level parameter resolution with full source traceability.

    1. API request parameter        explicit per-call override
    2. entity_config / site_config  the operator's per-KDVR:ChanNo default
    3. SYSTEM_DEFAULTS              last-resort fallback, flagged

The two alert bands are NOT in that chain. ``risk_threshold`` and
``critical_risk_threshold`` are no longer configuration: they are clustered
from the entity's own calibrated risk at training time (K-Means, K=3) and
frozen into the model. The resolver reads them back from the model row and
reports the source as ``model_kmeans``. When banding was impossible the
documented default is served and the source says ``system_default``, so a
fallback is never mistaken for a measured value.

Also implements the ``red_model`` / ``red_eval`` split (decision D10): the
thresholds frozen into the feature vector at training time are never changed by
a request, while the evaluation thresholds that drive risk, alerts and quantity
are freely overridable. When the two differ the caller is warned explicitly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from settings import (
    DEFAULT_BAND_CRITICAL,
    DEFAULT_BAND_WARNING,
    ENTITY_LEVEL_PARAMS,
    FEATURE_COUPLED_PARAMS,
    REQUEST_OVERRIDABLE,
    SITE_LEVEL_PARAMS,
    SYSTEM_DEFAULTS,
)
from registry.entities import get_entity, get_site

LOGGER = logging.getLogger("platform.resolver")

REQUEST = "request_override"
ENTITY = "entity_config"
SITE = "site_config"
SYSTEM = "system_default"
MODEL = "model_registry"
BANDS = "model_kmeans"

# Measured, not configured — but overridable for a single call.
BAND_PARAMS = {"risk_threshold", "critical_risk_threshold"}


@dataclass
class ResolvedConfig:
    entity_id: str
    values: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    entity: dict = field(default_factory=dict)
    site: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.values[key]

    def get(self, key, default=None):
        return self.values.get(key, default)

    @property
    def unconfigured(self) -> list:
        """Parameters that fell through to a system default.

        The alert bands are excluded: they are measured, not configured, so a
        fallback there is reported by its own warning rather than as a gap in
        the operator's configuration.
        """
        return sorted(k for k, v in self.sources.items()
                      if v == SYSTEM and k not in BAND_PARAMS)

    def as_payload(self) -> dict:
        return {
            k: {"value": self.values[k], "source": self.sources[k]}
            for k in sorted(self.values)
        }

    def policy_kwargs(self) -> dict:
        """Arguments for the audited PolicyConfig (evaluation thresholds)."""
        return {
            "red_threshold": float(self.values["red_threshold"]),
            "amber_threshold": float(self.values["amber_threshold"]),
            "target_stock": float(self.values["target_stock"]),
            "risk_threshold": float(self.values["risk_threshold"]),
            "lead_time_min": float(self.values["lead_time_min"]),
            "cooldown_min": float(self.values["cooldown_min"]),
            "min_carts_per_trip": int(self.values["min_carts_per_trip"]),
            "max_carts_per_trip": int(self.values["max_carts_per_trip"]),
            "quantity_rule": str(self.values["quantity_rule"]),
        }

    def inventory_kwargs(self) -> dict:
        return {
            "bay_capacity": float(self.values["bay_capacity"]),
            "usage": str(self.values["usage"]),
            "consumption_model": str(self.values["consumption_model"]),
            "retention_halflife_min": float(self.values["retention_halflife_min"]),
        }


def resolve(
    entity_id: str,
    overrides: dict | None = None,
    model_meta: dict | None = None,
) -> ResolvedConfig:
    """Resolve every parameter for one entity, recording where it came from."""
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    entity = get_entity(entity_id)
    site = get_site(entity["site_key"]) or {}

    res = ResolvedConfig(entity_id=entity_id, entity=entity, site=site)

    for name, default in SYSTEM_DEFAULTS.items():
        value, source = default, SYSTEM

        if name in ENTITY_LEVEL_PARAMS and entity.get(name) is not None:
            value, source = entity[name], ENTITY
        elif name in SITE_LEVEL_PARAMS and site.get(name) is not None:
            value, source = site[name], SITE

        if name in overrides:
            if name not in REQUEST_OVERRIDABLE:
                res.warnings.append(f"'{name}' cannot be overridden per request; ignored")
            else:
                value, source = overrides[name], REQUEST

        res.values[name] = value
        res.sources[name] = source

    res.values["_band_overrides"] = {
        k: overrides.get(k)
        for k in ("risk_threshold", "critical_risk_threshold")
    }

    unknown = set(overrides) - REQUEST_OVERRIDABLE - BAND_PARAMS
    for u in sorted(unknown):
        res.warnings.append(f"unknown parameter '{u}' ignored")

    _apply_alert_bands(res, model_meta)
    _apply_feature_threshold_split(res, model_meta, overrides)
    _sanity_checks(res)
    return res


def _apply_alert_bands(res: ResolvedConfig, model_meta) -> None:
    """Read the clustered alert bands back off the model.

    These are measured, not configured, so an operator value is not consulted:
    there is nothing sensible for a request to override when the bands describe
    where this entity's own risk distribution actually separates. Rejecting an
    override loudly is better than silently ignoring it.
    """
    from training.alert_bands import bands_from_meta

    overrides = res.values.pop("_band_overrides", {}) or {}

    bands = bands_from_meta(model_meta)
    if bands:
        res.values["risk_threshold"] = float(bands["warning"])
        res.values["critical_risk_threshold"] = float(bands["critical"])
        res.sources["risk_threshold"] = BANDS
        res.sources["critical_risk_threshold"] = BANDS
        _apply_band_overrides(res, overrides)
        return

    res.values["risk_threshold"] = float(DEFAULT_BAND_WARNING)
    res.values["critical_risk_threshold"] = float(DEFAULT_BAND_CRITICAL)
    res.sources["risk_threshold"] = SYSTEM
    res.sources["critical_risk_threshold"] = SYSTEM
    if model_meta and model_meta.get("model_id"):
        # A real model that should have carried bands but does not.
        status = model_meta.get("alert_band_status") or "absent"
        res.warnings.append(
            f"alert bands were not clustered for this model ({status}); "
            f"serving the default {DEFAULT_BAND_WARNING}/{DEFAULT_BAND_CRITICAL}. "
            f"Retrain to measure them from this entity's risk distribution."
        )
    _apply_band_overrides(res, overrides)


def _apply_band_overrides(res: ResolvedConfig, overrides: dict) -> None:
    """Per-call escape hatch for diagnostics and audited exceptions.

    A band may be overridden for ONE call. It is never written to
    ``entity_config`` and never survives the request, so the measured operating
    point remains the entity's default. The source says ``request_override`` so
    a bypass is always visible in the decision record.
    """
    for name in ("risk_threshold", "critical_risk_threshold"):
        if overrides.get(name) is None:
            continue
        res.values[name] = float(overrides[name])
        res.sources[name] = REQUEST
        res.warnings.append(
            f"{name} was overridden for this call ({overrides[name]}); the "
            f"model's clustered band is not in effect"
        )


def _apply_feature_threshold_split(res: ResolvedConfig, model_meta, overrides) -> None:
    """Keep feature-time thresholds frozen; warn when evaluation differs (D10).

    17 of the served features are threshold-derived (``below_red``,
    ``frac_below_amber_*``, ``ttb_red_*``, ``ttb_amber_*``). They are built with
    the thresholds the model was TRAINED on. Honouring a request override there
    would hand the model a feature vector it has never seen, silently.
    """
    red_model = (model_meta or {}).get("red_model", SYSTEM_DEFAULTS["red_threshold"])
    amber_model = (model_meta or {}).get("amber_model", SYSTEM_DEFAULTS["amber_threshold"])

    res.values["red_model"] = red_model
    res.values["amber_model"] = amber_model
    res.sources["red_model"] = MODEL
    res.sources["amber_model"] = MODEL

    for name, frozen in (("red_threshold", red_model), ("amber_threshold", amber_model)):
        evaluated = res.values[name]
        if float(evaluated) != float(frozen):
            feature_name = name.replace("_threshold", "_model")
            res.warnings.append(
                f"{name}={evaluated} differs from {feature_name}={frozen}; features were "
                f"built with the trained threshold, while risk, alerts and quantity use "
                f"{evaluated}. Retrain to change the feature-time threshold."
            )


def _sanity_checks(res: ResolvedConfig) -> None:
    v = res.values
    if v["amber_threshold"] <= v["red_threshold"]:
        res.warnings.append(
            f"amber_threshold ({v['amber_threshold']}) is not above red_threshold "
            f"({v['red_threshold']})"
        )
    if v["target_stock"] <= v["amber_threshold"]:
        res.warnings.append(
            f"target_stock ({v['target_stock']}) is not above amber_threshold "
            f"({v['amber_threshold']})"
        )
    if v["bay_capacity"] <= v["target_stock"]:
        res.warnings.append(
            f"bay_capacity ({v['bay_capacity']}) is not above target_stock "
            f"({v['target_stock']}); top-ups will be capacity-constrained"
        )
    if v["max_carts_per_trip"] < v["min_carts_per_trip"]:
        res.warnings.append("max_carts_per_trip is below min_carts_per_trip")
    if res.entity.get("threshold_source") == "derived":
        res.warnings.append(
            "thresholds are percentile-derived from history, not a business rule"
        )
    if res.unconfigured:
        res.warnings.append(
            f"using system defaults for unconfigured parameters: "
            f"{', '.join(res.unconfigured)}"
        )
    if res.entity.get("reserve_per_day") is None and res.site.get("reserve_per_day") is None:
        res.warnings.append(
            "reserve inventory is unknown; recommendations are conditionally feasible only"
        )
