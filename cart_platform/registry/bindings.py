"""Per-entity model bindings, promotion gates and rollback.

A finished training job produces a **candidate**. Promotion is always explicit:
nothing in this module is ever called automatically.
"""
from __future__ import annotations

import logging

from registry import entities as reg
from registry.database import execute, from_json, query, query_one, utc_now
from training import calibration as cal

LOGGER = logging.getLogger("platform.bindings")

CANDIDATE, ACTIVE, RETIRED = "candidate", "active", "retired"


class PromotionBlocked(ValueError):
    def __init__(self, message: str, reasons: list | None = None):
        super().__init__(message)
        self.reasons = reasons or [message]


class BindingNotFound(KeyError):
    pass


def list_bindings(entity_id: str, role: str | None = None) -> list[dict]:
    sql = "SELECT * FROM entity_model_bindings WHERE entity_id=?"
    params: list = [entity_id]
    if role:
        sql += " AND role=?"
        params.append(role)
    return query(sql + " ORDER BY created_at DESC", tuple(params))


def active_binding(entity_id: str) -> dict | None:
    return query_one(
        "SELECT * FROM entity_model_bindings WHERE entity_id=? AND role=? "
        "ORDER BY promoted_at DESC",
        (entity_id, ACTIVE),
    )


def get_binding(entity_id: str, model_id: str) -> dict | None:
    return query_one(
        "SELECT * FROM entity_model_bindings WHERE entity_id=? AND model_id=?",
        (entity_id, model_id),
    )


def readiness(entity_id: str) -> dict:
    """Everything a caller needs to know about this entity's model state."""
    active = active_binding(entity_id)
    candidates = list_bindings(entity_id, CANDIDATE)
    return {
        "entity_id": entity_id,
        "model_status": "ready" if active else (
            "candidate_not_promoted" if candidates else "not_ready"
        ),
        "active": _decorate(active),
        "candidates": [_decorate(c) for c in candidates],
        "history": [_decorate(b) for b in list_bindings(entity_id, RETIRED)],
    }


def _decorate(binding: dict | None) -> dict | None:
    if binding is None:
        return None
    out = dict(binding)
    out["blocking_reasons"] = from_json(binding.get("blocking_reasons"), []) or []
    out["promotable"] = bool(binding.get("promotable"))
    model = query_one("SELECT * FROM model_registry WHERE model_id=?",
                      (binding["model_id"],))
    if model:
        out["model"] = {
            "model_id": model["model_id"],
            "status": model["status"],
            "trained_at": model["trained_at"],
            "feature_count": model["feature_count"],
            "feature_schema_hash": model.get("feature_schema_hash"),
            "red_model": model["red_model"],
            "amber_model": model["amber_model"],
            "upload_id": model.get("upload_id"),
            "job_id": model.get("job_id"),
            "calibration_status": model.get("calibration_status"),
            "min_history_records": model.get("min_history_records"),
        }
    record = cal.get_calibration(binding["calibration_id"]) if binding.get(
        "calibration_id") else None
    if record:
        out["calibration"] = {
            "calibration_id": record["calibration_id"],
            "status": record["status"], "scope": record["scope"],
            "method": record["method"], "n_samples": record["n_samples"],
            "n_positive": record["n_positive"], "brier": record["brier"],
            "ece": record["ece"], "coverage": record["coverage"],
        }
    return out


# --------------------------------------------------------------------------
# Promotion
# --------------------------------------------------------------------------
def validate_promotion(entity_id: str, model_id: str) -> list[str]:
    """Gate checks. Empty list means the candidate may be activated."""
    reasons: list[str] = []
    binding = get_binding(entity_id, model_id)
    if binding is None:
        raise BindingNotFound(
            f"model '{model_id}' is not a candidate for entity '{entity_id}'"
        )
    if binding["role"] == ACTIVE:
        reasons.append("model is already active for this entity")

    model = query_one("SELECT * FROM model_registry WHERE model_id=?", (model_id,))
    if model is None:
        reasons.append(f"model '{model_id}' is not in the model registry")
    elif model.get("entity_id") and model["entity_id"] != entity_id:
        reasons.append(
            f"model was trained for entity '{model['entity_id']}', not '{entity_id}'"
        )

    record = cal.get_calibration(binding["calibration_id"]) if binding.get(
        "calibration_id") else None
    if record is None:
        reasons.append("no calibration record is linked to this candidate")
    elif record["status"] != cal.FITTED:
        reasons.append(
            f"calibration status is '{record['status']}'; calibrated risk scoring "
            f"cannot be promoted"
        )
    elif record["model_id"] != model_id:
        reasons.append("calibrator is bound to a different model_id")

    reasons += from_json(binding.get("blocking_reasons"), []) or []
    # Deduplicate while preserving order.
    seen, out = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def promote(entity_id: str, model_id: str, promoted_by: str = "api",
            force_reason: str | None = None) -> dict:
    """Atomically activate a candidate. Never called automatically."""
    reasons = validate_promotion(entity_id, model_id)
    if reasons and not force_reason:
        raise PromotionBlocked(
            "candidate does not pass the promotion gates", reasons
        )

    binding = get_binding(entity_id, model_id)
    previous = active_binding(entity_id)
    now = utc_now()

    if previous and previous["model_id"] != model_id:
        execute(
            "UPDATE entity_model_bindings SET role=?, retired_at=? "
            "WHERE entity_id=? AND model_id=?",
            (RETIRED, now, entity_id, previous["model_id"]),
        )
        execute("UPDATE model_registry SET status=? WHERE model_id=?",
                (RETIRED, previous["model_id"]))

    execute(
        "UPDATE entity_model_bindings SET role=?, promoted_at=?, promoted_by=? "
        "WHERE entity_id=? AND model_id=?",
        (ACTIVE, now, promoted_by, entity_id, model_id),
    )
    execute("UPDATE model_registry SET status=? WHERE model_id=?",
            ("champion", model_id))
    # Keep the legacy entity_config binding in sync for backward compatibility.
    execute(
        "UPDATE entity_config SET model_id=?, calibration_id=?, updated_at=? "
        "WHERE entity_id=?",
        (model_id, binding.get("calibration_id"), now, entity_id),
    )
    try:
        reg.set_status(entity_id, reg.ACTIVE, updated_by=promoted_by)
    except reg.UnknownEntity:  # pragma: no cover
        pass

    LOGGER.info("Promoted %s -> %s by %s%s", entity_id, model_id, promoted_by,
                " (FORCED)" if force_reason else "")
    out = readiness(entity_id)
    out["promoted"] = True
    out["previous_model_id"] = previous["model_id"] if previous else None
    out["forced"] = bool(force_reason)
    out["force_reason"] = force_reason
    out["audited_exception"] = force_reason
    return out


def rollback(entity_id: str, promoted_by: str = "api") -> dict:
    """Restore the most recently retired binding for this entity."""
    current = active_binding(entity_id)
    previous = query_one(
        "SELECT * FROM entity_model_bindings WHERE entity_id=? AND role=? "
        "ORDER BY retired_at DESC",
        (entity_id, RETIRED),
    )
    if previous is None:
        raise BindingNotFound(f"no retired binding to roll back to for '{entity_id}'")
    now = utc_now()
    if current:
        execute(
            "UPDATE entity_model_bindings SET role=?, retired_at=? "
            "WHERE entity_id=? AND model_id=?",
            (RETIRED, now, entity_id, current["model_id"]),
        )
        execute("UPDATE model_registry SET status=? WHERE model_id=?",
                (RETIRED, current["model_id"]))
    execute(
        "UPDATE entity_model_bindings SET role=?, promoted_at=?, promoted_by=?, "
        "retired_at=NULL WHERE entity_id=? AND model_id=?",
        (ACTIVE, now, promoted_by, entity_id, previous["model_id"]),
    )
    execute("UPDATE model_registry SET status=? WHERE model_id=?",
            ("champion", previous["model_id"]))
    execute(
        "UPDATE entity_config SET model_id=?, calibration_id=?, updated_at=? "
        "WHERE entity_id=?",
        (previous["model_id"], previous.get("calibration_id"), now, entity_id),
    )
    LOGGER.warning("Rolled back %s to %s", entity_id, previous["model_id"])
    out = readiness(entity_id)
    out["rolled_back_to"] = previous["model_id"]
    return out
