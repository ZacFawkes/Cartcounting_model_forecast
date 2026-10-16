"""Versioned result event envelope shared by the ML server and the receiver."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from settings import EVENT_SCHEMA_VERSION

TRAINING_COMPLETED = "training.completed"
CALIBRATION_COMPLETED = "calibration.completed"
BACKTEST_COMPLETED = "backtest.completed"
SCORING_RECORDED = "scoring.recorded"

ROUTES = {
    TRAINING_COMPLETED: "/v1/results/training",
    CALIBRATION_COMPLETED: "/v1/results/calibration",
    BACKTEST_COMPLETED: "/v1/results/backtest",
    SCORING_RECORDED: "/v1/results/scoring",
}


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def payload_hash(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass
class ResultEvent:
    event_id: str
    event_type: str
    payload: dict
    entity_id: str | None = None
    site_key: str | None = None
    upload_id: str | None = None
    job_id: str | None = None
    model_id: str | None = None
    schema_version: int = EVENT_SCHEMA_VERSION
    occurred_at: str = field(default_factory=utc_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["site_key"] = None if self.site_key is None else str(self.site_key)
        return d

    @property
    def route(self) -> str:
        return ROUTES[self.event_type]


def _slug(value) -> str:
    return str(value).replace(" ", "_") if value is not None else "none"


def training_event_id(job_id, entity_id, status) -> str:
    return f"{_slug(job_id)}:{_slug(entity_id)}:training:{_slug(status)}:v{EVENT_SCHEMA_VERSION}"


def calibration_event_id(job_id, entity_id, calibration_id) -> str:
    return (f"{_slug(job_id)}:{_slug(entity_id)}:calibration:"
            f"{_slug(calibration_id)}:v{EVENT_SCHEMA_VERSION}")


def backtest_event_id(job_id, entity_id, backtest_type, status) -> str:
    return (f"{_slug(job_id)}:{_slug(entity_id)}:{_slug(backtest_type)}:"
            f"{_slug(status)}:v{EVENT_SCHEMA_VERSION}")


def scoring_event_id(entity_id, decision_time, model_id) -> str:
    return (f"score:{_slug(entity_id)}:{_slug(decision_time)}:"
            f"{_slug(model_id)}:v{EVENT_SCHEMA_VERSION}")
