-- Standalone result receiver database (results_receiver.db).
-- Represents a future external backend. The ML server writes here ONLY over
-- HTTP; it never opens this file directly.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS received_events (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    schema_version  INTEGER NOT NULL,
    entity_id       TEXT,
    site_key        TEXT,
    upload_id       TEXT,
    job_id          TEXT,
    model_id        TEXT,
    occurred_at     TEXT,
    received_at     TEXT NOT NULL,
    payload_hash    TEXT NOT NULL,
    payload_json    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_events_entity ON received_events(entity_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_events_type ON received_events(event_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_events_job ON received_events(job_id);

CREATE TABLE IF NOT EXISTS training_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      TEXT NOT NULL UNIQUE,
    job_id        TEXT,
    entity_id     TEXT,
    site_key      TEXT,
    upload_id     TEXT,
    model_id      TEXT,
    status        TEXT,
    occurred_at   TEXT,
    received_at   TEXT,
    metrics_json  TEXT,
    metadata_json TEXT,
    config_hash           TEXT,
    frozen_config_json    TEXT,
    effective_config_json TEXT,
    FOREIGN KEY (event_id) REFERENCES received_events(event_id)
);

CREATE INDEX IF NOT EXISTS ix_training_entity ON training_results(entity_id, occurred_at DESC);

CREATE TABLE IF NOT EXISTS calibration_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT NOT NULL UNIQUE,
    job_id         TEXT,
    entity_id      TEXT,
    site_key       TEXT,
    upload_id      TEXT,
    model_id       TEXT,
    calibrator_id  TEXT,
    status         TEXT,
    scope          TEXT,
    method         TEXT,
    occurred_at    TEXT,
    received_at    TEXT,
    metrics_json   TEXT,
    metadata_json  TEXT,
    config_hash    TEXT,
    FOREIGN KEY (event_id) REFERENCES received_events(event_id)
);

CREATE INDEX IF NOT EXISTS ix_calib_entity ON calibration_results(entity_id, occurred_at DESC);

CREATE TABLE IF NOT EXISTS backtest_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id         TEXT NOT NULL UNIQUE,
    job_id           TEXT,
    entity_id        TEXT,
    site_key         TEXT,
    upload_id        TEXT,
    model_id         TEXT,
    report_id        TEXT,
    backtest_type    TEXT NOT NULL,
    status           TEXT,
    reason           TEXT,
    occurred_at      TEXT,
    received_at      TEXT,
    metrics_json     TEXT,
    assumptions_json TEXT,
    caveat           TEXT,
    artifacts_json   TEXT,
    config_hash           TEXT,
    frozen_config_json    TEXT,
    effective_config_json TEXT,
    FOREIGN KEY (event_id) REFERENCES received_events(event_id)
);

CREATE INDEX IF NOT EXISTS ix_backtest_entity
    ON backtest_results(entity_id, backtest_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_backtest_type ON backtest_results(backtest_type);

CREATE TABLE IF NOT EXISTS scoring_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id                TEXT NOT NULL UNIQUE,
    request_id              TEXT,
    entity_id               TEXT,
    kdvr                    INTEGER,
    chan_no                 INTEGER,
    site_key                TEXT,
    decision_time           TEXT,
    current_cart            REAL,
    simulated_cart          REAL,
    pred_cart_20min         REAL,
    pred_cart_40min         REAL,
    pred_min_20min          REAL,
    pred_min_40min          REAL,
    lower_bound_min_40min   REAL,
    shortage_risk           REAL,
    alert_state             TEXT,
    quantity                INTEGER,
    model_id                TEXT,
    calibration_id          TEXT,
    scored_at               TEXT,
    FOREIGN KEY (event_id) REFERENCES received_events(event_id)
);

CREATE INDEX IF NOT EXISTS ix_scoring_entity
    ON scoring_results(entity_id, decision_time DESC);
CREATE UNIQUE INDEX IF NOT EXISTS ux_scoring_decision
    ON scoring_results(entity_id, decision_time, model_id);
