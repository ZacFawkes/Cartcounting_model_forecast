-- Cart Forecasting Platform — registry, operational and log tables.
-- Grain: entity = one camera stream = (KDVR, ChanNo), optionally prefixed by
-- siteKey when DVR ids are only unique within a store (decision D9b).

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- registry
CREATE TABLE IF NOT EXISTS site_config (
    site_key            INTEGER PRIMARY KEY,
    display_name        TEXT,
    timezone            TEXT    DEFAULT 'UTC',
    -- store-level operational parameters: one staff team serves every bay
    lead_time_min       REAL,
    reserve_per_day     INTEGER,          -- NULL => unconstrained (must be labelled)
    min_carts_per_trip  INTEGER,
    max_carts_per_trip  INTEGER,
    cooldown_min        REAL,
    updated_at          TEXT,
    updated_by          TEXT
);

CREATE TABLE IF NOT EXISTS entity_config (
    entity_id             TEXT PRIMARY KEY,   -- "215:0" or "163:215:0"
    kdvr                  INTEGER NOT NULL,
    chan_no               INTEGER NOT NULL,
    site_key              INTEGER NOT NULL,
    area_id               INTEGER,
    display_name          TEXT,
    status                TEXT NOT NULL DEFAULT 'shadow',  -- active|shadow|disabled

    -- physical, per bay (REQUIRED, cannot be guessed)
    bay_capacity          INTEGER,

    -- thresholds, per bay
    red_threshold         INTEGER,
    amber_threshold       INTEGER,
    target_stock          INTEGER,
    threshold_source      TEXT,             -- business|derived|manual_override

    -- operational overrides (NULL => inherit from site_config)
    lead_time_min         REAL,
    reserve_per_day       INTEGER,
    min_carts_per_trip    INTEGER,
    max_carts_per_trip    INTEGER,
    cooldown_min          REAL,
    quantity_rule         TEXT,             -- point|risk

    -- simulation assumptions (backtest only)
    consumption_model     TEXT,
    usage                 TEXT,
    retention_halflife_min REAL,

    -- data handling
    max_sequence_gap_min  REAL,

    -- bindings
    model_id              TEXT,
    calibration_id        TEXT,

    n_days_history        INTEGER DEFAULT 0,
    updated_at            TEXT,
    updated_by            TEXT,
    UNIQUE (kdvr, chan_no, site_key),
    FOREIGN KEY (site_key) REFERENCES site_config(site_key)
);

CREATE INDEX IF NOT EXISTS ix_entity_site ON entity_config(site_key);
CREATE INDEX IF NOT EXISTS ix_entity_status ON entity_config(status);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id            TEXT PRIMARY KEY,
    version             INTEGER NOT NULL,
    status              TEXT NOT NULL,     -- candidate|champion|retired
    trained_at          TEXT,
    artifact_path       TEXT NOT NULL,
    feature_count       INTEGER,
    selected_models     TEXT,              -- JSON
    entities_trained_on TEXT,              -- JSON list of entity_id
    sites_trained_on    TEXT,              -- JSON list of site_key
    train_window        TEXT,
    -- thresholds FROZEN into the feature vector at training time (decision D10)
    red_model           INTEGER NOT NULL,
    amber_model         INTEGER NOT NULL,
    metrics             TEXT,              -- JSON
    loso_metrics        TEXT,              -- JSON
    notes               TEXT,
    -- alert bands clustered from this entity's own calibrated risk (K-Means K=3)
    alert_band_id       TEXT,
    alert_band_status   TEXT,              -- fitted|insufficient_data|failed
    alert_band_json     TEXT               -- JSON: bands + clusters + diagnostics
);

CREATE TABLE IF NOT EXISTS calibrator_registry (
    calibration_id      TEXT PRIMARY KEY,
    model_id            TEXT NOT NULL,
    entity_id           TEXT,              -- NULL => global fallback calibrator
    scope               TEXT NOT NULL,     -- global|entity
    fitted_at           TEXT,
    window_start        TEXT,
    window_end          TEXT,
    n_residuals         INTEGER,
    alpha               REAL,
    lower_offset        REAL,
    empirical_coverage  REAL,
    brier               REAL,
    pr_auc              REAL,
    status              TEXT DEFAULT 'active',
    artifact_path       TEXT,
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

CREATE INDEX IF NOT EXISTS ix_calib_entity ON calibrator_registry(entity_id, status);

-- ------------------------------------------------------------- operational
CREATE TABLE IF NOT EXISTS scoring_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id              TEXT,
    entity_id               TEXT NOT NULL,
    kdvr                    INTEGER,
    chan_no                 INTEGER,
    site_key                INTEGER,
    decision_time           TEXT NOT NULL,
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
    UNIQUE (entity_id, decision_time)      -- idempotency enforced by the DB
);

CREATE INDEX IF NOT EXISTS ix_scoring_entity_time
    ON scoring_results(entity_id, decision_time DESC);
CREATE INDEX IF NOT EXISTS ix_scoring_site ON scoring_results(site_key, decision_time DESC);

CREATE TABLE IF NOT EXISTS replenishment_orders (
    order_id                 TEXT PRIMARY KEY,
    entity_id                TEXT NOT NULL,
    kdvr                     INTEGER,
    chan_no                  INTEGER,
    site_key                 INTEGER,
    decision_time            TEXT NOT NULL,
    quantity                 INTEGER NOT NULL,
    expected_completion_time TEXT NOT NULL,
    status                   TEXT NOT NULL,   -- PENDING|COMPLETED|CANCELLED
    delivered_quantity       REAL DEFAULT 0,
    delivered_at             TEXT,
    urgency                  TEXT,
    risk_at_decision         REAL,
    predicted_min_at_decision REAL,
    source                   TEXT,            -- score|backtest
    -- Replenishment ACTION LOG: the audit's blocking gap. Filled by operations.
    executed_by              TEXT,
    executed_at              TEXT,
    actual_carts             INTEGER,
    execution_notes          TEXT
);

CREATE INDEX IF NOT EXISTS ix_orders_entity_status
    ON replenishment_orders(entity_id, status);

CREATE TABLE IF NOT EXISTS session_state (
    entity_id            TEXT PRIMARY KEY,
    surplus              REAL DEFAULT 0,
    last_record_time     TEXT,
    last_decision_time   TEXT,
    alert_state          TEXT DEFAULT 'NORMAL',
    alert_streak         INTEGER DEFAULT 0,
    last_alert_time      TEXT,
    n_calls              INTEGER DEFAULT 0,
    reserve_used_today   REAL DEFAULT 0,
    reserve_day          TEXT,
    updated_at           TEXT
);

-- --------------------------------------------------------------------- logs
CREATE TABLE IF NOT EXISTS api_requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT,
    entity_id     TEXT,
    site_key      INTEGER,
    endpoint      TEXT,
    method        TEXT,
    status        TEXT,
    status_code   INTEGER,
    latency_ms    REAL,
    client        TEXT,
    request_json  TEXT,
    response_json TEXT,
    error         TEXT,
    created_at    TEXT
);

CREATE INDEX IF NOT EXISTS ix_requests_entity ON api_requests(entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_requests_status ON api_requests(status, created_at DESC);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id                 TEXT PRIMARY KEY,
    request_id             TEXT,
    entity_id              TEXT,
    site_key               INTEGER,
    label                  TEXT,
    mode                   TEXT,
    model_id               TEXT,
    window_start           TEXT,
    window_end             TEXT,
    n_records              INTEGER,
    scored_origins         INTEGER,
    baseline_events        INTEGER,
    policy_events          INTEGER,
    event_reduction_pct    REAL,
    duration_reduction_pct REAL,
    severity_reduction_pct REAL,
    replenishment_actions  INTEGER,
    carts_moved            INTEGER,
    trips_per_day          REAL,
    policy_cost_j          REAL,
    baseline_cost_j        REAL,
    mae_min_40min          REAL,
    lower_bound_coverage   REAL,
    resolved_config        TEXT,
    artifact_dir           TEXT,
    latency_ms             REAL,
    created_at             TEXT
);

CREATE INDEX IF NOT EXISTS ix_backtest_entity ON backtest_runs(entity_id, created_at DESC);

CREATE TABLE IF NOT EXISTS drift_metrics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id      TEXT NOT NULL,
    metric_date    TEXT NOT NULL,
    n_scored       INTEGER,
    mean_risk      REAL,
    mean_pred_min  REAL,
    brier          REAL,
    coverage       REAL,
    alerts_per_day REAL,
    trips_per_day  REAL,
    status         TEXT,              -- ok|warn|degraded|critical
    created_at     TEXT,
    UNIQUE (entity_id, metric_date)
);

-- ===========================================================================
-- On-demand dataset registration, scoped training and result publishing
-- (CART_PLATFORM_ON_DEMAND_ENTITY_MODULE_SPEC). All CREATE IF NOT EXISTS so
-- initialisation stays idempotent against an existing platform.db.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS dataset_uploads (
    upload_id           TEXT PRIMARY KEY,
    name                TEXT,
    original_filename   TEXT,
    stored_path         TEXT NOT NULL,
    file_checksum       TEXT NOT NULL,
    config_fingerprint  TEXT NOT NULL,
    config_json         TEXT NOT NULL,
    status              TEXT NOT NULL,        -- registered|rejected
    n_rows              INTEGER,
    n_rows_usable       INTEGER,
    n_entities          INTEGER,
    timezone            TEXT,
    quality_json        TEXT,
    warnings_json       TEXT,
    errors_json         TEXT,
    uploaded_by         TEXT,
    created_at          TEXT,
    UNIQUE (file_checksum, config_fingerprint)
);

CREATE TABLE IF NOT EXISTS upload_entities (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id           TEXT NOT NULL,
    entity_id           TEXT NOT NULL,
    kdvr                INTEGER NOT NULL,
    chan_no             INTEGER NOT NULL,
    site_key            INTEGER,
    n_records           INTEGER,
    n_usable            INTEGER,
    n_days              INTEGER,
    first_record        TEXT,
    last_record         TEXT,
    cart_min            REAL,
    cart_max            REAL,
    cart_median         REAL,
    shortage_events     INTEGER,
    trainable           INTEGER DEFAULT 0,
    blocking_reasons    TEXT,
    quality_json        TEXT,
    score_url           TEXT,
    created_at          TEXT,
    UNIQUE (upload_id, entity_id),
    FOREIGN KEY (upload_id) REFERENCES dataset_uploads(upload_id)
);

CREATE INDEX IF NOT EXISTS ix_upload_entities ON upload_entities(upload_id);

CREATE TABLE IF NOT EXISTS training_jobs (
    job_id          TEXT PRIMARY KEY,
    upload_id       TEXT NOT NULL,
    entity_ids      TEXT NOT NULL,            -- JSON list, scoped to this upload
    strategy        TEXT NOT NULL DEFAULT 'per_entity',
    config_json     TEXT,
    status          TEXT NOT NULL,            -- queued|running|succeeded|partial|failed
    progress        REAL DEFAULT 0,
    message         TEXT,
    error           TEXT,
    results_json    TEXT,                     -- per-entity outcome
    requested_by    TEXT,
    created_at      TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    heartbeat_at    TEXT,
    FOREIGN KEY (upload_id) REFERENCES dataset_uploads(upload_id)
);

CREATE INDEX IF NOT EXISTS ix_jobs_upload ON training_jobs(upload_id, created_at DESC);

CREATE TABLE IF NOT EXISTS entity_model_bindings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id       TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    calibration_id  TEXT,
    role            TEXT NOT NULL,            -- candidate|active|retired
    job_id          TEXT,
    upload_id       TEXT,
    promotable      INTEGER DEFAULT 0,
    blocking_reasons TEXT,
    promoted_at     TEXT,
    promoted_by     TEXT,
    retired_at      TEXT,
    created_at      TEXT,
    UNIQUE (entity_id, model_id)
);

CREATE INDEX IF NOT EXISTS ix_bindings_entity ON entity_model_bindings(entity_id, role);

CREATE TABLE IF NOT EXISTS calibration_records (
    calibration_id    TEXT PRIMARY KEY,
    job_id            TEXT,
    entity_id         TEXT NOT NULL,
    model_id          TEXT NOT NULL,
    status            TEXT NOT NULL,          -- fitted|insufficient_data|failed
    scope             TEXT NOT NULL,          -- entity|global_fallback|none
    method            TEXT,
    n_samples         INTEGER,
    n_positive        INTEGER,
    alpha             REAL,
    lower_offset      REAL,
    brier             REAL,
    brier_uncalibrated REAL,
    ece               REAL,
    coverage          REAL,
    window_start      TEXT,
    window_end        TEXT,
    artifact_path     TEXT,
    reliability_path  TEXT,
    metrics_json      TEXT,
    warnings_json     TEXT,
    created_at        TEXT
);

CREATE INDEX IF NOT EXISTS ix_calibration_entity ON calibration_records(entity_id, model_id);

CREATE TABLE IF NOT EXISTS backtest_reports (
    report_id       TEXT PRIMARY KEY,
    job_id          TEXT,
    entity_id       TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    calibration_id  TEXT,
    upload_id       TEXT,
    backtest_type   TEXT NOT NULL,            -- forecast_accuracy|policy_replay|closed_loop_robustness
    status          TEXT NOT NULL,            -- completed|insufficient_data|failed
    reason          TEXT,
    metrics_json    TEXT,
    assumptions_json TEXT,
    caveat          TEXT,
    artifacts_json  TEXT,
    report_path     TEXT,
    window_start    TEXT,
    window_end      TEXT,
    created_at      TEXT,
    UNIQUE (job_id, entity_id, backtest_type)
);

CREATE INDEX IF NOT EXISTS ix_backtest_entity ON backtest_reports(entity_id, backtest_type);

CREATE TABLE IF NOT EXISTS scoring_artifacts (
    request_id      TEXT PRIMARY KEY,
    entity_id       TEXT NOT NULL,
    decision_time   TEXT NOT NULL,
    artifact_path   TEXT NOT NULL,
    model_id        TEXT,
    calibration_id  TEXT,
    created_at      TEXT,
    UNIQUE (entity_id, decision_time)
);

-- Durable outbox: events are enqueued with the local result and delivered to
-- the standalone receiver by a background worker.
CREATE TABLE IF NOT EXISTS result_outbox (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id         TEXT NOT NULL UNIQUE,
    event_type       TEXT NOT NULL,
    destination      TEXT NOT NULL,
    entity_id        TEXT,
    job_id           TEXT,
    model_id         TEXT,
    payload_json     TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',  -- pending|sending|sent|dead_letter
    attempts         INTEGER DEFAULT 0,
    next_attempt_at  TEXT,
    last_error       TEXT,
    response_code    INTEGER,
    created_at       TEXT,
    sent_at          TEXT
);

CREATE INDEX IF NOT EXISTS ix_outbox_status ON result_outbox(status, next_attempt_at);
CREATE INDEX IF NOT EXISTS ix_outbox_entity ON result_outbox(entity_id, created_at DESC);

-- Immutable per-(job, entity) configuration snapshot. Makes a historical
-- backtest reproducible after the entity configuration changes.
CREATE TABLE IF NOT EXISTS config_snapshots (
    config_hash          TEXT NOT NULL,
    job_id               TEXT NOT NULL,
    entity_id            TEXT NOT NULL,
    model_id             TEXT,
    red_model            REAL NOT NULL,
    amber_model          REAL NOT NULL,
    model_dependent_json TEXT NOT NULL,
    operational_json     TEXT NOT NULL,
    sources_json         TEXT,
    warnings_json        TEXT,
    created_at           TEXT,
    PRIMARY KEY (job_id, entity_id)
);

CREATE INDEX IF NOT EXISTS ix_snapshots_entity ON config_snapshots(entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_snapshots_model ON config_snapshots(model_id);

-- Audit trail of the disk-retention job (registry/retention.py). Files of a
-- retired model are deletable; the record that they existed is not.
CREATE TABLE IF NOT EXISTS artifact_purges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    entity_id   TEXT,
    job_id      TEXT,
    upload_id   TEXT,
    kind        TEXT NOT NULL,   -- model_bundle|training_artifacts|score_artifact|upload
    path        TEXT NOT NULL,
    bytes       INTEGER,
    purged_at   TEXT
);

CREATE INDEX IF NOT EXISTS ix_purges_model ON artifact_purges(model_id);
CREATE INDEX IF NOT EXISTS ix_purges_time ON artifact_purges(purged_at DESC);
