"""Shared fixtures: isolated databases, in-process receiver, sample dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

import settings  # noqa: E402,F401  — extends sys.path to the modelling pipeline


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory) -> Path:
    """Three interleaved (KDVR, ChanNo) streams derived from real telemetry."""
    from scripts.make_sample_dataset import build

    path = tmp_path_factory.mktemp("datasets") / "sample_multi_entity.csv"
    build().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def isolated_db(tmp_path_factory):
    """Point every module at a fresh platform.db and artifact tree."""
    import settings as C
    from registry import database

    root = tmp_path_factory.mktemp("platform")
    C.DB_PATH = root / "platform.db"
    database.DB_PATH = C.DB_PATH
    C.UPLOAD_DIR = root / "uploads"
    C.LOG_DIR = root / "logs"
    C.RUN_DIR = C.LOG_DIR / "runs"
    C.MODEL_DIR = root / "models"
    C.PER_ENTITY_MODEL_DIR = C.MODEL_DIR / "per_entity"
    C.SCORE_ARTIFACT_DIR = C.LOG_DIR / "scores"
    C.TRAINING_ARTIFACT_DIR = C.LOG_DIR / "training"
    C.SERVER_LOG = C.LOG_DIR / "server.log"
    for d in (C.UPLOAD_DIR, C.LOG_DIR, C.RUN_DIR, C.MODEL_DIR,
              C.PER_ENTITY_MODEL_DIR, C.SCORE_ARTIFACT_DIR, C.TRAINING_ARTIFACT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Modules that captured the originals at import time (``from settings
    # import X`` binds a copy, so re-pointing settings alone is not enough).
    import server.artifacts as artifacts
    import server.logs as server_logs
    import training.jobs as jobs
    import training.trainer as trainer

    artifacts.SCORE_ARTIFACT_DIR = C.SCORE_ARTIFACT_DIR
    jobs.TRAINING_ARTIFACT_DIR = C.TRAINING_ARTIFACT_DIR
    trainer.PER_ENTITY_MODEL_DIR = C.PER_ENTITY_MODEL_DIR
    server_logs.LOG_DIR = C.LOG_DIR
    server_logs.RUN_DIR = C.RUN_DIR
    server_logs.SERVER_LOG = C.SERVER_LOG

    import registry.uploads as up

    up.UPLOAD_DIR = C.UPLOAD_DIR

    # Never let a test publish to a receiver that happens to be running on the
    # developer's machine: a live :8200 would consume outbox events and the
    # in-process receiver assertions would silently see nothing.
    import integrations.publisher as publisher

    C.RESULT_RECEIVER_ENABLED = False
    C.RESULT_RECEIVER_URL = "http://127.0.0.1:1"
    publisher.RESULT_RECEIVER_ENABLED = False
    publisher.RESULT_RECEIVER_URL = C.RESULT_RECEIVER_URL

    database.init_db(C.DB_PATH)
    return root


@pytest.fixture(scope="module")
def receiver_client(tmp_path_factory):
    """Standalone receiver running in-process against its own SQLite file."""
    from fastapi.testclient import TestClient

    import settings as C
    from receiver import database as rdb

    C.RECEIVER_DB_PATH = tmp_path_factory.mktemp("receiver") / "results_receiver.db"
    rdb.reset_connection()
    rdb.configure(C.RECEIVER_DB_PATH)

    from receiver.app import app as receiver_app

    with TestClient(receiver_app) as c:
        yield c


class _InProcessSession:
    """Routes publisher HTTP calls to the in-process receiver TestClient."""

    def __init__(self, client, fail_times: int = 0, status: int = 503):
        self.client = client
        self.fail_times = fail_times
        self.status = status
        self.calls = 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1

            class _Resp:
                status_code = self.status
                text = "simulated receiver failure"
                headers: dict = {}

            return _Resp()
        path = url.split("8200", 1)[1] if "8200" in url else url
        return self.client.post(path, json=json, headers=headers)


@pytest.fixture
def publish_session(receiver_client):
    return _InProcessSession(receiver_client)
