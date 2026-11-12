"""Disk and database retention for retired models.

A model whose ``model_registry.status`` is ``retired`` can never be scored
again: the resolver only ever binds a ``champion``, and a rollback promotes a
``candidate``. Its bundle, training logs, backtest and calibration artifacts
and — when nothing else references it — its source upload are therefore dead
weight, on disk and in ``platform.db``.

Four passes, all gated on the same age cutoff (default one week):

1. **files** — bundle, training artifacts and source upload of a retired model;
2. **database** — the dependent rows of a retired model (calibration records,
   backtest reports, config snapshots, bindings, its training job and unshared
   upload);
3. **decisions** — ``scoring_results`` and ``replenishment_orders``, which
   otherwise grow forever at one row per record per entity. Both are forwarded
   to the result receiver, which is their system of record, so once a decision
   is **acknowledged there** and older than the cutoff the local copy is
   redundant;
4. **housekeeping** — per-decision score CSVs, delivered outbox events and the
   ``api_requests`` traffic log.

What is **never** deleted:

* anything belonging to a model that is not ``retired`` — champion and
  candidate bundles, their training artifacts, calibration and backtest rows
  and their registry entries are protected by an explicit live-model guard on
  every statement;
* the ``model_registry`` row itself, so the audit trail of what was trained,
  promoted and retired survives; it is stamped with ``purged_at`` /
  ``purged_bytes`` instead;
* **a decision the receiver has not acknowledged** — an event still
  ``pending``, ``sending`` or ``dead_letter`` means the other server does not
  have that record, so the local copy is the only one and it stays;
* an open (``PENDING``) replenishment order — that is outstanding work on the
  floor, not a log line;
* undelivered outbox work (``pending`` / ``sending``).

Safety rules for paths:

* a path is deleted only if it resolves inside ``models/``, ``logs/`` or
  ``data/uploads/`` — never outside the platform tree, never a root itself;
* a training-job directory is pruned per entity, because one job can train
  several entities, and the job directory goes only once it is empty;
* an upload directory goes only when every model trained from it is retired
  and no unfinished training job still references it.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import settings as C
from registry.database import connect, utc_now

LOGGER = logging.getLogger("platform.retention")

DEFAULT_GRACE_DAYS = 7

#: Training jobs in these states may still write into their artifact tree.
LIVE_JOB_STATES = ("queued", "running")

#: Outbox events still owed to the receiver. Never deleted at any age.
UNDELIVERED_OUTBOX_STATES = ("pending", "sending")

#: Dependent rows keyed by ``model_id``, deleted when that model is retired.
#: ``scoring_results`` is absent: a decision is deleted by its own pass, and
#: only once the receiver has acknowledged it — never because its model was
#: retired. ``result_outbox`` is absent too: delivery state, not model state,
#: decides when an event may go.
MODEL_SCOPED_TABLES: tuple[tuple[str, str], ...] = (
    ("scoring_artifacts", "model_id"),
    ("calibration_records", "model_id"),
    ("calibrator_registry", "model_id"),
    ("backtest_reports", "model_id"),
    ("backtest_runs", "model_id"),
    ("config_snapshots", "model_id"),
    ("entity_model_bindings", "model_id"),
)

#: Rows keyed by ``job_id``, deleted only when no live model and no unfinished
#: job still needs that training job.
JOB_SCOPED_TABLES: tuple[tuple[str, str], ...] = (
    ("config_snapshots", "job_id"),
    ("backtest_reports", "job_id"),
    ("calibration_records", "job_id"),
    ("training_jobs", "job_id"),
)

#: Rows keyed by ``upload_id``, deleted only when the upload is unshared.
UPLOAD_SCOPED_TABLES: tuple[tuple[str, str], ...] = (
    ("upload_entities", "upload_id"),
    ("dataset_uploads", "upload_id"),
)


@dataclass
class PurgeTarget:
    """One filesystem path scheduled for deletion."""

    path: Path
    kind: str           # model_bundle|training_artifacts|score_artifact|upload
    model_id: str
    bytes: int = 0


@dataclass
class ModelPurge:
    model_id: str
    entity_id: str | None
    job_id: str | None
    upload_id: str | None
    retired_at: str | None
    targets: list[PurgeTarget] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(t.bytes for t in self.targets)


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def _safe_roots() -> list[Path]:
    """Resolved at call time: the test suite rebinds these settings."""
    roots = []
    for root in (C.MODEL_DIR, C.LOG_DIR, C.UPLOAD_DIR):
        try:
            roots.append(Path(root).resolve())
        except OSError:          # pragma: no cover — unreadable mount
            continue
    return roots


def _is_safe(path: Path) -> bool:
    """True when *path* lives strictly below a managed root."""
    try:
        resolved = path.resolve()
    except OSError:              # pragma: no cover — broken symlink chain
        return False
    for root in _safe_roots():
        if resolved == root:
            return False         # never delete a root itself
        if root in resolved.parents:
            return True
    return False


def _size_of(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:          # pragma: no cover
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:      # pragma: no cover — vanished mid-walk
                pass
    return total


def _entity_dirname(entity_id: str | None) -> str | None:
    return entity_id.replace(":", "_") if entity_id else None


def _bundle_dir(artifact_path: str | None, model_id: str,
                entity_id: str | None) -> Path | None:
    """The per-model directory holding ``bundle.pkl`` and its sidecars."""
    if artifact_path:
        path = Path(artifact_path)
        candidate = path if path.is_dir() else path.parent
        if candidate.name == model_id:
            return candidate
        if candidate.exists():
            return candidate if path.is_dir() else path
    entity_dir = _entity_dirname(entity_id)
    if entity_dir:
        return Path(C.PER_ENTITY_MODEL_DIR) / entity_dir / model_id
    return None


def _add(purge: ModelPurge, path: Path | None, kind: str,
         seen: set[Path]) -> None:
    if path is None:
        return
    try:
        resolved = path.resolve()
    except OSError:              # pragma: no cover
        return
    if resolved in seen or not path.exists() or not _is_safe(path):
        return
    seen.add(resolved)
    purge.targets.append(
        PurgeTarget(path=path, kind=kind, model_id=purge.model_id,
                    bytes=_size_of(path))
    )


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def _cutoff(grace_days: int, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return (now - timedelta(days=max(grace_days, 0))).isoformat(
        timespec="milliseconds")


def _retired_models(conn, grace_days: int, now=None,
                    model_ids: list[str] | None = None) -> list[dict]:
    """Retired, not already purged, retired long enough ago to be cold."""
    sql = """
        SELECT m.model_id, m.entity_id, m.job_id, m.upload_id, m.artifact_path,
               COALESCE(
                   (SELECT MAX(b.retired_at) FROM entity_model_bindings b
                     WHERE b.model_id = m.model_id AND b.retired_at IS NOT NULL),
                   m.trained_at
               ) AS retired_at
          FROM model_registry m
         WHERE m.status = 'retired'
           AND (m.purged_at IS NULL OR m.purged_at = '')
    """
    params: list = []
    if model_ids:
        sql += f" AND m.model_id IN ({','.join('?' * len(model_ids))})"
        params.extend(model_ids)
    rows = [dict(r) for r in conn.execute(sql, params)]
    cutoff = _cutoff(grace_days, now)
    return [r for r in rows if not r["retired_at"] or r["retired_at"] <= cutoff]


def _job_is_shared(conn, job_id: str, model_id: str) -> bool:
    """True when a live model or an unfinished job still needs the job tree."""
    if not job_id:
        return True
    live_model = conn.execute(
        "SELECT 1 FROM model_registry "
        " WHERE job_id = ? AND model_id != ? AND status != 'retired' LIMIT 1",
        (job_id, model_id),
    ).fetchone()
    if live_model:
        return True
    running = conn.execute(
        f"SELECT 1 FROM training_jobs WHERE job_id = ? AND status IN "
        f"({','.join('?' * len(LIVE_JOB_STATES))}) LIMIT 1",
        (job_id, *LIVE_JOB_STATES),
    ).fetchone()
    return running is not None


def _upload_is_shared(conn, upload_id: str) -> bool:
    """True when any non-retired model or unfinished job still needs the CSV."""
    if not upload_id:
        return True
    live_model = conn.execute(
        "SELECT 1 FROM model_registry "
        " WHERE upload_id = ? AND status != 'retired' LIMIT 1",
        (upload_id,),
    ).fetchone()
    if live_model:
        return True
    running = conn.execute(
        f"SELECT 1 FROM training_jobs WHERE upload_id = ? AND status IN "
        f"({','.join('?' * len(LIVE_JOB_STATES))}) LIMIT 1",
        (upload_id, *LIVE_JOB_STATES),
    ).fetchone()
    return running is not None


def plan_purge(conn, grace_days: int = DEFAULT_GRACE_DAYS, now=None,
               model_ids: list[str] | None = None) -> list[ModelPurge]:
    """Resolve every deletable path for each cold retired model."""
    seen: set[Path] = set()
    plans: list[ModelPurge] = []
    uploads_claimed: set[str] = set()

    for row in _retired_models(conn, grace_days, now, model_ids):
        purge = ModelPurge(
            model_id=row["model_id"], entity_id=row["entity_id"],
            job_id=row["job_id"], upload_id=row["upload_id"],
            retired_at=row["retired_at"],
        )

        _add(purge, _bundle_dir(row["artifact_path"], row["model_id"],
                                row["entity_id"]), "model_bundle", seen)

        # Training artifacts: calibration, backtests and the training log live
        # under logs/training/<job_id>/<entity>. One job can train several
        # entities, so only this entity's subtree goes here.
        job_id = row["job_id"]
        if job_id and not _job_is_shared(conn, job_id, row["model_id"]):
            job_dir = Path(C.TRAINING_ARTIFACT_DIR) / job_id
            entity_dir = _entity_dirname(row["entity_id"])
            target = job_dir / entity_dir if entity_dir else job_dir
            _add(purge, target if target.exists() else job_dir,
                 "training_artifacts", seen)

        # Per-request scoring CSVs written while this model was champion.
        for art in conn.execute(
            "SELECT artifact_path FROM scoring_artifacts WHERE model_id = ?",
            (row["model_id"],),
        ):
            if art["artifact_path"]:
                _add(purge, Path(art["artifact_path"]), "score_artifact", seen)

        upload_id = row["upload_id"]
        if (upload_id and upload_id not in uploads_claimed
                and not _upload_is_shared(conn, upload_id)):
            uploads_claimed.add(upload_id)
            stored = conn.execute(
                "SELECT stored_path FROM dataset_uploads WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            upload_dir = Path(C.UPLOAD_DIR) / upload_id
            if not upload_dir.exists() and stored and stored["stored_path"]:
                upload_dir = Path(stored["stored_path"]).parent
            _add(purge, upload_dir, "upload", seen)

        plans.append(purge)
    return plans


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def _delete(path: Path) -> bool:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except OSError as exc:       # pragma: no cover — permissions / races
        LOGGER.warning("Could not delete %s: %s", path, exc)
        return False


def _prune_empty_parents(path: Path) -> None:
    """Remove directories left empty by the deletion, stopping at the roots."""
    roots = _safe_roots()
    parent = path.parent
    while parent.is_dir() and parent.resolve() not in roots:
        if any(p.name != ".DS_Store" for p in parent.iterdir()):
            return
        for leftover in parent.iterdir():
            leftover.unlink(missing_ok=True)
        try:
            parent.rmdir()
        except OSError:          # pragma: no cover
            return
        parent = parent.parent


# ---------------------------------------------------------------------------
# database rows
# ---------------------------------------------------------------------------

def _table_exists(conn, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone() is not None


def _sweep(conn, table: str, where: str, params, apply: bool,
           counts: dict[str, int]) -> None:
    """Count matching rows, and delete them when *apply*."""
    if not _table_exists(conn, table):
        return
    n = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {where}", params).fetchone()[0]
    if not n:
        return
    if apply:
        conn.execute(f"DELETE FROM {table} WHERE {where}", params)
    counts[table] = counts.get(table, 0) + n


def _is_retired(conn, model_id: str) -> bool:
    """Final guard: never let a statement run against a live model."""
    row = conn.execute(
        "SELECT status FROM model_registry WHERE model_id = ?",
        (model_id,)).fetchone()
    return row is not None and row["status"] == "retired"


def _delete_model_rows(conn, purge: "ModelPurge", apply: bool) -> dict[str, int]:
    """Dependent rows of one retired model. ``model_registry`` is kept."""
    counts: dict[str, int] = {}
    if not _is_retired(conn, purge.model_id):
        LOGGER.warning("Refusing to purge rows of non-retired model %s",
                       purge.model_id)
        return counts

    for table, column in MODEL_SCOPED_TABLES:
        _sweep(conn, table, f"{column} = ?", (purge.model_id,), apply, counts)

    if purge.job_id and not _job_is_shared(conn, purge.job_id, purge.model_id):
        for table, column in JOB_SCOPED_TABLES:
            _sweep(conn, table, f"{column} = ?", (purge.job_id,), apply, counts)

    if purge.upload_id and not _upload_is_shared(conn, purge.upload_id):
        for table, column in UPLOAD_SCOPED_TABLES:
            _sweep(conn, table, f"{column} = ?", (purge.upload_id,), apply,
                   counts)

    return counts


def _merge(into: dict[str, int], other: dict[str, int]) -> None:
    for key, value in other.items():
        into[key] = into.get(key, 0) + value


# ---------------------------------------------------------------------------
# operational log housekeeping (independent of model status)
# ---------------------------------------------------------------------------

def purge_operational_logs(grace_days: int = DEFAULT_GRACE_DAYS,
                           apply: bool = False, path=None, now=None) -> dict:
    """Drop redundant operational records older than the cutoff.

    Three categories, none of which carries information that is not already
    held elsewhere:

    * **score CSVs** — one immutable file per decision under ``logs/scores``.
      Every column is a ``scoring_results`` row, so past the cutoff the file is
      a duplicate. The ``scoring_artifacts`` pointer row goes with it rather
      than being left dangling. Age is the ``decision_time``: in production
      that is the same moment as ``created_at`` — a decision is written when it
      is made — and the two diverge only for a backfill or an experiment
      replay, where it is still the age of the decision that makes the file
      stale.
    * **delivered outbox events** — ``sent`` rows the receiver has already
      acknowledged, and ``dead_letter`` rows describing a retired model, which
      can never be retried into anything meaningful. ``pending`` and
      ``sending`` are never touched, at any age, for any model.
    * **api_requests** — the HTTP traffic log.
    """
    conn = connect(path)
    cutoff = _cutoff(grace_days, now)
    counts: dict[str, int] = {}
    freed = 0
    files = 0

    stale = "COALESCE(decision_time, created_at) <= ?"
    for row in conn.execute(
        f"SELECT artifact_path FROM scoring_artifacts WHERE {stale}",
        (cutoff,),
    ).fetchall():
        artifact = row["artifact_path"]
        if not artifact:
            continue
        candidate = Path(artifact)
        if not candidate.exists() or not _is_safe(candidate):
            continue
        size = _size_of(candidate)
        if apply:
            if not _delete(candidate):
                continue
            _prune_empty_parents(candidate)
        freed += size
        files += 1

    _sweep(conn, "scoring_artifacts", stale, (cutoff,), apply, counts)

    if apply:
        conn.commit()

    orphan_files, orphan_bytes = _sweep_orphan_score_files(conn, cutoff, apply)
    files += orphan_files
    freed += orphan_bytes

    _sweep(conn, "result_outbox",
           "status = 'sent' AND created_at IS NOT NULL AND created_at <= ?",
           (cutoff,), apply, counts)

    _sweep(conn, "result_outbox",
           "status = 'dead_letter' AND created_at IS NOT NULL "
           " AND created_at <= ? AND model_id IN "
           " (SELECT model_id FROM model_registry WHERE status = 'retired')",
           (cutoff,), apply, counts)

    _sweep(conn, "api_requests",
           "created_at IS NOT NULL AND created_at <= ?", (cutoff,), apply,
           counts)

    if apply:
        conn.commit()

    return {
        "applied": apply,
        "cutoff": cutoff,
        "rows": counts,
        "rows_total": sum(counts.values()),
        "score_files": files,
        "bytes": freed,
        "mb": round(freed / 1_048_576, 2),
    }


def _sweep_orphan_score_files(conn, cutoff: str,
                              apply: bool) -> tuple[int, int]:
    """Score CSVs under a day directory older than the cutoff with no row.

    ``logs/scores/<entity>/<YYYY-MM-DD>/`` is written by the scoring route; a
    file whose ``scoring_artifacts`` row is already gone cannot be located
    through the API and is pure residue.
    """
    root = Path(C.SCORE_ARTIFACT_DIR)
    if not root.is_dir():
        return 0, 0
    day_cutoff = cutoff[:10]
    files = 0
    freed = 0
    for day_dir in sorted(root.glob("*/*")):
        if not day_dir.is_dir() or day_dir.name > day_cutoff:
            continue
        for csv in day_dir.glob("*.csv"):
            if not _is_safe(csv):
                continue
            referenced = conn.execute(
                "SELECT 1 FROM scoring_artifacts WHERE artifact_path = ? LIMIT 1",
                (str(csv),)).fetchone()
            if referenced:
                continue
            size = _size_of(csv)
            if apply:
                if not _delete(csv):
                    continue
                _prune_empty_parents(csv)
            files += 1
            freed += size
    return files, freed


def backfill_delivered(conn, apply: bool = True) -> int:
    """Copy delivery proof from surviving outbox rows onto the decisions.

    The outbox row that proves a decision reached the receiver is itself
    eventually purged, so the proof has to live on the decision. Rows written
    before ``scoring_results.delivered_at`` existed are stamped here from any
    ``sent`` event still present.
    """
    sql = (
        "UPDATE scoring_results SET delivered_at = ("
        "  SELECT COALESCE(o.sent_at, o.created_at) FROM result_outbox o"
        "   WHERE o.event_type = 'scoring.recorded' AND o.status = 'sent'"
        "     AND o.event_id = 'score:' || scoring_results.entity_id"
        "         || ':' || scoring_results.decision_time"
        "         || ':' || scoring_results.model_id || ':v' || ?)"
        " WHERE delivered_at IS NULL AND EXISTS ("
        "  SELECT 1 FROM result_outbox o"
        "   WHERE o.event_type = 'scoring.recorded' AND o.status = 'sent'"
        "     AND o.event_id = 'score:' || scoring_results.entity_id"
        "         || ':' || scoring_results.decision_time"
        "         || ':' || scoring_results.model_id || ':v' || ?)"
    )
    version = getattr(C, "EVENT_SCHEMA_VERSION", 1)
    if not apply:
        return 0
    cur = conn.execute(sql, (version, version))
    if cur.rowcount:
        conn.commit()
        LOGGER.info("Stamped %d decision(s) as delivered", cur.rowcount)
    return cur.rowcount or 0


def purge_delivered_decisions(grace_days: int = DEFAULT_GRACE_DAYS,
                              apply: bool = False, path=None, now=None,
                              force_undelivered: bool = False) -> dict:
    """Delete decisions and replenishment orders the receiver already holds.

    ``scoring_results`` and ``replenishment_orders`` grow without bound — one
    row per record per entity, forever — and both are forwarded to the result
    receiver, which is their system of record. Once a decision is acknowledged
    there and older than the cutoff, the local copy is redundant.

    A decision is deleted only when ``delivered_at`` is set. An event still
    ``pending``, ``sending`` or ``dead_letter`` has *not* reached the other
    server, so its decision stays — deleting it would destroy the only
    remaining copy. ``force_undelivered`` overrides that and loses data.

    A ``PENDING`` replenishment order is open operational work — somebody still
    has to move those carts — so it is kept regardless of age.
    """
    conn = connect(path)
    cutoff = _cutoff(grace_days, now)
    counts: dict[str, int] = {}

    backfill_delivered(conn, apply=apply)

    # decision_time is ISO with a 'T' in scoring_results and a space in
    # replenishment_orders; normalise before comparing against the cutoff.
    decision_age = "replace(decision_time, ' ', 'T') <= ?"
    delivered = "delivered_at IS NOT NULL"

    where = decision_age if force_undelivered else f"{decision_age} AND {delivered}"
    _sweep(conn, "scoring_results", where, (cutoff,), apply, counts)

    # An order rides along inside its decision's event, so the decision's
    # delivery is the order's delivery. An undelivered decision still present
    # therefore pins its orders.
    order_where = (
        f"status <> 'PENDING' AND {decision_age}"
        " AND NOT EXISTS (SELECT 1 FROM scoring_results s"
        "   WHERE s.entity_id = replenishment_orders.entity_id"
        "     AND replace(s.decision_time, ' ', 'T')"
        "         = replace(replenishment_orders.decision_time, ' ', 'T')"
        "     AND s.delivered_at IS NULL)"
    )
    if force_undelivered:
        order_where = f"status <> 'PENDING' AND {decision_age}"
    _sweep(conn, "replenishment_orders", order_where, (cutoff,), apply, counts)

    if apply:
        conn.commit()

    undelivered = conn.execute(
        f"SELECT COUNT(*) FROM scoring_results WHERE {decision_age}"
        "   AND delivered_at IS NULL", (cutoff,)).fetchone()[0]
    if undelivered:
        LOGGER.warning("%d decision(s) older than the cutoff are kept: the "
                       "receiver has not acknowledged them", undelivered)

    return {
        "applied": apply,
        "cutoff": cutoff,
        "rows": counts,
        "rows_total": sum(counts.values()),
        "kept_undelivered": undelivered,
    }


def drop_orphan_rows(apply: bool = False, path=None) -> dict:
    """Rows pointing at a ``model_id`` that no longer exists in the registry."""
    conn = connect(path)
    counts: dict[str, int] = {}
    for table, column in MODEL_SCOPED_TABLES:
        _sweep(conn, table,
               f"{column} IS NOT NULL AND {column} NOT IN "
               f"(SELECT model_id FROM model_registry)", (), apply, counts)
    if apply:
        conn.commit()
    return {"rows": counts, "rows_total": sum(counts.values())}


def vacuum(path=None) -> int:
    """Rebuild the database so deleted pages are returned to the filesystem.

    SQLite needs an exclusive lock to rebuild. A running server holds the file,
    so a lock failure is logged and reported as zero reclaimed rather than
    raised: the rows are already deleted, and the next run reclaims the pages.
    """
    db_path = Path(path or C.DB_PATH)
    before = db_path.stat().st_size if db_path.exists() else 0
    conn = connect(path)
    try:
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
    except sqlite3.OperationalError as exc:
        LOGGER.warning("VACUUM skipped (%s) — space stays inside %s until the "
                       "next run can take an exclusive lock", exc, db_path.name)
        return 0
    # In WAL mode the rebuilt database lands in the write-ahead log first, so
    # the main file only shrinks once it is checkpointed — without this the
    # reclaimed size reads as zero even though the rebuild succeeded.
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError as exc:   # pragma: no cover — busy writer
        LOGGER.debug("Checkpoint deferred: %s", exc)
    after = db_path.stat().st_size if db_path.exists() else 0
    freed = max(before - after, 0)
    LOGGER.info("VACUUM freed %.2f MB", freed / 1_048_576)
    return freed


def purge_retired(grace_days: int = DEFAULT_GRACE_DAYS, apply: bool = False,
                  path=None, now=None, model_ids: list[str] | None = None,
                  delete_rows: bool = True) -> dict:
    """Purge artifacts of retired models. Dry-run unless *apply*.

    Files always; dependent database rows too unless *delete_rows* is False.
    The ``model_registry`` row itself is always kept and stamped.
    """
    conn = connect(path)
    plans = plan_purge(conn, grace_days=grace_days, now=now,
                       model_ids=model_ids)

    deleted_bytes = 0
    deleted_paths = 0
    row_counts: dict[str, int] = {}
    purged_models: list[dict] = []
    stamp = utc_now()

    for purge in plans:
        model_bytes = 0
        model_paths = 0
        for target in purge.targets:
            if not apply:
                model_bytes += target.bytes
                model_paths += 1
                continue
            if not _delete(target.path):
                continue
            _prune_empty_parents(target.path)
            model_bytes += target.bytes
            model_paths += 1
            conn.execute(
                "INSERT INTO artifact_purges (model_id, entity_id, job_id, "
                "upload_id, kind, path, bytes, purged_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (purge.model_id, purge.entity_id, purge.job_id,
                 purge.upload_id, target.kind, str(target.path),
                 target.bytes, stamp),
            )
        model_rows: dict[str, int] = {}
        if delete_rows:
            # Row deletion runs before the registry stamp, because the stamp
            # is what excludes this model from the next run's plan.
            model_rows = _delete_model_rows(conn, purge, apply)
            _merge(row_counts, model_rows)
        if apply:
            conn.execute(
                "UPDATE model_registry SET purged_at = ?, purged_bytes = ? "
                " WHERE model_id = ?",
                (stamp, model_bytes, purge.model_id),
            )
        deleted_bytes += model_bytes
        deleted_paths += model_paths
        purged_models.append({
            "model_id": purge.model_id,
            "entity_id": purge.entity_id,
            "retired_at": purge.retired_at,
            "bytes": model_bytes,
            "rows": model_rows,
            "paths": [{"kind": t.kind, "path": str(t.path), "bytes": t.bytes}
                      for t in purge.targets],
        })

    if apply:
        conn.commit()

    report = {
        "applied": apply,
        "grace_days": grace_days,
        "ran_at": stamp,
        "models": len(plans),
        "paths": deleted_paths,
        "bytes": deleted_bytes,
        "mb": round(deleted_bytes / 1_048_576, 2),
        "rows": row_counts,
        "rows_total": sum(row_counts.values()),
        "detail": purged_models,
    }
    LOGGER.info("Retired-model purge %s: %d model(s), %d path(s), %.2f MB, "
                "%d row(s)", "applied" if apply else "dry-run",
                report["models"], report["paths"], report["mb"],
                report["rows_total"])
    return report


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_retention(grace_days: int = DEFAULT_GRACE_DAYS, apply: bool = False,
                  path=None, now=None, model_ids: list[str] | None = None,
                  delete_rows: bool = True, housekeeping: bool = True,
                  orphans: bool = True, do_vacuum: bool = True,
                  decisions: bool = True,
                  force_undelivered: bool = False) -> dict:
    """Full retention sweep: retired models, decisions, operational logs.

    The decision pass runs **before** the operational-log pass, because the
    ``sent`` outbox rows it reads as delivery proof are what that pass deletes.

    ``VACUUM`` runs last and only when something was actually deleted — SQLite
    keeps freed pages in the file otherwise, so the space is not returned to
    the filesystem without it.
    """
    models = purge_retired(grace_days=grace_days, apply=apply, path=path,
                           now=now, model_ids=model_ids,
                           delete_rows=delete_rows)

    wide = not model_ids

    decided = (purge_delivered_decisions(grace_days=grace_days, apply=apply,
                                         path=path, now=now,
                                         force_undelivered=force_undelivered)
               if decisions and wide else None)

    logs = (purge_operational_logs(grace_days=grace_days, apply=apply,
                                   path=path, now=now)
            if housekeeping and wide else None)

    orphaned = (drop_orphan_rows(apply=apply, path=path)
                if orphans and wide else None)

    rows: dict[str, int] = {}
    _merge(rows, models["rows"])
    for extra in (decided, logs, orphaned):
        if extra:
            _merge(rows, extra["rows"])

    file_bytes = models["bytes"] + (logs["bytes"] if logs else 0)
    rows_total = sum(rows.values())

    freed_db = 0
    if apply and do_vacuum and (rows_total or models["paths"]):
        freed_db = vacuum(path)

    report = {
        "applied": apply,
        "grace_days": grace_days,
        "ran_at": models["ran_at"],
        "models": models["models"],
        "paths": models["paths"] + (logs["score_files"] if logs else 0),
        "file_bytes": file_bytes,
        "file_mb": round(file_bytes / 1_048_576, 2),
        "db_bytes_reclaimed": freed_db,
        "db_mb_reclaimed": round(freed_db / 1_048_576, 2),
        "total_mb": round((file_bytes + freed_db) / 1_048_576, 2),
        "rows": rows,
        "rows_total": rows_total,
        "kept_undelivered": decided["kept_undelivered"] if decided else 0,
        "retired_models": models,
        "decisions": decided,
        "operational_logs": logs,
        "orphans": orphaned,
    }
    LOGGER.info("Retention %s: %d retired model(s), %d file(s) %.2f MB, "
                "%d row(s), %.2f MB reclaimed from the database",
                "applied" if apply else "dry-run", report["models"],
                report["paths"], report["file_mb"], rows_total,
                report["db_mb_reclaimed"])
    return report
