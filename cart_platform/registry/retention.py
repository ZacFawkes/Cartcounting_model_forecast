"""Disk retention for retired models.

A model whose ``model_registry.status`` is ``retired`` can never be scored
again: the resolver only ever binds a ``champion``, and a rollback promotes a
``candidate``. Its bundle, training logs, backtest and calibration artifacts
and — when nothing else references it — its source upload are therefore dead
weight on disk.

This module deletes **files only**. Every database row is kept so the audit
trail of what was trained, promoted and retired survives the purge; the
``model_registry`` row is stamped with ``purged_at`` / ``purged_bytes`` and one
row per deleted path is written to ``artifact_purges``.

Safety rules:

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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import settings as C
from registry.database import connect, utc_now

LOGGER = logging.getLogger("platform.retention")

DEFAULT_GRACE_DAYS = 14

#: Training jobs in these states may still write into their artifact tree.
LIVE_JOB_STATES = ("queued", "running")


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


def purge_retired(grace_days: int = DEFAULT_GRACE_DAYS, apply: bool = False,
                  path=None, now=None,
                  model_ids: list[str] | None = None) -> dict:
    """Delete on-disk artifacts of retired models. Dry-run unless *apply*.

    Returns a report suitable for logging or ``--json`` output.
    """
    conn = connect(path)
    plans = plan_purge(conn, grace_days=grace_days, now=now,
                       model_ids=model_ids)

    deleted_bytes = 0
    deleted_paths = 0
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
        "detail": purged_models,
    }
    LOGGER.info("Retention %s: %d model(s), %d path(s), %.2f MB",
                "applied" if apply else "dry-run",
                report["models"], report["paths"], report["mb"])
    return report
