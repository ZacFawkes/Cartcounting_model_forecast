"""Empty every SQLite database while keeping its structure.

Destructive. Nothing is touched until ``--yes`` is passed.

    python scripts/reset_databases.py                 # show what would go
    python scripts/reset_databases.py --yes           # reset both databases
    python scripts/reset_databases.py --db platform --yes
    python scripts/reset_databases.py --yes --with-artifacts

Two databases are in scope:

* ``data/platform.db``        — registry, uploads, training, scoring, outbox
* ``data/results_receiver.db`` — the receiver's immutable event log

Structure is rebuilt by each module's own ``init_db()``, which runs its
``schema.sql`` and then its additive column migrations, so the result is
exactly the schema the code expects — not a copy that can drift.

Two modes:

``recreate`` (default)
    Delete the file and rebuild it from ``schema.sql``. A genuinely fresh
    database. The table and column inventory is captured first and compared
    afterwards; anything that existed before and is not recreated aborts the
    run unless ``--force`` is given, so hand-made structure is never lost
    silently.

``truncate``
    Keep the file and delete every row, then reset AUTOINCREMENT counters.
    Structure cannot change at all. Use this when the database has columns the
    schema files do not define.

``--with-artifacts`` additionally empties ``models/``, ``logs/`` and
``data/uploads/``, because a reset registry cannot explain bundles and score
CSVs that are still on disk. Without it those files are left alone and become
orphans.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import settings as C  # noqa: E402

#: Directories emptied by --with-artifacts, and recreated empty afterwards.
ARTIFACT_DIRS = ("MODEL_DIR", "UPLOAD_DIR", "LOG_DIR")

#: Files SQLite keeps beside the database in WAL mode.
SIDECARS = ("-wal", "-shm", "-journal")


def _targets(which: str) -> list[tuple[str, Path]]:
    all_dbs = [("platform", Path(C.DB_PATH)),
               ("receiver", Path(C.RECEIVER_DB_PATH))]
    return [t for t in all_dbs if which in ("all", t[0])]


def _inventory(path: Path) -> dict[str, set[str]]:
    """``{table: {column, ...}}`` for every user table in the database."""
    if not path.exists():
        return {}
    conn = sqlite3.connect(str(path))
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            " AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
                for t in tables}
    finally:
        conn.close()


def _row_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    conn = sqlite3.connect(str(path))
    try:
        counts = {}
        for table in _inventory(path):
            counts[table] = conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return counts
    finally:
        conn.close()


def _in_use(path: Path) -> bool:
    """True when another process holds a write lock on the database."""
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(str(path), timeout=1.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.rollback()
            return False
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return True


def _backup(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    target = path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")
    source = sqlite3.connect(str(path))
    try:
        # The online backup API copies a consistent snapshot including the WAL,
        # which a plain file copy would miss.
        dest = sqlite3.connect(str(target))
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    return target


def _init(name: str) -> None:
    """Rebuild structure through the module that owns the database."""
    if name == "platform":
        from registry import database as pdb
        pdb.reset_connection()
        pdb.init_db()
        pdb.reset_connection()
    else:
        from receiver import database as rdb
        rdb.reset_connection()
        rdb.configure(C.RECEIVER_DB_PATH)
        rdb.init_db()
        rdb.reset_connection()


def _recreate(name: str, path: Path) -> None:
    # Cached handles must go before the file does: SQLite keeps writing to an
    # unlinked inode, so a surviving handle would silently bypass the new file.
    _drop_cached_handles(name)
    for suffix in SIDECARS:
        Path(str(path) + suffix).unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    _init(name)


def _drop_cached_handles(name: str) -> None:
    if name == "platform":
        from registry import database as pdb
        pdb.reset_connection()
    else:
        from receiver import database as rdb
        rdb.reset_connection()


def _truncate(name: str, path: Path) -> None:
    _init(name)
    _drop_cached_handles(name)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            " AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        # Restart AUTOINCREMENT ids so a reset database really starts at 1.
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                        " AND name='sqlite_sequence'").fetchone():
            conn.execute("DELETE FROM sqlite_sequence")
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()


def _clear_artifacts() -> list[str]:
    cleared = []
    for attr in ARTIFACT_DIRS:
        root = Path(getattr(C, attr))
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        cleared.append(str(root))
    # settings recreates the standard subdirectories on import; do it here too
    # so the tree is immediately usable without a server restart.
    for attr in ("RUN_DIR", "SCORE_ARTIFACT_DIR", "TRAINING_ARTIFACT_DIR",
                 "PER_ENTITY_MODEL_DIR", "UPLOAD_DIR"):
        Path(getattr(C, attr)).mkdir(parents=True, exist_ok=True)
    return cleared


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", choices=("all", "platform", "receiver"),
                        default="all", help="which database to reset")
    parser.add_argument("--mode", choices=("recreate", "truncate"),
                        default="recreate",
                        help="rebuild the file from schema.sql (default) or "
                             "delete every row in place")
    parser.add_argument("--yes", action="store_true",
                        help="actually do it (default: dry-run)")
    parser.add_argument("--no-backup", action="store_true",
                        help="skip the timestamped .bak snapshot")
    parser.add_argument("--with-artifacts", action="store_true",
                        help="also empty models/, logs/ and data/uploads/")
    parser.add_argument("--force", action="store_true",
                        help="proceed despite a live writer or lost structure")
    args = parser.parse_args(argv)

    targets = _targets(args.db)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    before = {name: _inventory(path) for name, path in targets}

    print(f"\n{'RESETTING' if args.yes else 'WOULD RESET'} "
          f"({args.mode}):")
    busy = []
    for name, path in targets:
        counts = _row_counts(path)
        size = path.stat().st_size / 1_048_576 if path.exists() else 0
        state = "missing" if not path.exists() else f"{size:.1f} MB"
        print(f"\n  {name}: {path}  [{state}]")
        for table, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            if n:
                print(f"      {table:<26} {n:>10,} rows")
        if not any(counts.values()):
            print("      (already empty)")
        if _in_use(path):
            busy.append(name)

    if busy:
        print(f"\n  WARNING: another process is writing to: {', '.join(busy)}")
        print("  Stop the server(s) first — a reset under a live writer "
              "leaves that process holding handles to a database that no "
              "longer exists.")
        if not args.force:
            print("  Refusing. Re-run with --force to override.")
            return 1

    if args.with_artifacts:
        print(f"\n  artifacts: will empty "
              f"{', '.join(str(Path(getattr(C, a))) for a in ARTIFACT_DIRS)}")

    if not args.yes:
        print("\nRe-run with --yes to apply.\n")
        return 0

    for name, path in targets:
        if not args.no_backup and path.exists():
            backup = _backup(path, stamp)
            print(f"\n  {name}: backed up to {backup.name}")
        if args.mode == "recreate":
            _recreate(name, path)
        else:
            _truncate(name, path)

        after = _inventory(path)
        missing = {t: cols for t, cols in before[name].items()
                   if t not in after}
        lost_columns = {t: sorted(cols - after.get(t, set()))
                        for t, cols in before[name].items()
                        if t in after and cols - after[t]}
        if (missing or lost_columns) and not args.force:
            print(f"  ERROR: structure would be lost in {name}: "
                  f"tables={sorted(missing)} columns={lost_columns}")
            print("  The backup above still holds the original. Re-run with "
                  "--mode truncate to keep the structure exactly, or --force "
                  "to accept the schema.sql definition.")
            return 1
        if missing or lost_columns:
            print(f"  WARNING: dropped tables={sorted(missing)} "
                  f"columns={lost_columns}")
        print(f"  {name}: {len(after)} table(s), "
              f"{sum(_row_counts(path).values())} row(s)")

    if args.with_artifacts:
        for root in _clear_artifacts():
            print(f"  emptied {root}")

    print("\nDone. Structure preserved, data cleared.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
