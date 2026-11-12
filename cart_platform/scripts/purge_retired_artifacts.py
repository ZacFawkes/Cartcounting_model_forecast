"""Reclaim space: delete artifacts and records that are no longer needed.

Dry-run by default — nothing is deleted until ``--apply`` is passed.

    python scripts/purge_retired_artifacts.py                 # show the plan
    python scripts/purge_retired_artifacts.py --apply         # free the space
    python scripts/purge_retired_artifacts.py --grace-days 30 # keep longer

Three passes, all gated on the same age cutoff (default one week):

1. retired models — bundle, training artifacts, unshared source upload, and
   their dependent database rows;
2. decisions — ``scoring_results`` and ``replenishment_orders`` the result
   receiver has already acknowledged;
3. operational logs — per-decision score CSVs, delivered outbox events, the
   api_requests log; plus orphan rows pointing at a model that is gone.

Champion and candidate models are protected: their artifacts, registry entries,
calibration, backtests and bindings are never touched. The ``model_registry``
row of a retired model is kept too, stamped with ``purged_at``. A decision the
receiver has not acknowledged is never deleted, whatever its age.

Scheduled fortnightly by ``scripts/install_purge_schedule.sh``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import settings as C                                      # noqa: E402
from registry.database import init_db                     # noqa: E402
from registry.retention import DEFAULT_GRACE_DAYS, run_retention  # noqa: E402


def _configure_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _print_human(report: dict, apply: bool) -> None:
    mode = "DELETED" if apply else "WOULD DELETE"
    print(f"\n{mode}: {report['models']} retired model(s), "
          f"{report['paths']} file(s) / {report['file_mb']} MB, "
          f"{report['rows_total']} database row(s)")

    for model in report["retired_models"]["detail"]:
        if not model["paths"] and not model["rows"]:
            continue
        print(f"  {model['model_id']}  "
              f"({round(model['bytes'] / 1_048_576, 2)} MB, "
              f"{sum(model['rows'].values())} rows)")
        for path in model["paths"]:
            print(f"      [{path['kind']}] {path['path']}")

    logs = report.get("operational_logs")
    if logs:
        print(f"\n  operational logs older than {logs['cutoff'][:10]}: "
              f"{logs['score_files']} score CSV(s) / {logs['mb']} MB")

    if report.get("kept_undelivered"):
        print(f"\n  KEPT {report['kept_undelivered']:,} decision(s) older than "
              f"the cutoff: the receiver has not acknowledged them, so this is "
              f"the only copy. Drain the outbox and they go on the next run.")

    if report["rows"]:
        print("\n  rows by table:")
        for table, n in sorted(report["rows"].items(), key=lambda kv: -kv[1]):
            print(f"      {table:<24} {n:>9,}")

    if apply:
        print(f"\n  database reclaimed: {report['db_mb_reclaimed']} MB")
        print(f"  total freed:        {report['total_mb']} MB")
    elif report["paths"] or report["rows_total"]:
        print("\nRe-run with --apply to delete.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="actually delete (default: dry-run)")
    parser.add_argument("--grace-days", type=int, default=DEFAULT_GRACE_DAYS,
                        help=f"keep anything newer than this many days "
                             f"(default {DEFAULT_GRACE_DAYS})")
    parser.add_argument("--model-id", action="append", dest="model_ids",
                        help="restrict to this model id; skips the "
                             "operational-log passes (repeatable)")
    parser.add_argument("--files-only", action="store_true",
                        help="delete artifacts but keep every database row")
    parser.add_argument("--skip-housekeeping", action="store_true",
                        help="skip score CSVs, outbox and api_requests")
    parser.add_argument("--keep-decisions", action="store_true",
                        help="keep scoring_results and replenishment_orders "
                             "even once the receiver has acknowledged them")
    parser.add_argument("--force-undelivered", action="store_true",
                        help="DANGER: delete decisions the receiver has not "
                             "acknowledged — that data exists nowhere else")
    parser.add_argument("--no-vacuum", action="store_true",
                        help="skip VACUUM (space stays inside platform.db)")
    parser.add_argument("--json", action="store_true",
                        help="print the full report as JSON")
    parser.add_argument("--log-file", default=str(C.LOG_DIR / "retention"
                                                  / "purge.log"))
    args = parser.parse_args(argv)

    _configure_logging(Path(args.log_file) if args.log_file else None)
    init_db()

    report = run_retention(
        grace_days=args.grace_days,
        apply=args.apply,
        model_ids=args.model_ids,
        delete_rows=not args.files_only,
        housekeeping=not args.skip_housekeeping and not args.files_only,
        orphans=not args.files_only,
        do_vacuum=not args.no_vacuum,
        decisions=not args.keep_decisions and not args.files_only,
        force_undelivered=args.force_undelivered,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
