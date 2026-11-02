"""Delete the on-disk artifacts of retired models.

Dry-run by default — nothing is deleted until ``--apply`` is passed.

    python scripts/purge_retired_artifacts.py                 # show the plan
    python scripts/purge_retired_artifacts.py --apply         # free the space
    python scripts/purge_retired_artifacts.py --grace-days 30 # keep longer

Scope: models whose ``model_registry.status`` is ``retired`` and that were
retired at least ``--grace-days`` ago. Database rows are never deleted; the
``model_registry`` row is stamped with ``purged_at`` and every deleted path is
recorded in ``artifact_purges``.

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
from registry.retention import DEFAULT_GRACE_DAYS, purge_retired  # noqa: E402


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="actually delete (default: dry-run)")
    parser.add_argument("--grace-days", type=int, default=DEFAULT_GRACE_DAYS,
                        help=f"keep artifacts this long after retirement "
                             f"(default {DEFAULT_GRACE_DAYS})")
    parser.add_argument("--model-id", action="append", dest="model_ids",
                        help="restrict to this model id (repeatable)")
    parser.add_argument("--json", action="store_true",
                        help="print the full report as JSON")
    parser.add_argument("--log-file", default=str(C.LOG_DIR / "retention"
                                                  / "purge.log"))
    args = parser.parse_args(argv)

    _configure_logging(Path(args.log_file) if args.log_file else None)
    init_db()

    report = purge_retired(grace_days=args.grace_days, apply=args.apply,
                           model_ids=args.model_ids)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        mode = "DELETED" if args.apply else "WOULD DELETE"
        print(f"\n{mode}: {report['models']} retired model(s), "
              f"{report['paths']} path(s), {report['mb']} MB")
        for model in report["detail"]:
            if not model["paths"]:
                continue
            print(f"  {model['model_id']}  "
                  f"({round(model['bytes'] / 1_048_576, 2)} MB)")
            for path in model["paths"]:
                print(f"      [{path['kind']}] {path['path']}")
        if not args.apply and report["paths"]:
            print("\nRe-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
