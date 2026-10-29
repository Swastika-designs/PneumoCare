"""Correct misleading entries in the analysis history (one-off audit repair, safe to re-run).

1. Removes duplicate runs: same module/tutorial/title recorded within 5 seconds of the previous one.
   These came from React StrictMode running page-load effects twice in development.
2. Un-tags plain pneumonia predictions from T6. Before the fix every prediction was counted as a
   T6 run, although T6 (visualisation) was never executed. Only predictions made from the Image
   Laboratory's Visualise tab (context=T6) count as T6 now.

Usage: python scripts/repair_history.py [--dry-run]
"""
import argparse
from datetime import timedelta

import _bootstrap  # noqa: F401
from sqlalchemy import select

from app.db import database
from app.models.entities import AnalysisRun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    database.configure()
    database.init_db()
    db = database.SessionLocal()
    rows = db.execute(select(AnalysisRun).order_by(AnalysisRun.created_at)).scalars().all()
    last_seen, duplicates, untagged = {}, [], []
    for r in rows:
        key = (r.module, r.tutorial, r.title)
        prev = last_seen.get(key)
        if prev is not None and r.created_at - prev <= timedelta(seconds=5):
            duplicates.append(r)
            continue
        last_seen[key] = r.created_at
        if r.module == "screening" and r.tutorial == "T6":
            untagged.append(r)
    print(f"{len(rows)} runs; {len(duplicates)} duplicates to delete; {len(untagged)} screening runs to un-tag from T6")
    for r in duplicates:
        print(f"  delete #{r.id} {r.tutorial} {r.title} {r.created_at}")
    if args.dry_run:
        return
    for r in duplicates:
        db.delete(r)
    for r in untagged:
        r.tutorial = None
    db.commit()
    print("History repaired.")


if __name__ == "__main__":
    main()
