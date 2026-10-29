"""Record analysis runs so the dashboard and reports can show real history."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.utils import to_jsonable
from app.models.entities import AnalysisRun

DUPLICATE_WINDOW = timedelta(seconds=5)


def record(db: Session | None, module: str, title: str, summary: dict, tutorial: str | None = None,
           dataset: str | None = None) -> bool:
    """Store one analysis run. Returns False when an identical run was stored seconds ago.

    The duplicate guard stops double-mounted React components (StrictMode in development) or
    double clicks from inflating the per-tutorial run counts shown on the dashboard.
    """
    if db is None:
        return False
    since = datetime.now(timezone.utc) - DUPLICATE_WINDOW
    duplicate = db.execute(
        select(AnalysisRun.id).where(AnalysisRun.module == module, AnalysisRun.title == title,
                                     AnalysisRun.tutorial.is_(None) if tutorial is None
                                     else AnalysisRun.tutorial == tutorial,
                                     AnalysisRun.created_at >= since).limit(1)
    ).first()
    if duplicate:
        return False
    db.add(AnalysisRun(module=module, tutorial=tutorial, title=title, dataset=dataset,
                       summary=to_jsonable(summary)))
    db.commit()
    return True


def recent(db: Session, limit: int = 20) -> list[dict]:
    rows = db.execute(select(AnalysisRun).order_by(AnalysisRun.created_at.desc()).limit(limit)).scalars().all()
    return [
        {"id": r.id, "module": r.module, "tutorial": r.tutorial, "title": r.title, "dataset": r.dataset,
         "summary": r.summary, "created_at": r.created_at.isoformat()}
        for r in rows
    ]


def counts_by_tutorial(db: Session) -> dict[str, int]:
    rows = db.execute(select(AnalysisRun.tutorial, func.count()).group_by(AnalysisRun.tutorial)).all()
    return {t: n for t, n in rows if t}


def total(db: Session) -> int:
    return db.execute(select(func.count()).select_from(AnalysisRun)).scalar_one()
