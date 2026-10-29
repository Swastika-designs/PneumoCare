"""Report export endpoints (CSV / JSON / PDF)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.services import reports

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("")
def list_reports() -> list[dict]:
    return reports.available_reports()


@router.get("/{report_id}")
def export(report_id: str, format: str = Query("json", pattern="^(json|csv|pdf)$"),
           source: str | None = None, save: bool = False, db: Session = Depends(get_db)):
    params = {"source": source} if source else {}
    report = reports.build(report_id, db, **params)
    body, media_type, filename = reports.render(report, format)
    if save:
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        (settings.reports_dir / filename).write_bytes(body)
    return Response(content=body, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
