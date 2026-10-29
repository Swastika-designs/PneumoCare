"""Health, dashboard status, history, registries and tutorials."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import NotFound
from app.db.database import check_db, get_db
from app.models.entities import Prediction
from app.schemas.api import HealthResponse
from app.services import history, registry, tutorials

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name, version=settings.version, database=check_db())


@router.get("/system/status")
def status(db: Session = Depends(get_db)) -> dict:
    datasets = registry.dataset_status()
    models = registry.model_status()
    run_counts = history.counts_by_tutorial(db)
    catalogue = tutorials.catalogue(run_counts)
    return {
        "database": check_db(),
        "datasets": {"total": len(datasets), "available": sum(d["available"] for d in datasets),
                     "required": sum(d["required"] for d in datasets),
                     "required_available": sum(d["available"] for d in datasets if d["required"]),
                     "items": [{k: d[k] for k in ("id", "name", "domain", "available", "synthetic", "required")}
                               for d in datasets]},
        "models": [{k: m[k] for k in ("id", "name", "status", "metrics", "updated_at")} for m in models],
        "tutorials": {"total": len(catalogue), "verified": sum(t["status"] == "verified" for t in catalogue),
                      "data_ready": sum(t["data_ready"] for t in catalogue),
                      "explored": sum(t["runs"] > 0 for t in catalogue),
                      "items": [{k: t[k] for k in ("id", "title", "module", "route", "data_ready", "runs", "status")}
                                for t in catalogue]},
        "history": {"total_runs": history.total(db), "recent": history.recent(db, 8),
                    "predictions": db.execute(select(func.count()).select_from(Prediction)).scalar_one()},
    }


@router.get("/history")
def get_history(limit: int = 50, db: Session = Depends(get_db)) -> list[dict]:
    return history.recent(db, min(max(limit, 1), 500))


@router.get("/datasets")
def datasets() -> list[dict]:
    return registry.dataset_status()


@router.get("/models")
def models() -> list[dict]:
    return registry.model_status()


@router.get("/tutorials")
def tutorial_list(db: Session = Depends(get_db)) -> list[dict]:
    return tutorials.catalogue(history.counts_by_tutorial(db))


@router.get("/tutorials/{tutorial_id}")
def tutorial_detail(tutorial_id: str, db: Session = Depends(get_db)) -> dict:
    for item in tutorials.catalogue(history.counts_by_tutorial(db)):
        if item["id"].lower() == tutorial_id.lower():
            return item
    raise NotFound(f"Tutorial '{tutorial_id}' does not exist (expected T1-T15).")
