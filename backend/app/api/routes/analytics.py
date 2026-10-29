"""T13-T15: PCA and adult weight analytics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services import dimensionality, history, lifestyle
from app.services.ehr_data import list_sources

router = APIRouter(prefix="/api/analytics", tags=["advanced analytics"])


@router.get("/pca/sources")
def pca_sources(db: Session = Depends(get_db)) -> list[dict]:
    sources = [{"id": k, "label": v} for k, v in dimensionality.PCA_SOURCES.items()]
    sources += [{"id": s["id"], "label": s["label"]} for s in list_sources(db) if s["id"].startswith("upload:")]
    return sources


@router.get("/pca")
def pca(source: str = "nhanes_adults", scaling: str = "standard", n_components: int = Query(5, ge=2, le=20),
        missing: str = "impute", color_by: str | None = None, columns: list[str] | None = Query(None),
        db: Session = Depends(get_db)) -> dict:
    df, color_options = dimensionality.load_pca_source(source, db)
    result = dimensionality.run_pca(df, color_options, scaling, n_components, missing, color_by, columns)
    history.record(db, "analytics", f"PCA on {source} ({scaling})",
                   {"components_for_80pct": result["components_for_80pct"], "features": len(result["features"])},
                   tutorial="T13", dataset=source)
    return result


@router.get("/weight/explore")
def weight_explore(variable: str = "bmi", db: Session = Depends(get_db)) -> dict:
    result = lifestyle.explore(variable)
    history.record(db, "analytics", "Adult weight exploration",
                   {"n_adults": result["n_adults"], "obesity": result["obesity_prevalence"]}, tutorial="T14",
                   dataset="nhanes")
    return result


@router.post("/weight/train")
def weight_train(db: Session = Depends(get_db)) -> dict:
    result = lifestyle.train_models()
    history.record(db, "analytics", "Trained adult weight models", result["summary"], tutorial="T15",
                   dataset="nhanes")
    return result


@router.get("/weight/results")
def weight_results() -> dict:
    return lifestyle.model_results() or {"trained": False}
