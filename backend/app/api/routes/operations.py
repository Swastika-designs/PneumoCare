"""T7-T9: healthcare operations analytics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.api import ScenarioRequest
from app.services import history, operations

router = APIRouter(prefix="/api/operations", tags=["operations"])


@router.get("/forecast")
def forecast(freq: str = "month", encounter_class: str = "all", horizon: int = Query(6, ge=1, le=24),
             start_year: int = Query(2016, ge=1940, le=2030), hours_per_fte: float | None = Query(None, gt=0),
             utilization_target: float = Query(0.85, gt=0, le=1), db: Session = Depends(get_db)) -> dict:
    result = operations.forecast_volume(freq, encounter_class, horizon, start_year=start_year,
                                        hours_per_fte=hours_per_fte, utilization_target=utilization_target)
    history.record(db, "operations", f"Volume forecast ({freq}, {encounter_class})",
                   {"best_model": result["best_model"], "results": result["results"]}, tutorial="T7",
                   dataset="synthea:encounters")
    return result


@router.get("/kpis")
def kpis(start_year: int = Query(2016, ge=1940, le=2030), group_by: str = "encounter_class",
         metric: str = "TOTAL_CLAIM_COST", db: Session = Depends(get_db)) -> dict:
    result = operations.kpis(start_year, group_by, metric)
    history.record(db, "operations", f"KPIs by {group_by}", {"headline": result["headline"]}, tutorial="T8",
                   dataset="synthea:encounters")
    return result


@router.get("/utilization")
def utilization(start_year: int = Query(2016, ge=1940, le=2030), beds: int = Query(5, ge=1, le=1000),
                db: Session = Depends(get_db)) -> dict:
    result = operations.operations(start_year, beds)
    history.record(db, "operations", "Demand & utilisation analysis", {"beds": beds}, tutorial="T9",
                   dataset="synthea:encounters")
    return result


@router.post("/scenarios")
def scenarios(body: ScenarioRequest, db: Session = Depends(get_db)) -> dict:
    result = operations.scenarios([s.model_dump() for s in body.scenarios], body.start_year,
                                  body.hours_per_fte_month)
    history.record(db, "operations", f"Compared {len(body.scenarios)} scenarios",
                   {"scenarios": [{"name": s["name"], "utilization_pct": s["utilization_pct"]}
                                  for s in result["scenarios"]]}, tutorial="T9")
    return result
