"""T1-T3: EHR exploration, preprocessing, missing data and redundancy."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services import ehr_analysis as analysis
from app.services import ehr_data as data
from app.services import history

router = APIRouter(prefix="/api/ehr", tags=["ehr / preprocessing"])
DEFAULT_SOURCE = "synthea:patients"


@router.get("/sources")
def sources(db: Session = Depends(get_db)) -> list[dict]:
    return data.list_sources(db)


@router.post("/upload")
def upload(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    record = data.save_upload(db, file.filename or "upload.csv", file.file.read())
    return {"id": f"upload:{record.id}", "name": record.name, "rows": record.rows, "columns": record.columns}


@router.get("/overview")
def overview(source: str = DEFAULT_SOURCE, db: Session = Depends(get_db)) -> dict:
    result = analysis.overview(data.load_source(source, db))
    history.record(db, "ehr", f"Explored {source}", {"rows": result["rows"], "columns": result["columns"]},
                   tutorial="T1", dataset=source)
    return result


@router.get("/column")
def column(column: str, source: str = DEFAULT_SOURCE, db: Session = Depends(get_db)) -> dict:
    return analysis.column_profile(data.load_source(source, db), column)


@router.get("/teaching-log")
def teaching_log() -> dict:
    data.require_synthea()
    messy, clean, log = data.make_messy_patients()
    return {"rows": len(messy), "original_rows": len(clean), "corruptions": log,
            "note": "Errors are injected deterministically (seed 7) into a copy of Synthea patients.csv so that the "
                    "cleaning pipeline can be checked against the original values."}


@router.get("/clean")
def clean(source: str = "teaching:patients_messy", db: Session = Depends(get_db)) -> dict:
    df = data.load_source(source, db)
    truth = data.make_messy_patients()[1] if source == "teaching:patients_messy" else None
    result = analysis.cleaning_report(df, truth)
    history.record(db, "ehr", f"Standardised {source}",
                   {"steps": len(result["steps"]),
                    "violations_before": sum(v["violations"] for v in result["validation_before"]),
                    "violations_after": sum(v["violations"] for v in result["validation_after"])},
                   tutorial="T2", dataset=source)
    return result


@router.get("/missingness")
def missingness(source: str = "synthea:patient_profile", normalize_tokens: bool = True,
                db: Session = Depends(get_db)) -> dict:
    result = analysis.missingness(data.load_source(source, db), normalize_tokens)
    history.record(db, "ehr", f"Missingness of {source}", {"missing_cell_pct": result["missing_cell_pct"]},
                   tutorial="T3", dataset=source)
    return result


@router.get("/duplicates")
def duplicates(source: str = "teaching:patients_messy", key: str | None = None,
               near: list[str] | None = Query(default=None), db: Session = Depends(get_db)) -> dict:
    result = analysis.duplicates(data.load_source(source, db), key, near)
    history.record(db, "ehr", f"Duplicate detection on {source}",
                   {k: result[k] for k in ("exact_duplicates", "near_duplicates", "key_duplicates", "removed")},
                   tutorial="T3", dataset=source)
    return result


@router.get("/imputation")
def imputation(source: str = "synthea:patient_profile", target: str | None = None, mask_fraction: float = 0.15,
               db: Session = Depends(get_db)) -> dict:
    result = analysis.imputation_comparison(data.load_source(source, db), target, min(max(mask_fraction, 0.05), 0.5))
    history.record(db, "ehr", f"Imputation comparison on {result['target']}",
                   {"best": result["best_method"], "results": result["results"]}, tutorial="T3", dataset=source)
    return result


@router.get("/quality")
def quality(source: str = "teaching:patients_messy", db: Session = Depends(get_db)) -> dict:
    result = analysis.quality_report(data.load_source(source, db))
    history.record(db, "ehr", f"Data-quality report for {source}", {"overall": result["overall"],
                   "scores": result["scores"]}, tutorial="T3", dataset=source)
    return result
