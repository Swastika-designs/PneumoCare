"""Loading EHR tables (Synthea, derived tables, teaching variants and uploads).

Data-source identifiers used across the API:
  * ``synthea:<table>``          raw Synthea CSV, e.g. ``synthea:patients``
  * ``synthea:patient_profile``   one row per patient: demographics + latest vitals/labs + utilisation counts
  * ``teaching:patients_messy``   Synthea patients with *documented, deliberately injected* quality problems
  * ``upload:<id>``               a CSV the user uploaded through the EHR explorer
"""
from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import DatasetUnavailable, NotFound, ValidationFailed
from app.models.entities import UploadedDataset

SYNTHEA_TABLES = [
    "patients", "encounters", "conditions", "observations", "medications", "procedures", "immunizations",
    "allergies", "careplans", "devices", "imaging_studies", "organizations", "providers", "payers", "supplies",
]
MAX_UPLOAD_ROWS = 200_000


def synthea_dir() -> Path:
    return settings.raw_dir / "synthea"


def require_synthea() -> None:
    if not (synthea_dir() / "patients.csv").exists():
        raise DatasetUnavailable(
            "The Synthea dataset has not been downloaded. Run `python scripts/download_datasets.py --only synthea`.",
            details={"dataset": "synthea"},
        )


@lru_cache(maxsize=32)
def _read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, dtype={"ZIP": str, "FIPS": str, "CODE": str, "REASONCODE": str})


def read_synthea(table: str) -> pd.DataFrame:
    require_synthea()
    if table not in SYNTHEA_TABLES:
        raise NotFound(f"Unknown Synthea table '{table}'. Choose one of: {', '.join(SYNTHEA_TABLES)}")
    path = synthea_dir() / f"{table}.csv"
    if not path.exists():
        raise DatasetUnavailable(f"Synthea table file is missing: {path.name}")
    return _read_csv_cached(str(path), path.stat().st_mtime).copy()


def reference_date() -> pd.Timestamp:
    """Latest encounter start in the data; used as 'today' so ages are reproducible."""
    enc = read_synthea("encounters")
    return pd.to_datetime(enc["START"], utc=True).max().tz_localize(None).normalize()


# Latest value of each measurement per patient. Lab panels (lipids, HbA1c, metabolic) are
# only recorded for some patients, which gives the profile realistic missingness.
PROFILE_MEASUREMENTS = {
    "Body Height": "HEIGHT_CM",
    "Body Weight": "WEIGHT_KG",
    "Body mass index (BMI) [Ratio]": "BMI",
    "Systolic Blood Pressure": "SYSTOLIC_BP",
    "Diastolic Blood Pressure": "DIASTOLIC_BP",
    "Heart rate": "HEART_RATE",
    "Respiratory rate": "RESP_RATE",
    "Body temperature": "TEMPERATURE_C",
    "Hemoglobin [Mass/volume] in Blood": "HEMOGLOBIN",
    "Leukocytes [#/volume] in Blood by Automated count": "WBC",
    "Platelets [#/volume] in Blood by Automated count": "PLATELETS",
    "Cholesterol [Mass/volume] in Serum or Plasma": "CHOLESTEROL",
    "Cholesterol in HDL [Mass/volume] in Serum or Plasma": "HDL",
    "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay": "LDL",
    "Triglyceride [Mass/volume] in Serum or Plasma": "TRIGLYCERIDES",
    "Hemoglobin A1c/Hemoglobin.total in Blood": "HBA1C",
    "Glucose [Mass/volume] in Blood": "GLUCOSE",
    "Creatinine [Mass/volume] in Blood": "CREATININE",
    "QALY": "QALY",
}


@lru_cache(maxsize=2)
def _patient_profile_cached(mtime: float) -> pd.DataFrame:
    patients = read_synthea("patients")
    ref = reference_date()
    birth = pd.to_datetime(patients["BIRTHDATE"])
    death = pd.to_datetime(patients["DEATHDATE"])
    end = death.fillna(ref)
    profile = pd.DataFrame({
        "Id": patients["Id"],
        "AGE": ((end - birth).dt.days / 365.25).round(1),
        "GENDER": patients["GENDER"],
        "RACE": patients["RACE"],
        "ETHNICITY": patients["ETHNICITY"],
        "MARITAL": patients["MARITAL"],
        "DECEASED": death.notna().astype(int),
        "INCOME": patients["INCOME"],
        "HEALTHCARE_EXPENSES": patients["HEALTHCARE_EXPENSES"],
        "HEALTHCARE_COVERAGE": patients["HEALTHCARE_COVERAGE"],
    })
    for table, col in (("encounters", "N_ENCOUNTERS"), ("conditions", "N_CONDITIONS"),
                       ("medications", "N_MEDICATIONS")):
        counts = read_synthea(table)["PATIENT"].value_counts()
        profile[col] = profile["Id"].map(counts).fillna(0).astype(int)

    obs = read_synthea("observations")
    obs = obs[(obs["TYPE"] == "numeric") & obs["DESCRIPTION"].isin(PROFILE_MEASUREMENTS)].copy()
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    latest = (obs.dropna(subset=["VALUE"]).sort_values("DATE")
              .groupby(["PATIENT", "DESCRIPTION"])["VALUE"].last().unstack())
    latest = latest.rename(columns=PROFILE_MEASUREMENTS)
    ordered = [c for c in PROFILE_MEASUREMENTS.values() if c in latest.columns]
    profile = profile.merge(latest[ordered], left_on="Id", right_index=True, how="left")
    return profile


def patient_profile() -> pd.DataFrame:
    require_synthea()
    return _patient_profile_cached((synthea_dir() / "observations.csv").stat().st_mtime).copy()


# ------------------------------------------------------------------ teaching dataset (T2/T3)

MESSY_COLUMNS = ["Id", "BIRTHDATE", "DEATHDATE", "FIRST", "LAST", "MARITAL", "RACE", "ETHNICITY", "GENDER",
                 "STATE", "COUNTY", "ZIP", "LAT", "LON", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME"]


def make_messy_patients(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Inject realistic data-entry problems into Synthea patients.

    Returns (messy, ground_truth, corruption_log). Every corruption is listed in the
    log so students can verify that the cleaning pipeline undoes it.
    """
    rng = np.random.default_rng(seed)
    clean = read_synthea("patients")[MESSY_COLUMNS].copy()
    clean["ZIP"] = clean["ZIP"].map(lambda z: str(z).zfill(5) if pd.notna(z) else z)
    messy = clean.astype(object).copy()
    n = len(messy)
    log: list[dict] = []

    def pick(frac: float) -> np.ndarray:
        return rng.choice(n, size=max(1, int(round(frac * n))), replace=False)

    gender_variants = {"F": ["F", "f", "Female", "female", " F "], "M": ["M", "m", "Male", "MALE", "male "]}
    rows = pick(0.6)
    messy.loc[rows, "GENDER"] = [rng.choice(gender_variants[g]) for g in clean.loc[rows, "GENDER"]]
    log.append({"column": "GENDER", "issue": "inconsistent coding (F/f/Female/female, extra spaces)",
                "rows": len(rows)})

    marital_variants = {"M": ["married", "Married"], "S": ["single", "Single"], "D": ["divorced"],
                        "W": ["widowed"]}
    rows = [r for r in pick(0.5) if clean.loc[r, "MARITAL"] in marital_variants]
    messy.loc[rows, "MARITAL"] = [rng.choice(marital_variants[m]) for m in clean.loc[rows, "MARITAL"]]
    log.append({"column": "MARITAL", "issue": "codes replaced by words (M -> married)", "rows": len(rows)})

    rows = pick(0.5)
    messy.loc[rows, "RACE"] = [rng.choice([v.upper(), v.title(), f"  {v} "]) for v in clean.loc[rows, "RACE"]]
    log.append({"column": "RACE", "issue": "mixed letter case and padding spaces", "rows": len(rows)})

    rows = pick(0.5)
    messy.loc[rows, "STATE"] = [rng.choice(["MA", "massachusetts", "MASSACHUSETTS", " Massachusetts"])
                                for _ in rows]
    log.append({"column": "STATE", "issue": "full name vs abbreviation vs case variants", "rows": len(rows)})

    birth = pd.to_datetime(clean["BIRTHDATE"])
    rows = pick(0.45)
    for r in rows:
        fmt = rng.choice(["%m/%d/%Y", "%d %b %Y", "%Y/%m/%d"])
        messy.loc[r, "BIRTHDATE"] = birth[r].strftime(fmt)
    log.append({"column": "BIRTHDATE", "issue": "mixed date formats (MM/DD/YYYY, DD Mon YYYY, YYYY/MM/DD)",
                "rows": len(rows)})

    rows = pick(0.4)
    messy.loc[rows, "HEALTHCARE_EXPENSES"] = [f"${v:,.2f}" for v in clean.loc[rows, "HEALTHCARE_EXPENSES"]]
    log.append({"column": "HEALTHCARE_EXPENSES", "issue": "currency strings ('$12,345.67')", "rows": len(rows)})

    rows = pick(0.3)
    messy.loc[rows, "INCOME"] = [f"{int(v):,}" for v in clean.loc[rows, "INCOME"]]
    log.append({"column": "INCOME", "issue": "thousands separators ('79,577')", "rows": len(rows)})

    rows = pick(0.35)
    rows = [r for r in rows if isinstance(clean.loc[r, "ZIP"], str) and clean.loc[r, "ZIP"].strip("0")]
    messy.loc[rows, "ZIP"] = [z.lstrip("0") for z in clean.loc[rows, "ZIP"]]
    log.append({"column": "ZIP", "issue": "leading zeros lost (01095 -> 1095)", "rows": len(rows)})

    invalid_income = pick(0.03)
    messy.loc[invalid_income, "INCOME"] = -1
    log.append({"column": "INCOME", "issue": "invalid negative values (-1 used as a sentinel)",
                "rows": len(invalid_income)})
    invalid_lat = pick(0.02)
    messy.loc[invalid_lat, "LAT"] = [v * 10 for v in clean.loc[invalid_lat, "LAT"]]
    log.append({"column": "LAT", "issue": "out-of-range latitude (decimal point shifted)", "rows": len(invalid_lat)})

    null_rows = pick(0.08)
    messy.loc[null_rows, "INCOME"] = [rng.choice(["", "N/A", "null", "unknown"]) for _ in null_rows]
    log.append({"column": "INCOME", "issue": "missing values written as text tokens (N/A, null, unknown)",
                "rows": len(null_rows)})
    null_rows = pick(0.1)
    messy.loc[null_rows, "MARITAL"] = np.nan
    log.append({"column": "MARITAL", "issue": "additional values deleted (missing)", "rows": len(null_rows)})

    dup_rows = messy.iloc[rng.choice(n, size=5, replace=False)]
    near = messy.iloc[rng.choice(n, size=3, replace=False)].copy()
    near["FIRST"] = near["FIRST"].astype(str).str.upper() + " "
    messy = pd.concat([messy, dup_rows, near], ignore_index=True)
    log.append({"column": "(rows)", "issue": "exact duplicate records appended", "rows": 5})
    log.append({"column": "(rows)", "issue": "near-duplicates (same Id, name re-typed in upper case)", "rows": 3})
    messy = messy.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return messy, clean, log


def messy_patients() -> pd.DataFrame:
    return make_messy_patients()[0]


# ------------------------------------------------------------------ uploads


def save_upload(db: Session, filename: str, content: bytes) -> UploadedDataset:
    if not filename.lower().endswith(".csv"):
        raise ValidationFailed("Only .csv files can be uploaded to the EHR explorer.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise ValidationFailed(f"CSV is larger than the {settings.max_upload_mb:g} MB upload limit.")
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:
        raise ValidationFailed(f"Could not parse CSV: {exc}") from exc
    if df.empty or df.shape[1] == 0:
        raise ValidationFailed("The CSV has no rows or no columns.")
    if len(df) > MAX_UPLOAD_ROWS:
        raise ValidationFailed(f"CSV has {len(df):,} rows; the limit is {MAX_UPLOAD_ROWS:,}.")
    folder = settings.uploads_dir / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).name)
    record = UploadedDataset(name=safe, stored_path="", rows=len(df), columns=df.shape[1])
    db.add(record)
    db.flush()
    path = folder / f"{record.id}_{safe}"
    path.write_bytes(content)
    record.stored_path = str(path)
    db.commit()
    return record


def list_sources(db: Session | None = None) -> list[dict]:
    sources = []
    if (synthea_dir() / "patients.csv").exists():
        sources.append({"id": "synthea:patient_profile", "label": "Synthea - patient profile (derived)",
                        "group": "Synthea (synthetic EHR)"})
        sources += [{"id": f"synthea:{t}", "label": f"Synthea - {t}", "group": "Synthea (synthetic EHR)"}
                    for t in SYNTHEA_TABLES if (synthea_dir() / f"{t}.csv").exists()]
        sources.append({"id": "teaching:patients_messy",
                        "label": "Teaching - Synthea patients with injected errors",
                        "group": "Teaching variants (documented corruption)"})
    if db is not None:
        for up in db.query(UploadedDataset).order_by(UploadedDataset.created_at.desc()).all():
            sources.append({"id": f"upload:{up.id}", "label": f"Upload - {up.name} ({up.rows} rows)",
                            "group": "Uploaded files"})
    return sources


def load_source(source: str, db: Session | None = None) -> pd.DataFrame:
    kind, _, name = source.partition(":")
    if kind == "synthea":
        return patient_profile() if name == "patient_profile" else read_synthea(name)
    if kind == "teaching" and name == "patients_messy":
        require_synthea()
        return messy_patients()
    if kind == "upload":
        if db is None or not name.isdigit():
            raise NotFound(f"Unknown upload '{source}'")
        record = db.get(UploadedDataset, int(name))
        if record is None or not Path(record.stored_path).exists():
            raise NotFound(f"Uploaded dataset {name} was not found.")
        return pd.read_csv(record.stored_path, low_memory=False)
    raise NotFound(f"Unknown data source '{source}'.")
