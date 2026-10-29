"""T1-T3 calculations on small hand-made frames, plus API tests on Synthea."""
import io

import numpy as np
import pandas as pd
import pytest

from app.services import ehr_analysis as a
from tests.conftest import requires_synthea


@pytest.fixture
def messy() -> pd.DataFrame:
    return pd.DataFrame({
        "Id": ["a", "b", "c", "c", "d", "e"],
        "GENDER": [" female", "M", "Male", "Male", "f", "N/A"],
        "BIRTHDATE": ["1980-01-02", "03/15/1990", "1975/07/30", "1975/07/30", "12 Feb 2001", None],
        "INCOME": ["$1,200", "3400", "-1", "-1", "unknown", "5000"],
        "STATE": ["Massachusetts", "MA", "massachusetts", "massachusetts", "ma", "MA"],
        "ZIP": ["1095", "02138", "02139", "02139", None, "2140"],
        "LAT": [42.1, 421.0, 42.3, 42.3, 41.9, 42.0],
    })


def test_overview_counts(messy):
    o = a.overview(messy)
    assert o["rows"] == 6 and o["columns"] == 7
    assert o["duplicate_rows"] == 1
    schema = {s["column"]: s for s in o["schema"]}
    assert schema["BIRTHDATE"]["missing"] == 1
    assert schema["Id"]["semantic_type"] == "identifier"


def test_column_profile_numeric_histogram():
    df = pd.DataFrame({"x": np.arange(100, dtype=float)})
    prof = a.column_profile(df, "x", bins=10)
    assert prof["chart"] == "histogram"
    assert sum(b["count"] for b in prof["data"]) == 100


def test_standardize_fixes_types_and_categories(messy):
    clean, steps = a.standardize(messy)
    assert set(clean["GENDER"].dropna()) == {"F", "M"}
    assert clean["GENDER"].isna().sum() == 1  # "N/A" -> missing
    assert set(clean["STATE"]) == {"MA"}
    assert clean["INCOME"].dtype.kind == "f"
    assert clean.loc[0, "INCOME"] == 1200
    assert pd.isna(clean.loc[2, "INCOME"])  # negative sentinel -> missing
    assert pd.api.types.is_datetime64_any_dtype(clean["BIRTHDATE"])
    assert clean.loc[1, "BIRTHDATE"] == pd.Timestamp("1990-03-15")
    assert clean.loc[0, "ZIP"] == "01095"
    assert pd.isna(clean.loc[1, "LAT"])  # out-of-range latitude
    assert any("Convert INCOME to numeric" in s["step"] for s in steps)


def test_validation_issue_counts(messy):
    issues = {i["rule"]: i["violations"] for i in a.validation_issues(messy)}
    assert issues["LAT must be within [-90, 90]"] == 1
    assert issues["INCOME must be >= 0"] == 2
    assert issues["ZIP must be 5 digits"] == 2


def test_missingness_converts_null_tokens(messy):
    m = a.missingness(messy)
    assert m["null_tokens_converted"] == 2  # "N/A" and "unknown"
    per = {r["column"]: r["missing"] for r in m["per_column"]}
    assert per["INCOME"] == 1 and per["GENDER"] == 1


def test_duplicates_exact_key_and_near():
    df = pd.DataFrame({"Id": [1, 2, 2, 3, 3], "NAME": ["Ann", "Bob", "Bob", "Cy", "CY "]})
    d = a.duplicates(df, key="Id")
    assert d["exact_duplicates"] == 1
    assert d["near_duplicates"] == 1
    assert d["rows_after_dedup"] == 3


def test_imputation_comparison_ranks_methods():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "y": 3 * x + rng.normal(scale=0.1, size=300), "z": rng.normal(size=300)})
    df.loc[:30, "y"] = np.nan
    r = a.imputation_comparison(df, target="y")
    methods = {row["method"]: row["rmse"] for row in r["results"]}
    assert set(methods) == {"mean", "median", "knn", "iterative"}
    # y is almost a linear function of x, so model-based imputation must beat the mean.
    assert methods["iterative"] < methods["mean"]


def test_quality_report_improves_after_cleaning(messy):
    before = a.quality_report(messy)
    after = a.quality_report(a.standardize(messy)[0].drop_duplicates(subset="Id"))
    assert after["overall"] > before["overall"]


@requires_synthea
def test_teaching_dataset_cleaning_recovers_ground_truth(client):
    body = client.get("/api/ehr/clean").json()
    assert sum(v["violations"] for v in body["validation_after"]) == 0
    recovery = {r["column"]: r for r in body["recovery_vs_ground_truth"]}
    for col in ("GENDER", "RACE", "STATE", "BIRTHDATE", "HEALTHCARE_EXPENSES", "ZIP"):
        assert recovery[col]["match_pct"] == 100.0, col


@requires_synthea
def test_ehr_api_endpoints(client):
    sources = [s["id"] for s in client.get("/api/ehr/sources").json()]
    assert "synthea:patient_profile" in sources and "teaching:patients_messy" in sources
    ov = client.get("/api/ehr/overview", params={"source": "synthea:patients"}).json()
    assert ov["rows"] > 0 and any(s["column"] == "GENDER" for s in ov["schema"])
    assert client.get("/api/ehr/column", params={"source": "synthea:patients", "column": "GENDER"}).json()["chart"] \
        == "bar"
    assert client.get("/api/ehr/missingness").json()["total_missing_cells"] > 0
    assert client.get("/api/ehr/duplicates").json()["removed"] == 8
    assert client.get("/api/ehr/imputation").json()["best_method"]
    assert client.get("/api/ehr/overview", params={"source": "bogus:x"}).status_code == 404


def test_csv_upload_validation(client):
    bad = client.post("/api/ehr/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert bad.status_code == 422
    csv = io.BytesIO(b"a,b,c\n1,2,x\n3,,y\n")
    ok = client.post("/api/ehr/upload", files={"file": ("small.csv", csv, "text/csv")})
    assert ok.status_code == 200
    source = ok.json()["id"]
    ov = client.get("/api/ehr/overview", params={"source": source}).json()
    assert ov["rows"] == 2 and ov["total_missing_cells"] == 1
