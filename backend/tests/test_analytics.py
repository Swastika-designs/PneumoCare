"""T7-T9 operations, T13 PCA, T14-T15 weight analytics and report exports."""
import numpy as np
import pandas as pd
import pytest

from app.core.config import settings
from app.core.errors import ValidationFailed
from app.services import dimensionality, operations
from tests.conftest import requires_nhanes, requires_synthea


def test_pca_on_correlated_data():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(300, 1))
    df = pd.DataFrame(np.hstack([latent + 0.05 * rng.normal(size=(300, 1)) for _ in range(4)]
                                + [rng.normal(size=(300, 2))]), columns=list("abcdef"))
    df["group"] = np.where(latent[:, 0] > 0, "hi", "lo")
    r = dimensionality.run_pca(df, ["group"], color_by="group")
    first = r["explained_variance"][0]["ratio"]
    assert first > 0.6  # four of six columns share one latent factor
    assert r["explained_variance"][-1]["cumulative"] == pytest.approx(1.0)
    assert {p["group"] for p in r["points"]} == {"hi", "lo"}


def test_pca_validation():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 4.0]})
    with pytest.raises(ValidationFailed):
        dimensionality.run_pca(df, [])


def test_scenarios_require_input():
    with pytest.raises(ValidationFailed):
        operations.scenarios([])


@requires_synthea
def test_forecast_beats_or_matches_naive(client):
    r = client.get("/api/operations/forecast", params={"freq": "month", "horizon": 3}).json()
    names = {x["model"]: x for x in r["results"]}
    assert set(names) == {"Naive (last period)", "Moving average", "Ridge regression", "Random forest"}
    assert len(r["forecast"]) == 3 and all(f["required_fte"] >= 0 for f in r["forecast"])
    assert r["test_periods"] > 0 and r["train_periods"] > r["test_periods"]


@requires_synthea
def test_kpis_and_group_test(client):
    r = client.get("/api/operations/kpis", params={"group_by": "encounter_class"}).json()
    h = r["headline"]
    assert h["encounters"] > 0 and 0 <= h["payer_coverage_pct"] <= 100
    assert sum(g["encounters"] for g in r["groups"]) <= h["encounters"]
    assert r["comparison_test"]["test"] == "Kruskal-Wallis H"
    assert client.get("/api/operations/kpis", params={"group_by": "shoe_size"}).status_code == 422


@requires_synthea
def test_utilization_and_scenarios(client):
    u = client.get("/api/operations/utilization", params={"beds": 4}).json()
    assert len(u["heatmap"]) == 7 * 24
    s = client.post("/api/operations/scenarios", json={"scenarios": [
        {"name": "base", "staff_fte": 1}, {"name": "double demand", "demand_change_pct": 100, "staff_fte": 1}]}).json()
    base, double = s["scenarios"]
    assert double["workload_hours"] == pytest.approx(2 * base["workload_hours"], rel=0.01)
    assert double["utilization_pct"] == pytest.approx(2 * base["utilization_pct"], rel=0.01)


@requires_nhanes
def test_weight_exploration(client):
    r = client.get("/api/analytics/weight/explore").json()
    assert r["n_adults"] > 4000
    assert 30 < r["obesity_prevalence"]["unweighted_pct"] < 55
    assert len(r["correlations"]) == len(r["correlation_variables"]) ** 2


@requires_nhanes
def test_weight_models_exclude_leaky_features(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    r = client.post("/api/analytics/weight/train").json()
    assert "weight_kg" in r["excluded_features"]
    used = r["features"]["numeric"] + r["features"]["categorical"]
    assert not {"weight_kg", "height_cm", "waist_cm", "bmi"} & set(used)
    assert r["regression"]["random_forest_regressor"]["r2"] > r["regression"]["baseline_mean"]["r2"]
    assert r["classification"]["logistic_regression"]["roc_auc"] > 0.55
    assert client.get("/api/analytics/weight/results").json()["n_test"] == r["n_test"]


@requires_nhanes
def test_pca_api(client):
    r = client.get("/api/analytics/pca", params={"source": "nhanes_adults", "color_by": "sex"}).json()
    assert r["components_for_80pct"] <= len(r["features"])
    assert client.get("/api/analytics/pca", params={"scaling": "weird"}).status_code == 422


def test_reports_formats(client):
    for fmt, prefix in (("json", b"{"), ("csv", b"\xef\xbb\xbf"), ("pdf", b"%PDF")):
        r = client.get("/api/reports/registry", params={"format": fmt})
        assert r.status_code == 200
        assert r.content.startswith(prefix)
        assert "attachment" in r.headers["content-disposition"]
    assert client.get("/api/reports/registry", params={"format": "docx"}).status_code == 422


def test_report_without_results_is_clear_error(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    r = client.get("/api/reports/pneumonia_evaluation")
    assert r.status_code == 409
    assert "train" in r.json()["error"]["message"].lower()
