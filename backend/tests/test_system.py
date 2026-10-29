"""Health, database initialisation, registries, tutorials and error format."""
from sqlalchemy import inspect

from app.db import database


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] is True


def test_database_tables_created(client):
    tables = set(inspect(database.engine).get_table_names())
    assert {"analysis_runs", "predictions", "uploaded_datasets"} <= tables


def test_system_status_shape(client):
    body = client.get("/api/system/status").json()
    assert body["tutorials"]["total"] == 15
    assert 0 <= body["tutorials"]["verified"] <= 15
    assert {i["status"] for i in body["tutorials"]["items"]} <= {"verified", "not verified", "verification failed"}
    assert body["datasets"]["required"] == body["datasets"]["total"] - 1  # Kaggle is optional
    assert {"total", "available", "items"} <= set(body["datasets"])
    assert isinstance(body["models"], list) and body["models"]


def test_tutorial_catalogue_complete(client):
    items = client.get("/api/tutorials").json()
    assert [t["id"] for t in items] == [f"T{i}" for i in range(1, 16)]
    for t in items:
        assert t["route"].startswith("/") and t["endpoints"] and t["objective"]


def test_tutorial_detail_and_404(client):
    assert client.get("/api/tutorials/t12").json()["id"] == "T12"
    r = client.get("/api/tutorials/T42")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_dataset_registry_flags_synthetic(client):
    datasets = {d["id"]: d for d in client.get("/api/datasets").json()}
    assert datasets["synthea"]["synthetic"] is True
    assert datasets["wsd_examples"]["synthetic"] is True
    assert datasets["nhanes"]["synthetic"] is False
    assert "Kaggle account" in datasets["kaggle_chest_xray"]["access"]


def test_request_validation_uses_error_envelope(client):
    r = client.post("/api/operations/scenarios", json={"scenarios": [{"staff_fte": 0}]})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
    assert "staff_fte" in r.json()["error"]["message"]


def test_unknown_report_404(client):
    r = client.get("/api/reports/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_history_ignores_immediate_duplicates(client):
    from app.services import history

    db = database.SessionLocal()
    try:
        assert history.record(db, "test", "Duplicate guard", {}, tutorial="T1") is True
        assert history.record(db, "test", "Duplicate guard", {}, tutorial="T1") is False
        assert history.record(db, "test", "Duplicate guard", {}, tutorial="T2") is True
    finally:
        db.close()


def test_unknown_route_uses_error_envelope(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
