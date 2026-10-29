"""T10-T12: preprocessing, NER with negation, ranking and disambiguation."""
import pytest

from app.core.config import settings
from app.core.errors import ModelUnavailable, ValidationFailed
from app.services import clinical_ner, text_analytics, wsd
from tests.conftest import requires_corpus


def _ents(text):
    return {(e["text"].lower(), e["label"]): e for e in clinical_ner.extract(text)["entities"]}


def test_ner_labels_and_negation():
    ents = _ents("Patient denies chest pain or fever but reports cough. Started amoxicillin 500 mg tid. "
                 "BP 130/85, SpO2 93%.")
    assert ents[("chest pain", "PROBLEM")]["negated"] is True
    assert ents[("fever", "PROBLEM")]["negated"] is True
    assert ents[("cough", "PROBLEM")]["negated"] is False  # 'but' terminates the negation scope
    assert ("amoxicillin", "MEDICATION") in ents
    assert ("500 mg tid", "DOSAGE") in ents
    assert ("bp 130/85", "VITAL") in ents


def test_ner_family_and_structured_output():
    r = clinical_ner.extract("Family history of asthma in mother. Chest x-ray showed no pleural effusion.")
    assert "asthma" in r["structured"]["family_history"]
    assert "pleural effusion" in r["structured"]["problems_negated"]
    assert "Chest x-ray" in r["structured"]["procedures"]


def test_ner_rejects_empty_text():
    with pytest.raises(ValidationFailed):
        clinical_ner.extract("   ")


def test_wsd_uses_context():
    assert wsd.disambiguate("Oxygen saturation 92% on RA", variant="lesk")["predicted_sense"] == "room air"
    assert wsd.disambiguate("RA flare with swollen joints on methotrexate", variant="lesk")["predicted_sense"] \
        == "rheumatoid arthritis"
    with pytest.raises(ValidationFailed):
        wsd.disambiguate("No abbreviation here")


def test_classify_without_model_is_409(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    with pytest.raises(ModelUnavailable):
        text_analytics.classify("patient with chest pain and elevated troponin")
    r = client.post("/api/text/classify", json={"text": "patient with chest pain and elevated troponin"})
    assert r.status_code == 409


def test_text_api_validation(client):
    assert client.post("/api/text/entities", json={"text": ""}).status_code == 422
    assert client.post("/api/text/wsd", json={"sentence": "MS relapse", "variant": "magic"}).status_code == 422


@requires_corpus
def test_preprocess_pipeline():
    r = text_analytics.preprocess_demo("The patients were treated with antibiotics.")
    lemmas = r["steps"][3]["tokens"]
    assert "patient" in lemmas and "treat" in lemmas and "the" not in lemmas


@requires_corpus
def test_ranking_returns_relevant_terms():
    r = text_analytics.rank("myocardial infarction coronary artery", "bm25", 5)
    assert len(r["results"]) == 5
    assert r["results"][0]["score"] >= r["results"][-1]["score"]
    assert r["results"][0]["matched_terms"]
    assert r["label_distribution"].get("cardiovascular diseases", 0) >= 3


@requires_corpus
def test_wsd_evaluation_beats_baseline():
    r = wsd.evaluate()
    assert r["accuracy"]["corpus_only"] > r["accuracy"]["first_sense_baseline"]


@requires_corpus
def test_train_and_classify(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    text_analytics._load_models.cache_clear()
    m = client.post("/api/text/train").json()
    assert set(m["models"]) == {"logistic_regression", "linear_svm", "complement_nb"}
    # Five balanced-ish classes: anything well above chance (0.2-0.3) shows the pipeline learns.
    assert m["models"][m["best_model"]]["accuracy"] > 0.45
    r = client.post("/api/text/classify", json={"text": "Carcinoma of the breast treated with chemotherapy; "
                                                         "tumor metastasis to lymph nodes."}).json()
    assert r["predictions"]["logistic_regression"]["prediction"] == "neoplasms"
    text_analytics._load_models.cache_clear()
