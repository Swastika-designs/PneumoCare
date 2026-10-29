"""T10-T12: clinical text analytics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.errors import NotFound
from app.db.database import get_db
from app.schemas.api import RankRequest, TextRequest, WsdRequest
from app.services import clinical_ner, history, text_analytics, wsd

router = APIRouter(prefix="/api/text", tags=["clinical text"])


@router.get("/corpus")
def corpus() -> dict:
    return text_analytics.corpus_summary()


@router.get("/corpus/{doc_id}")
def corpus_document(doc_id: int) -> dict:
    train, _, _ = text_analytics.load_corpus()
    if not 0 <= doc_id < len(train):
        raise NotFound(f"Document {doc_id} does not exist (0-{len(train) - 1}).")
    row = train.iloc[doc_id]
    return {"doc_id": doc_id, "label": row["label"], "title": row["title"], "text": row["medical_abstract"]}


@router.post("/preprocess")
def preprocess(body: TextRequest) -> dict:
    return text_analytics.preprocess_demo(body.text)


@router.post("/train")
def train(db: Session = Depends(get_db)) -> dict:
    result = text_analytics.train_classifiers()
    history.record(db, "text", "Trained clinical text classifiers",
                   {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in result["models"].items()},
                   tutorial="T10", dataset="medical_abstracts")
    return result


@router.get("/metrics")
def metrics() -> dict:
    result = text_analytics.classifier_metrics()
    return result or {"trained": False}


@router.post("/classify")
def classify(body: TextRequest, db: Session = Depends(get_db)) -> dict:
    result = text_analytics.classify(body.text)
    history.record(db, "text", "Classified clinical text",
                   {"prediction": result["predictions"]["logistic_regression"]["prediction"]}, tutorial="T10")
    return result


@router.get("/cluster")
def cluster(k: int = Query(5, ge=2, le=12), sample_size: int = Query(3000, ge=200, le=11550),
            db: Session = Depends(get_db)) -> dict:
    result = text_analytics.cluster(k, sample_size)
    history.record(db, "text", f"k-means clustering (k={k})",
                   {"silhouette": result["silhouette"], "nmi": result["normalized_mutual_info"]}, tutorial="T10",
                   dataset="medical_abstracts")
    return result


@router.get("/sample-notes")
def sample_notes() -> list[dict]:
    return clinical_ner.SAMPLE_NOTES


@router.post("/entities")
def entities(body: TextRequest, db: Session = Depends(get_db)) -> dict:
    result = clinical_ner.extract(body.text)
    history.record(db, "text", "Clinical entity extraction", {"counts": result["counts"]}, tutorial="T11")
    return result


@router.get("/entity-stats")
def entity_stats(per_class: int = Query(40, ge=5, le=200)) -> dict:
    return clinical_ner.corpus_entity_stats(per_class)


@router.post("/rank")
def rank(body: RankRequest, db: Session = Depends(get_db)) -> dict:
    result = text_analytics.rank(body.query, body.method, body.top_k)
    history.record(db, "text", f"Ranked documents for '{body.query[:40]}'", {"method": body.method},
                   tutorial="T12", dataset="medical_abstracts")
    return result


@router.get("/rank-evaluation")
def rank_evaluation(n_queries: int = Query(200, ge=20, le=1000), k: int = Query(10, ge=1, le=50)) -> dict:
    return text_analytics.evaluate_ranking(n_queries, k)


@router.post("/wsd")
def disambiguate(body: WsdRequest, db: Session = Depends(get_db)) -> dict:
    result = wsd.disambiguate(body.sentence, body.abbreviation, body.variant)
    history.record(db, "text", f"Disambiguated {result['abbreviation']}", {"sense": result["predicted_sense"]},
                   tutorial="T12")
    return result


@router.get("/wsd-evaluation")
def wsd_evaluation() -> dict:
    return wsd.evaluate()
