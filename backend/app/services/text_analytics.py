"""T10 + T12 (ranking): clinical text classification, clustering and retrieval.

Corpus: Medical Abstracts Text Classification Corpus (Schopf et al., 2022) - 11,550 training
and 2,888 test abstracts, each labelled with one of 5 disease classes.
"""
from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from app.core.config import settings
from app.core.errors import DatasetUnavailable, ModelUnavailable, ValidationFailed
from app.core.utils import to_jsonable

_train_lock = threading.Lock()


def corpus_dir() -> Path:
    return settings.raw_dir / "medical_abstracts"


def text_models_dir() -> Path:
    return settings.models_dir / "text"


@lru_cache(maxsize=1)
def _load_corpus_cached(mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    folder = corpus_dir()
    labels = pd.read_csv(folder / "medical_tc_labels.csv")
    names = dict(zip(labels["condition_label"], labels["condition_name"]))
    train = pd.read_csv(folder / "medical_tc_train.csv")
    test = pd.read_csv(folder / "medical_tc_test.csv")
    for df in (train, test):
        df["label"] = df["condition_label"].map(names)
        df["title"] = df["medical_abstract"].str.split(r"(?<=\.)\s", n=1, regex=True).str[0]
    return train, test, names


def load_corpus() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    path = corpus_dir() / "medical_tc_train.csv"
    if not path.exists():
        raise DatasetUnavailable(
            "The Medical Abstracts corpus is missing. Run `python scripts/download_datasets.py --only "
            "medical_abstracts`.", details={"dataset": "medical_abstracts"})
    return _load_corpus_cached(path.stat().st_mtime)


def corpus_summary() -> dict:
    train, test, names = load_corpus()
    lengths = train["medical_abstract"].str.split().str.len()
    return to_jsonable({
        "train_documents": len(train),
        "test_documents": len(test),
        "classes": list(names.values()),
        "class_counts": {
            "train": train["label"].value_counts().to_dict(),
            "test": test["label"].value_counts().to_dict(),
        },
        "words_per_document": {"mean": lengths.mean(), "median": lengths.median(), "max": lengths.max()},
        "examples": train.groupby("label").head(1)[["label", "title"]].to_dict(orient="records"),
    })


# ------------------------------------------------------------------ preprocessing demo


def preprocess_demo(text: str) -> dict:
    if not text or not text.strip():
        raise ValidationFailed("Enter some text to preprocess.")
    from app.services.clinical_ner import get_nlp

    nlp, _ = get_nlp()
    doc = nlp(text[:5000])
    tokens = [t.text for t in doc if not t.is_space]
    lowered = [t.lower() for t in tokens]
    no_stop = [t.text.lower() for t in doc if t.is_alpha and not t.is_stop and t.text.lower() not in
               ENGLISH_STOP_WORDS]
    lemmas = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and t.text.lower() not in
              ENGLISH_STOP_WORDS]
    result = {
        "steps": [
            {"name": "1. Tokenisation (spaCy)", "tokens": tokens[:80], "count": len(tokens)},
            {"name": "2. Lower-casing", "tokens": lowered[:80], "count": len(lowered)},
            {"name": "3. Remove stop words, punctuation and numbers", "tokens": no_stop[:80], "count": len(no_stop)},
            {"name": "4. Lemmatisation", "tokens": lemmas[:80], "count": len(lemmas)},
        ],
    }
    model_path = text_models_dir() / "classifiers.joblib"
    if model_path.exists():
        vec = _load_models()["logistic_regression"].named_steps["tfidf"]
        weights = vec.transform([text]).toarray()[0]
        vocab = np.array(vec.get_feature_names_out())
        top = np.argsort(weights)[::-1][:12]
        result["tfidf_top_terms"] = [{"term": vocab[i], "weight": float(weights[i])} for i in top if weights[i] > 0]
    return to_jsonable(result)


# ------------------------------------------------------------------ T10: classification


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=60000,
                           stop_words="english")


CLASSIFIERS = {
    "logistic_regression": ("Logistic regression", lambda: LogisticRegression(max_iter=3000, C=4.0)),
    "linear_svm": ("Linear SVM", lambda: LinearSVC(C=0.5)),
    "complement_nb": ("Complement Naive Bayes", lambda: ComplementNB(alpha=0.3)),
}


def train_classifiers() -> dict:
    train, test, names = load_corpus()
    classes = list(names.values())
    out_dir = text_models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted, metrics = {}, {}
    with _train_lock:
        for key, (label, factory) in CLASSIFIERS.items():
            pipe = Pipeline([("tfidf", _vectorizer()), ("clf", factory())])
            pipe.fit(train["medical_abstract"], train["label"])
            pred = pipe.predict(test["medical_abstract"])
            report = classification_report(test["label"], pred, labels=classes, output_dict=True, zero_division=0)
            metrics[key] = {
                "name": label,
                "accuracy": accuracy_score(test["label"], pred),
                "macro_f1": f1_score(test["label"], pred, average="macro"),
                "weighted_f1": f1_score(test["label"], pred, average="weighted"),
                "per_class": [{"class": c, "precision": report[c]["precision"], "recall": report[c]["recall"],
                               "f1": report[c]["f1-score"], "support": report[c]["support"]} for c in classes],
                "confusion_matrix": confusion_matrix(test["label"], pred, labels=classes).tolist(),
            }
            fitted[key] = pipe
        lr = fitted["logistic_regression"]
        vocab = np.array(lr.named_steps["tfidf"].get_feature_names_out())
        coefs = lr.named_steps["clf"].coef_
        top_terms = {cls: vocab[np.argsort(coefs[i])[::-1][:15]].tolist()
                     for i, cls in enumerate(lr.named_steps["clf"].classes_)}
        result = {
            "classes": classes,
            "train_documents": len(train),
            "test_documents": len(test),
            "vectorizer": "TF-IDF, unigrams+bigrams, sublinear TF, min_df=2, max_df=0.9, English stop words",
            "models": metrics,
            "top_terms": top_terms,
            "best_model": max(metrics, key=lambda k: metrics[k]["macro_f1"]),
        }
        joblib.dump(fitted, out_dir / "classifiers.joblib")
        (out_dir / "classifier_metrics.json").write_text(json.dumps(to_jsonable(result), indent=2))
        _load_models.cache_clear()
    return to_jsonable(result)


@lru_cache(maxsize=1)
def _load_models() -> dict:
    return joblib.load(text_models_dir() / "classifiers.joblib")


def classifier_metrics() -> dict | None:
    path = text_models_dir() / "classifier_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def classify(text: str) -> dict:
    if not text or len(text.split()) < 3:
        raise ValidationFailed("Enter at least a few words of clinical text to classify.")
    if not (text_models_dir() / "classifiers.joblib").exists():
        raise ModelUnavailable("Text classifiers are not trained yet. Train them first (about 30 seconds).")
    models = _load_models()
    out = {}
    for key, pipe in models.items():
        item = {"name": CLASSIFIERS[key][0], "prediction": pipe.predict([text])[0]}
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            probs = pipe.predict_proba([text])[0]
            item["probabilities"] = {c: float(p) for c, p in zip(pipe.named_steps["clf"].classes_, probs)}
        out[key] = item
    lr = models["logistic_regression"]
    vec, clf = lr.named_steps["tfidf"], lr.named_steps["clf"]
    x = vec.transform([text])
    cls_idx = list(clf.classes_).index(out["logistic_regression"]["prediction"])
    contrib = x.multiply(clf.coef_[cls_idx]).toarray()[0]
    vocab = vec.get_feature_names_out()
    top = np.argsort(contrib)[::-1][:10]
    return to_jsonable({
        "predictions": out,
        "explanation": [{"term": vocab[i], "contribution": float(contrib[i])} for i in top if contrib[i] > 0],
    })


# ------------------------------------------------------------------ T10: clustering


def cluster(k: int = 5, sample_size: int = 3000, seed: int = 0) -> dict:
    if not 2 <= k <= 12:
        raise ValidationFailed("k must be between 2 and 12.")
    train, _, _ = load_corpus()
    sample = train.sample(n=min(sample_size, len(train)), random_state=seed).reset_index(drop=True)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.8, stop_words="english", max_features=20000)
    x = vec.fit_transform(sample["medical_abstract"])
    svd = TruncatedSVD(n_components=100, random_state=seed)
    reduced = normalize(svd.fit_transform(x))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(reduced)
    vocab = np.array(vec.get_feature_names_out())
    top_terms = []
    for c in range(k):
        mean_tfidf = np.asarray(x[labels == c].mean(axis=0)).ravel()
        dominant = sample.loc[labels == c, "label"].value_counts()
        top_terms.append({"cluster": c, "size": int((labels == c).sum()),
                          "terms": vocab[np.argsort(mean_tfidf)[::-1][:10]].tolist(),
                          "dominant_label": dominant.index[0],
                          "dominant_share": float(dominant.iloc[0] / dominant.sum())})
    contingency = pd.crosstab(labels, sample["label"])
    purity = float(contingency.max(axis=1).sum() / len(sample))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sample), size=min(600, len(sample)), replace=False)
    points = [{"x": float(reduced[i, 0]), "y": float(reduced[i, 1]), "cluster": int(labels[i]),
               "label": sample.at[int(i), "label"], "title": sample.at[int(i), "title"][:100]} for i in idx]
    inertia = []
    for kk in range(2, 11):
        inertia.append({"k": kk, "inertia": float(KMeans(n_clusters=kk, n_init=3, random_state=seed)
                                                  .fit(reduced).inertia_)})
    return to_jsonable({
        "k": k,
        "documents": len(sample),
        "method": "TF-IDF -> Truncated SVD (100 dims, L2-normalised) -> k-means",
        "silhouette": float(silhouette_score(reduced, labels, sample_size=min(2000, len(sample)), random_state=seed)),
        "adjusted_rand_index": float(adjusted_rand_score(sample["label"], labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(sample["label"], labels)),
        "purity": purity,
        "clusters": top_terms,
        "contingency": {"clusters": [int(c) for c in contingency.index], "labels": list(contingency.columns),
                        "counts": contingency.values.tolist()},
        "points": points,
        "elbow": inertia,
        "projection_note": "Scatter shows the first two SVD components (a linear projection), so clusters overlap "
                           "visually even when they are separated in 100 dimensions.",
    })


# ------------------------------------------------------------------ T12: ranking


class _Index:
    def __init__(self, docs: pd.DataFrame):
        self.docs = docs
        self.tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)
        self.tfidf_matrix = self.tfidf.fit_transform(docs["medical_abstract"])
        self.counter = CountVectorizer(stop_words="english", min_df=2, vocabulary=self.tfidf.vocabulary_)
        tf = self.counter.transform(docs["medical_abstract"]).tocsr().astype(np.float32)
        self.doc_len = np.asarray(tf.sum(axis=1)).ravel()
        self.avg_len = float(self.doc_len.mean())
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        n = tf.shape[0]
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        self.tf = tf.tocsc()

    def scores(self, query: str, method: str) -> np.ndarray:
        if method == "tfidf":
            q = self.tfidf.transform([query])
            return (self.tfidf_matrix @ q.T).toarray().ravel()
        k1, b = 1.5, 0.75
        q_terms = self.counter.transform([query]).indices
        scores = np.zeros(self.tf.shape[0], dtype=np.float32)
        norm = k1 * (1 - b + b * self.doc_len / self.avg_len)
        for t in q_terms:
            col = self.tf.getcol(t)
            rows, freqs = col.indices, col.data
            scores[rows] += self.idf[t] * freqs * (k1 + 1) / (freqs + norm[rows])
        return scores


@lru_cache(maxsize=1)
def _index() -> _Index:
    train, _, _ = load_corpus()
    return _Index(train)


def rank(query: str, method: str = "tfidf", top_k: int = 10) -> dict:
    if method not in {"tfidf", "bm25"}:
        raise ValidationFailed("method must be 'tfidf' or 'bm25'")
    if not query or not re.search(r"[A-Za-z]{3,}", query):
        raise ValidationFailed("Enter a query with at least one word.")
    index = _index()
    scores = index.scores(query, method)
    top = np.argsort(scores)[::-1][:max(1, min(top_k, 50))]
    q_terms = set(index.counter.build_analyzer()(query))
    results = []
    for rank_pos, i in enumerate(top, 1):
        doc = index.docs.iloc[i]
        doc_terms = set(index.counter.build_analyzer()(doc["medical_abstract"]))
        results.append({"rank": rank_pos, "doc_id": int(i), "score": float(scores[i]), "label": doc["label"],
                        "title": doc["title"], "snippet": doc["medical_abstract"][:400],
                        "matched_terms": sorted(q_terms & doc_terms)})
    labels = pd.Series([r["label"] for r in results]).value_counts().to_dict()
    return to_jsonable({"query": query, "method": method, "results": results, "label_distribution": labels,
                        "documents_indexed": len(index.docs)})


def evaluate_ranking(n_queries: int = 200, k: int = 10, seed: int = 0) -> dict:
    """Use test-set titles as queries; a retrieved training abstract counts as relevant
    when it has the same disease label. Reports precision@k and mean average precision@k."""
    _, test, _ = load_corpus()
    queries = test.sample(n=min(n_queries, len(test)), random_state=seed)
    index = _index()
    labels = index.docs["label"].values
    out = {}
    for method in ("tfidf", "bm25"):
        precisions, aps = [], []
        for _, q in queries.iterrows():
            top = np.argsort(index.scores(q["title"], method))[::-1][:k]
            rel = (labels[top] == q["label"]).astype(float)
            precisions.append(rel.mean())
            hits = np.cumsum(rel)
            aps.append(float((hits / np.arange(1, k + 1) * rel).sum() / max(rel.sum(), 1)))
        out[method] = {"precision_at_k": float(np.mean(precisions)), "map_at_k": float(np.mean(aps))}
    baseline = float((test["label"].value_counts(normalize=True) ** 2).sum())
    return to_jsonable({"queries": len(queries), "k": k, "results": out, "random_baseline_precision": baseline,
                        "protocol": "Queries are titles of held-out test abstracts; relevance = same disease label "
                                    "as the query abstract (a proxy for topical relevance)."})
