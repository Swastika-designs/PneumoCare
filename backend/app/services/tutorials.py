"""Catalogue of the 15 Healthcare Analytics tutorials and where each lives in the app."""
from __future__ import annotations

import json

from app.core.config import settings
from app.services.registry import dataset_status

TUTORIALS = [
    {
        "id": "T1", "title": "Introduction to EHR and Healthcare Data Analytics", "module": "EHR Data Explorer",
        "route": "/ehr?tab=explore", "datasets": ["synthea"],
        "objective": "Load an EHR table, understand its structure and summarise it.",
        "concepts": ["EHR tables and keys", "data types", "descriptive statistics", "distributions"],
        "inputs": "Any Synthea table, the derived patient profile, or an uploaded CSV",
        "outputs": "Schema table, numeric & categorical summaries, per-column charts, preview rows",
        "endpoints": ["GET /api/ehr/sources", "GET /api/ehr/overview", "GET /api/ehr/column"],
    },
    {
        "id": "T2", "title": "EHR Preprocessing: Standardization and Data Cleaning", "module": "EHR Data Explorer",
        "route": "/ehr?tab=clean", "datasets": ["synthea"],
        "objective": "Convert types, normalise categories and validate values; compare before vs after.",
        "concepts": ["type conversion", "category normalisation", "null tokens", "validation rules",
                     "ground-truth recovery"],
        "inputs": "Teaching dataset (Synthea patients with documented injected errors) or any source",
        "outputs": "Cleaning step log, dtype changes, value-count changes, validation before/after, recovery %",
        "endpoints": ["GET /api/ehr/clean", "GET /api/ehr/teaching-log"],
    },
    {
        "id": "T3", "title": "EHR Preprocessing: Redundancy and Missing Data", "module": "EHR Data Explorer",
        "route": "/ehr?tab=missing", "datasets": ["synthea"],
        "objective": "Quantify missingness and duplication, compare imputation methods and score data quality.",
        "concepts": ["missingness patterns", "exact/near/key duplicates", "redundant columns",
                     "mean/median/KNN/iterative imputation", "data-quality dimensions"],
        "inputs": "Any EHR source (patient profile has natural lab missingness)",
        "outputs": "Missingness table & matrix, co-missing correlations, duplicate report, imputation RMSE, "
                   "quality scorecard",
        "endpoints": ["GET /api/ehr/missingness", "GET /api/ehr/duplicates", "GET /api/ehr/imputation",
                      "GET /api/ehr/quality"],
    },
    {
        "id": "T4", "title": "Biomedical Image Analysis", "module": "Biomedical Image Laboratory",
        "route": "/image-lab?tab=enhance", "datasets": ["pneumoniamnist"],
        "objective": "Enhance, restore, filter, threshold, segment and compress chest X-rays.",
        "concepts": ["histogram equalisation / CLAHE", "gamma", "noise models", "denoising filters", "PSNR/SSIM",
                     "edge detection", "Otsu & adaptive thresholds", "morphology", "marker-controlled watershed",
                     "JPEG vs PNG compression"],
        "inputs": "Sample X-ray from the dataset or an uploaded image",
        "outputs": "Processed image panels, histograms, quality metrics, compression table",
        "endpoints": ["POST /api/imaging/enhance", "POST /api/imaging/restore", "POST /api/imaging/filters",
                      "POST /api/imaging/threshold", "POST /api/imaging/segment", "POST /api/imaging/compress"],
    },
    {
        "id": "T5", "title": "SIFT, RANSAC, and CNN Feature Extraction", "module": "Biomedical Image Laboratory",
        "route": "/image-lab?tab=features", "datasets": ["pneumoniamnist"],
        "objective": "Detect and match keypoints, estimate geometry robustly, and extract deep features.",
        "concepts": ["scale-space keypoints", "128-D descriptors", "Lowe ratio test", "homography", "RANSAC",
                     "CNN feature maps", "transfer features / linear probe"],
        "inputs": "One image (matched against a known synthetic transform) or two images",
        "outputs": "Keypoint plots, match/inlier visualisations, homography error, feature maps, embedding map",
        "endpoints": ["POST /api/imaging/sift", "POST /api/imaging/match", "POST /api/imaging/cnn-features",
                      "GET /api/imaging/feature-embedding"],
    },
    {
        "id": "T6", "title": "Biomedical Image Visualization", "module": "Biomedical Image Laboratory",
        "route": "/image-lab?tab=visualize", "datasets": ["pneumoniamnist"],
        "objective": "Compare and visualise images, histograms, overlays and model attention.",
        "concepts": ["side-by-side comparison", "histogram/CDF", "pseudo-colour", "overlays", "intensity profiles",
                     "Grad-CAM"],
        "inputs": "Sample or uploaded X-ray; trained pneumonia model for Grad-CAM",
        "outputs": "Comparison slider, overlays, profile plot, Grad-CAM heat-map",
        "endpoints": ["POST /api/imaging/visualize", "POST /api/screening/predict"],
    },
    {
        "id": "T7", "title": "Patient Prediction for Improved Healthcare Staffing", "module": "Healthcare Operations",
        "route": "/operations?tab=forecast", "datasets": ["synthea"],
        "objective": "Forecast patient volume and convert it into a workload-based staffing estimate.",
        "concepts": ["time-series aggregation", "lag features", "time-based train/test split",
                     "baselines vs ML", "MAE/RMSE/MAPE", "workload -> FTE"],
        "inputs": "Synthea encounters; period (week/month), encounter class, staffing assumptions",
        "outputs": "Model comparison, test-period predictions, forecast with required FTE",
        "endpoints": ["GET /api/operations/forecast"],
    },
    {
        "id": "T8", "title": "Healthcare Strategic Planning Part 1", "module": "Healthcare Operations",
        "route": "/operations?tab=kpis", "datasets": ["synthea"],
        "objective": "Compute healthcare KPIs, compare groups and follow trends.",
        "concepts": ["KPIs (cost, coverage, LOS, readmission)", "group comparison", "Kruskal-Wallis / Mann-Whitney",
                     "trend dashboards"],
        "inputs": "Synthea encounters + patients + payers + organizations",
        "outputs": "KPI cards, group table/chart with significance test, monthly & yearly trends",
        "endpoints": ["GET /api/operations/kpis"],
    },
    {
        "id": "T9", "title": "Healthcare Strategic Planning Part 2", "module": "Healthcare Operations",
        "route": "/operations?tab=planning", "datasets": ["synthea"],
        "objective": "Analyse demand patterns and resource utilisation; compare what-if scenarios.",
        "concepts": ["demand heat-maps", "growth rates", "provider workload", "bed utilisation",
                     "scenario analysis"],
        "inputs": "Synthea encounters; bed capacity; scenario definitions",
        "outputs": "Demand heat-map and trends, workload distribution, bed utilisation, scenario comparison",
        "endpoints": ["GET /api/operations/utilization", "POST /api/operations/scenarios"],
    },
    {
        "id": "T10", "title": "Clinical Text Classification and Clustering", "module": "Clinical Text Analytics",
        "route": "/clinical-text?tab=classify", "datasets": ["medical_abstracts"],
        "objective": "Preprocess clinical text, classify it with TF-IDF models and cluster it without labels.",
        "concepts": ["tokenisation, stop words, lemmatisation", "TF-IDF", "logistic regression / SVM / NB",
                     "macro-F1", "k-means", "silhouette, ARI, NMI, purity"],
        "inputs": "Medical Abstracts corpus; free text to classify",
        "outputs": "Metrics per model and class, confusion matrix, explanations, clusters and their terms",
        "endpoints": ["POST /api/text/preprocess", "POST /api/text/train", "GET /api/text/metrics",
                      "POST /api/text/classify", "GET /api/text/cluster"],
    },
    {
        "id": "T11", "title": "Clinical Information Extraction and Entity Recognition",
        "module": "Clinical Text Analytics", "route": "/clinical-text?tab=ner", "datasets": ["medical_abstracts"],
        "objective": "Extract problems, medications, doses, vitals, labs and procedures with negation.",
        "concepts": ["spaCy pipelines", "statistical vs rule-based NER", "EntityRuler", "NegEx-style negation",
                     "structured output"],
        "inputs": "Synthetic sample notes, corpus abstracts or pasted text",
        "outputs": "Highlighted entities, entity table, structured JSON summary, corpus entity statistics",
        "endpoints": ["POST /api/text/entities", "GET /api/text/entity-stats", "GET /api/text/sample-notes"],
    },
    {
        "id": "T12", "title": "Clinical Text Ranking and Word Disambiguation", "module": "Clinical Text Analytics",
        "route": "/clinical-text?tab=ranking", "datasets": ["medical_abstracts", "wsd_examples"],
        "objective": "Rank documents for a query and disambiguate clinical abbreviations from context.",
        "concepts": ["vector space model", "cosine similarity", "BM25", "precision@k / MAP", "Lesk algorithm"],
        "inputs": "Free-text query; sentence containing an ambiguous abbreviation",
        "outputs": "Ranked documents with matched terms, retrieval evaluation, sense scores and accuracy",
        "endpoints": ["POST /api/text/rank", "GET /api/text/rank-evaluation", "POST /api/text/wsd",
                      "GET /api/text/wsd-evaluation"],
    },
    {
        "id": "T13", "title": "Dimensionality Reduction for EDA", "module": "Advanced Analytics",
        "route": "/advanced?tab=pca", "datasets": ["nhanes", "synthea"],
        "objective": "Scale features, run PCA and interpret components in 2D/3D.",
        "concepts": ["standardisation", "covariance & eigenvectors", "explained variance", "loadings",
                     "scree plot"],
        "inputs": "NHANES adults, Synthea patient profile or uploaded CSV",
        "outputs": "Scree/cumulative variance, loadings, 2D & rotatable 3D score plots, correlation matrix",
        "endpoints": ["GET /api/analytics/pca"],
    },
    {
        "id": "T14", "title": "Behavioral Factors Influencing Adult Weight Part 1", "module": "Advanced Analytics",
        "route": "/advanced?tab=weight-eda", "datasets": ["nhanes"],
        "objective": "Explore adult BMI and lifestyle factors with descriptive statistics and correlations.",
        "concepts": ["recoding survey variables", "BMI categories", "distributions", "Spearman correlation",
                     "survey weights"],
        "inputs": "NHANES 2017-2018 adults (>= 20 y, not pregnant)",
        "outputs": "Descriptive table, histograms, correlation heat-map, BMI by group with tests",
        "endpoints": ["GET /api/analytics/weight/explore"],
    },
    {
        "id": "T15", "title": "Behavioral Factors Influencing Adult Weight Part 2", "module": "Advanced Analytics",
        "route": "/advanced?tab=weight-model", "datasets": ["nhanes"],
        "objective": "Model BMI and obesity from lifestyle factors and interpret the models.",
        "concepts": ["linear & logistic regression", "random forests", "cross-validation", "R2 / AUC",
                     "coefficients & odds ratios", "permutation importance", "target leakage"],
        "inputs": "NHANES adults; predictors exclude weight/height/waist",
        "outputs": "Regression & classification metrics vs baselines, ROC, coefficients, importances",
        "endpoints": ["POST /api/analytics/weight/train", "GET /api/analytics/weight/results"],
    },
]


def verification_results() -> dict:
    """Results written by scripts/verify_tutorials.py (empty if it has never been run)."""
    path = settings.reports_dir / "tutorial_verification.json"
    try:
        return json.loads(path.read_text()).get("tutorials", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def catalogue(run_counts: dict[str, int] | None = None) -> list[dict]:
    available = {d["id"]: d["available"] for d in dataset_status()}
    verified = verification_results()
    out = []
    for t in TUTORIALS:
        item = dict(t)
        v = verified.get(t["id"])
        # Status is evidence-based: "verified" only if the workflow checks passed against the live API.
        item["status"] = "not verified" if v is None else ("verified" if v["passed"] else "verification failed")
        item["verification"] = None if v is None else {
            "passed": v["passed"], "verified_at": v["verified_at"], "error": v.get("error"),
            "checks_passed": sum(c["passed"] for c in v["checks"]), "checks_total": len(v["checks"]),
            "checks": v["checks"]}
        item["data_ready"] = all(available.get(d, False) for d in t["datasets"])
        item["runs"] = (run_counts or {}).get(t["id"], 0)
        item["doc"] = f"docs/tutorials/{t['id']}.md"
        out.append(item)
    return out
