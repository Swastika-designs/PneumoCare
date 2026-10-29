# Tutorial Mapping (T1–T15)

All 15 tutorials are implemented as working modules in one application. 15 of 15 passed their
workflow verification (`scripts/verify_tutorials.py`, results in `reports/tutorial_verification.json`). They share navigation, the FastAPI
backend, the SQLite analysis history, the UI components and the reports service. The data domains stay separate.

| Tutorial | Title | App module / tab | Main backend code | Endpoints | Data | Verified |
|---|---|---|---|---|---|---|
| [T1](tutorials/T1.md) | Introduction to EHR and Healthcare Data Analytics | EHR Data Explorer → **Explore** tab | `backend/app/services/ehr_data.py` | `GET /api/ehr/sources` … | Synthea | yes |
| [T2](tutorials/T2.md) | EHR Preprocessing: Standardization and Data Cleaning | EHR Data Explorer → **Standardise & clean** tab | `backend/app/services/ehr_data.py` | `GET /api/ehr/teaching-log` … | Synthea (teaching variant) | yes |
| [T3](tutorials/T3.md) | EHR Preprocessing: Redundancy and Missing Data | EHR Data Explorer → **Missing data & redundancy** tab | `backend/app/services/ehr_analysis.py` | `GET /api/ehr/missingness` … | Synthea | yes |
| [T4](tutorials/T4.md) | Biomedical Image Analysis | Image Laboratory → **Enhance / Restore / Filter / Threshold / Segment / Compress** tabs | `backend/app/services/imaging.py` | `POST /api/imaging/{enhance\|restore\|filters\|threshold\|segment\|compress}` … | PneumoniaMNIST | yes |
| [T5](tutorials/T5.md) | SIFT, RANSAC, and CNN Feature Extraction | Image Laboratory → **SIFT · RANSAC · CNN** tab | `backend/app/services/imaging.py` | `POST /api/imaging/sift` … | PneumoniaMNIST | yes |
| [T6](tutorials/T6.md) | Biomedical Image Visualization | Image Laboratory → **Visualise** tab | `backend/app/services/imaging.py` | `POST /api/imaging/visualize?colormap=&row=` … | PneumoniaMNIST + trained model | yes |
| [T7](tutorials/T7.md) | Patient Prediction for Improved Healthcare Staffing | Operations Analytics → **Volume & staffing** tab | `backend/app/services/operations.py` | `GET /api/operations/forecast?freq=&encounter_class=&horizon=&hours_per_fte=&utilization_target=` | Synthea encounters | yes |
| [T8](tutorials/T8.md) | Healthcare Strategic Planning Part 1 | Operations Analytics → **KPIs & comparisons** tab | `backend/app/services/operations.py` | `GET /api/operations/kpis?group_by=&metric=&start_year=` | Synthea encounters | yes |
| [T9](tutorials/T9.md) | Healthcare Strategic Planning Part 2 | Operations Analytics → **Utilisation & scenarios** tab | `backend/app/services/operations.py` | `GET /api/operations/utilization?beds=` … | Synthea encounters | yes |
| [T10](tutorials/T10.md) | Clinical Text Classification and Clustering | Clinical Text → **Classification** and **Clustering** tabs | `backend/app/services/text_analytics.py` | `POST /api/text/preprocess` … | Medical Abstracts | yes |
| [T11](tutorials/T11.md) | Clinical Information Extraction and Entity Recognition | Clinical Text → **Entity extraction** tab | `backend/app/services/clinical_ner.py` | `GET /api/text/sample-notes` … | Synthetic notes + Medical Abstracts | yes |
| [T12](tutorials/T12.md) | Clinical Text Ranking and Word Disambiguation | Clinical Text → **Ranking & disambiguation** tab | `backend/app/services/text_analytics.py` | `POST /api/text/rank` … | Medical Abstracts + synthetic WSD set | yes |
| [T13](tutorials/T13.md) | Dimensionality Reduction for EDA | Advanced Analytics → **PCA & EDA** tab | `backend/app/services/dimensionality.py` | `GET /api/analytics/pca/sources` … | NHANES / Synthea profile | yes |
| [T14](tutorials/T14.md) | Behavioral Factors Influencing Adult Weight Part 1 | Advanced Analytics → **Adult weight: exploration** tab | `backend/app/services/lifestyle.py` | `GET /api/analytics/weight/explore?variable=` | NHANES 2017-2018 | yes |
| [T15](tutorials/T15.md) | Behavioral Factors Influencing Adult Weight Part 2 | Advanced Analytics → **Adult weight: models** tab | `backend/app/services/lifestyle.py` | `POST /api/analytics/weight/train` … | NHANES 2017-2018 | yes |

## Module overview

| App module | Route | Tutorials |
|---|---|---|
| Pneumonia Detection + Model Performance | `/pneumonia`, `/model-performance` | Main project, T6 (Grad-CAM) |
| Biomedical Image Laboratory | `/image-lab` | T4, T5, T6 |
| EHR Data Explorer | `/ehr` | T1, T2, T3 |
| Healthcare Operations Analytics | `/operations` | T7, T8, T9 |
| Clinical Text Analytics | `/clinical-text` | T10, T11, T12 |
| Advanced Analytics | `/advanced` | T13, T14, T15 |
| Tutorial Library | `/tutorials`, `/tutorials/:id` | all (catalogue served by `GET /api/tutorials`) |

Each tutorial card in the Tutorial Library opens the exact tab (`?tab=…`). Every run is logged with its tutorial ID,
so the Overview page shows real progress (runs per tutorial). See the per-tutorial guides in `docs/tutorials/`.

## Shared infrastructure used by every tutorial

- `AsyncView` loading/error/empty states and the `{error: {code, message}}` envelope (`dataset_unavailable`,
  `model_unavailable`, `validation_failed`, `invalid_request`, `not_found`).
- `history.record()` writes to `analysis_runs`, which drives the dashboard and the *Analysis history* report.
- `reports.py` exports EHR quality (T2–T3), operations KPIs (T8), text classification (T10) and weight models (T15)
  as CSV, JSON or PDF.
