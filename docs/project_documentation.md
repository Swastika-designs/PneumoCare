# PneumoCare: Project Documentation

## 1. Purpose and scope

PneumoCare is a course project for Healthcare Analytics. It combines:

1. a **main project**: pneumonia detection from chest X-rays using transfer learning, with Grad-CAM explanations;
2. **15 tutorial modules (T1–T15)**, each implemented as a working analysis with its own workspace, API
   endpoints, tests and documentation.

The data domains (images, EHR, operations, clinical text, lifestyle survey) stay **separate**. They are combined
through shared navigation, a shared backend, shared UI components, one analysis history and one reports service.
They are not forced into a single model.

## 2. Architecture

```text
Browser (React + TypeScript)
   │  axios, JSON / multipart
   ▼
Vite dev server ── proxy /api ──►  FastAPI (Uvicorn)
                                     │
             ┌───────────────────────┼─────────────────────────────┐
             ▼                       ▼                             ▼
     api/routes/*  ──calls──►  services/*  (pandas, sklearn,   ml/pneumonia/*  (PyTorch)
     (HTTP only)               spaCy, OpenCV)                    data · model · train · evaluate
             │                       │                             · gradcam · inference
             ▼                       ▼                             ▼
      SQLite (metadata)       data/raw, data/sample           models/ (checkpoints, metrics JSON, plots)
```

Design rules:

- **Routes stay thin.** A route validates input, calls a service, records the run in history, and returns JSON.
  All analysis lives in `services/` or `ml/`, so the same code is used by tests, scripts and the API.
- **The ML pipeline is independent of the API.** `scripts/train_pneumonia.py` and `scripts/evaluate_pneumonia.py`
  use `app/ml/pneumonia/*` directly. The API only loads the finished checkpoint.
- **Availability is computed, never hard-coded.** The dataset and model registries check the files on disk.
  A missing dataset produces HTTP 409 `dataset_unavailable`; a missing model produces 409 `model_unavailable`.
- **Errors use one consistent format.** Every error returns `{"error": {"code", "message", "details"}}`, and the
  frontend shows `message` as is.

### Backend modules

| Path | Responsibility |
|---|---|
| `app/main.py` | App factory, CORS, error handlers, router registration, DB initialisation on startup |
| `app/core/config.py` | Settings (paths, DB URL, upload limit) from environment / `.env` |
| `app/core/errors.py` | `AppError` hierarchy → JSON error envelope; request-validation errors use the same format |
| `app/core/utils.py` | JSON-safe conversion (NaN → null), PNG data URLs, image upload validation |
| `app/db/database.py` | Engine/session factory; `configure()` lets tests use a temporary DB |
| `app/models/entities.py` | ORM tables |
| `app/schemas/api.py` | Pydantic models (prediction response, text/rank/WSD/scenario requests) |
| `app/services/registry.py` | Dataset and model registry |
| `app/services/tutorials.py` | T1–T15 catalogue (objective, concepts, inputs, outputs, endpoints, route) |
| `app/services/history.py` | Records each analysis run |
| `app/services/ehr_data.py` | Synthea loading, derived patient profile, teaching dataset, CSV uploads |
| `app/services/ehr_analysis.py` | T1 profiling, T2 cleaning/validation, T3 missingness/duplicates/imputation/quality |
| `app/services/imaging.py` | T4 processing and classical lung-field segmentation, T5 SIFT/RANSAC/CNN features, T6 visualisation |
| `app/services/operations.py` | T7 forecasting and staffing, T8 KPIs and tests, T9 utilisation and scenarios |
| `app/services/text_analytics.py` | T10 classification and clustering, T12 ranking and evaluation |
| `app/services/clinical_ner.py` | T11 spaCy pipeline, clinical EntityRuler, negation |
| `app/services/wsd.py`, `wsd_data.py` | T12 Lesk disambiguation and labelled synthetic examples |
| `app/services/dimensionality.py` | T13 PCA |
| `app/services/lifestyle.py` | T14 NHANES exploration, T15 models |
| `app/services/reports.py` | CSV / JSON / PDF (ReportLab) export |
| `app/ml/pneumonia/*` | Main project ML pipeline (see `ml_documentation.md`) |

### API overview

| Prefix | Endpoints |
|---|---|
| `/api` | `GET health`, `GET system/status`, `GET history`, `GET datasets`, `GET models`, `GET tutorials`, `GET tutorials/{id}` |
| `/api/screening` | `GET status`, `POST predict` (multipart `file`, `?gradcam=`, `?context=T6` only from the T6 workflow), `GET predictions` |
| `/api/models/pneumonia` | `GET evaluation`, `GET training`, `GET dataset`, `GET plots/{name}` |
| `/api/ehr` | `GET sources`, `POST upload`, `GET overview`, `GET column`, `GET teaching-log`, `GET clean`, `GET missingness`, `GET duplicates`, `GET imputation`, `GET quality` |
| `/api/imaging` | `GET samples`, `GET samples/{name}`, `POST enhance/restore/filters/threshold/segment/compress/sift/match/cnn-features/visualize`, `GET feature-embedding` |
| `/api/text` | `GET corpus`, `GET corpus/{id}`, `POST preprocess/train/classify/entities/rank/wsd`, `GET metrics/cluster/sample-notes/entity-stats/rank-evaluation/wsd-evaluation` |
| `/api/operations` | `GET forecast`, `GET kpis`, `GET utilization`, `POST scenarios` |
| `/api/analytics` | `GET pca/sources`, `GET pca`, `GET weight/explore`, `POST weight/train`, `GET weight/results` |
| `/api/reports` | `GET` (list), `GET {id}?format=json|csv|pdf&save=` |

Interactive OpenAPI docs are served at `http://127.0.0.1:8000/docs`.

### Frontend modules

| Path | Responsibility |
|---|---|
| `src/layouts/AppLayout.tsx` | Responsive sidebar (drawer on mobile), live API status |
| `src/pages/*` | One page per module; tabs are stored in `?tab=` so tutorial links open the right workspace |
| `src/components/ui` | Card, Badge, StatCard, Tabs, Field/Select, `AsyncView` (loading/error/empty/content), `DataTable` |
| `src/components/charts` | Fixed categorical palette, sequential/diverging scales, `MatrixHeatmap`, `Scatter3D` |
| `src/components/images.tsx` | File drop with client-side validation, image panels, compare slider |
| `src/hooks/useApi.ts` | `useAction` (on-demand) and `useFetch` (automatic); ignores stale responses |
| `src/services/api.ts` | Axios instance and error-envelope parsing |

Charts follow one palette with a fixed categorical order. Bars have rounded data ends, every chart has a tooltip,
multi-series charts have a legend, and no chart uses two y-axes.

## 3. Database

SQLite file `data/pneumocare.db` (created on startup). It stores **metadata only**. Images, CSVs and checkpoints
stay on disk.

| Table | Columns | Written by |
|---|---|---|
| `analysis_runs` | id, module, tutorial, title, dataset, summary (JSON), created_at | Every analysis endpoint |
| `predictions` | id, filename, stored_path, predicted_label, probability_pneumonia, threshold, model_name, created_at | `POST /api/screening/predict` |
| `uploaded_datasets` | id, name, stored_path, rows, columns, created_at | `POST /api/ehr/upload` |

Uploaded files are saved under `data/uploads/` with sanitised names. Delete `data/pneumocare.db` to reset the history.

**Run counts.** `history.record()` ignores an identical run recorded less than 5 seconds earlier. Together with the
`useFetch` guard in the frontend, this stops React StrictMode (which runs effects twice in development) from doubling
the counts. Plain pneumonia predictions are not counted as T6; only `context=T6` calls from the Image Lab's
Visualise tab are. `scripts/repair_history.py` applied these rules to the history recorded before the fix (6 duplicate
runs removed, 10 predictions un-tagged from T6).

**Tutorial status.** The status shown in the Tutorial Library is evidence-based: *verified* when
`reports/tutorial_verification.json` (written by `scripts/verify_tutorials.py`) shows that all checks passed, *not
verified* when there is no result, and *verification failed* otherwise.

## 4. Main workflows

**Pneumonia screening**
`download_datasets.py` → `inspect_pneumonia_dataset.py` → `train_pneumonia.py` (best epoch by validation AUC,
threshold by validation Youden J) → `evaluate_pneumonia.py` (test metrics and plots) → the API loads the checkpoint
→ the user uploads an X-ray → validation (type, size, resolution, colour warning) → prediction + Grad-CAM → stored
in `predictions`.

**Analysis modules**
The user picks data and parameters → the frontend calls an endpoint → the service computes on real data → the run
is recorded in `analysis_runs` → the UI shows charts and tables → the Reports page can export stored results.

**Reports**
`GET /api/reports/{id}?format=pdf` builds sections (text, tables, images) from stored results. If the result does
not exist yet (for example, an untrained model), it returns 409 with instructions instead of an empty file.

## 5. Security and privacy notes

- All data are public or synthetic. No real patient data is stored.
- Uploads are size-limited (10 MB), type-checked (decoded with Pillow, not trusted by extension), and saved with
  sanitised names. Sample-image requests are checked against path traversal.
- `.env`, raw data, uploads, the database and checkpoints are git-ignored.
- No authentication: this is a single-user local application.

## 6. Known limitations

- Grad-CAM shows where the model looked, not where disease is.
- Lung-field segmentation (T4) is classical image processing without ground truth; it misses consolidated lung and
  is checked only for plausibility. See `model_inventory.md` for every model and algorithm.
- PneumoniaMNIST images are pre-cropped and down-sampled from the Kaggle originals. Performance on full-resolution
  hospital images is unknown.
- The Synthea sample contains only 111 patients, so operations metrics are noisy and must not be compared with real
  hospital benchmarks.
- spaCy's `en_core_web_sm` is a general-domain model. Clinical entities come mainly from the rule lexicon.
  Installing scispaCy's `en_ner_bc5cdr_md` (spaCy 3.7-compatible) adds statistical DISEASE/CHEMICAL entities;
  the code uses it automatically when it is installed.
