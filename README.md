# PneumoCare

A full-stack healthcare analytics course project. The main module is **chest X-ray pneumonia detection**
(ResNet-18 transfer learning with Grad-CAM). All **15 Healthcare Analytics tutorials (T1–T15)** are implemented
as working modules in the same application.

> **Educational project. Not a medical device.** The pneumonia model was trained on a public paediatric X-ray
> dataset and has not been clinically validated.

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite 5, Tailwind CSS 3, React Router 6, Axios, Recharts, Lucide |
| Backend | Python 3.11, FastAPI, Uvicorn, Pydantic 2, SQLAlchemy 2, SQLite |
| ML / analytics | PyTorch 2.5 (CPU), torchvision, OpenCV, Pillow, NumPy, pandas, scikit-learn, SciPy, spaCy 3.7 |
| Testing | pytest + HTTPX (FastAPI TestClient), Vitest + React Testing Library, ruff, `tsc` |

---

## 1. What is in the application

| Page | Tutorials | What it does (all results come from the backend) |
|---|---|---|
| Overview | – | System status, dataset availability, model status, tutorial progress, recent analyses |
| Pneumonia Detection | Main, T6 | Upload an X-ray → prediction, probability vs threshold, Grad-CAM overlay, prediction history |
| Model Performance | Main | Test metrics at two thresholds, confusion matrix, ROC curve, probability histogram, training curves, leakage checks |
| Image Laboratory | T4, T5, T6 | Enhancement, restoration (PSNR/SSIM), filters, thresholds, classical lung-field segmentation (watershed + plausibility checks), compression, SIFT, RANSAC, CNN feature maps, embedding map, visualisation, Grad-CAM |
| EHR Data Explorer | T1, T2, T3 | Schema and statistics, cleaning pipeline with before/after and ground-truth recovery, missingness, duplicates, imputation comparison, quality score, CSV upload |
| Operations Analytics | T7, T8, T9 | Patient-volume forecasting and workload-based staffing, KPIs with significance tests, demand heat-map, provider and bed utilisation, scenario comparison |
| Clinical Text | T10, T11, T12 | TF-IDF classifiers, k-means clustering, spaCy entity extraction with negation, TF-IDF/BM25 ranking with evaluation, Lesk abbreviation disambiguation |
| Advanced Analytics | T13, T14, T15 | PCA (scree, loadings, 2-D/3-D scores), NHANES adult-weight exploration, BMI regression and obesity classification |
| Tutorial Library | T1–T15 | Objectives, concepts, inputs/outputs, endpoints, data readiness, links into each workspace |
| Datasets & Models | – | Registry with sources, licences, access requirements and on-disk availability |
| Reports | – | Export results as JSON, CSV or PDF |

## 2. Datasets

| Domain | Dataset | Access | Used by |
|---|---|---|---|
| Imaging | **PneumoniaMNIST** (MedMNIST+, 224×224), derived from Kermany et al. 2018 | Public, scripted download | Pneumonia model, T4–T6 |
| Imaging (optional) | Kaggle *Chest X-Ray Images (Pneumonia)* | **Optional**; needs your own Kaggle account (`scripts/download_kaggle.py`) | External evaluation only (`--source kaggle`) |
| EHR / operations | **Synthea** synthetic sample (CSV) – fully synthetic | Public, scripted download | T1–T3, T7–T9, T13 |
| Clinical text | **Medical Abstracts TC Corpus** (5 disease classes) | Public, scripted download | T10–T12 |
| Clinical text | Abbreviation examples – **hand-written, synthetic** | Bundled in code | T12 |
| Lifestyle | **NHANES 2017–2018** (DEMO, BMX, PAQ, SLQ, SMQ, ALQ, DR1TOT) | Public, scripted download | T13–T15 |

The Kaggle dataset was **not** downloaded, because it needs authentication and no credentials were available.
It is **optional**: the included model was trained on PneumoniaMNIST (the same source images), and the app never
switches to Kaggle on its own. To install it, see `scripts/download_kaggle.py` in
[docs/dataset_registry.md](docs/dataset_registry.md).

## 3. Setup (Windows, PowerShell)

Prerequisites: Python 3.11, Node.js 20+ (tested with Node 24), Git. A GPU is not needed.

```powershell
# from the project root
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt     # CPU PyTorch wheels (~250 MB)
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
cd ..

# download the public datasets (~285 MB in total) and export demo X-rays
backend\.venv\Scripts\python.exe scripts\download_datasets.py
backend\.venv\Scripts\python.exe scripts\export_sample_images.py

# frontend dependencies
cd frontend
npm install
cd ..
```

macOS/Linux: use `backend/.venv/bin/python` instead of `backend\.venv\Scripts\python.exe`.

### Train and evaluate the pneumonia model

```powershell
backend\.venv\Scripts\python.exe scripts\inspect_pneumonia_dataset.py   # class balance + leakage checks
backend\.venv\Scripts\python.exe scripts\train_pneumonia.py --epochs 6  # ~40 min on a laptop CPU
backend\.venv\Scripts\python.exe scripts\evaluate_pneumonia.py          # writes evaluation.json + plots
```

For a quick smoke test, run `scripts\train_pneumonia.py --epochs 1 --max-train 256 --max-eval 128`.
In Google Colab, use a GPU runtime. The scripts detect CUDA automatically.

The text classifiers (T10) and adult-weight models (T15) are trained from buttons in the UI (each takes about a minute).

## 4. Run the application

Use two terminals:

```powershell
# terminal 1 - API on http://127.0.0.1:8000  (interactive docs at /docs)
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# terminal 2 - web app on http://localhost:5173
cd frontend
npm run dev
```

The Vite dev server proxies `/api` to the backend, so no CORS setup is needed. Open http://localhost:5173.

## 5. Tests and checks

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -v        # 85 tests
.\.venv\Scripts\python.exe -m ruff check app tests ..\scripts

cd ..\frontend
npm run typecheck                               # tsc
npm test                                        # 15 Vitest tests
npm run build
```

Tests that need a downloaded dataset are **skipped** (not failed) when it is missing. Results of the last run are in
[docs/testing_report.md](docs/testing_report.md).

Audit and end-to-end checks (run from the project root; the last three need the backend running):

```powershell
backend\.venv\Scripts\python.exe scripts\audit_pneumonia.py      # recompute model metrics independently
backend\.venv\Scripts\python.exe scripts\verify_tutorials.py     # run the T1-T15 workflows through the API (~2 min)
backend\.venv\Scripts\python.exe scripts\check_live_api.py       # error handling, upload validation, all exports
backend\.venv\Scripts\python.exe scripts\build_tutorial_docs.py  # refresh docs/tutorials from the verification results
```

A tutorial shows **verified** in the app only after `verify_tutorials.py` has passed all of its checks. The dashboard's
run counts are real backend executions; an identical run repeated within 5 s (for example from React StrictMode) is
ignored.

## 6. Project layout

```text
PneumoCare/
├── backend/
│   ├── app/
│   │   ├── api/routes/     # FastAPI routers: system, screening, ehr, imaging, text, operations, analytics, reports
│   │   ├── core/           # settings, error envelope, shared helpers
│   │   ├── db/             # SQLAlchemy engine/session
│   │   ├── models/         # ORM tables (analysis_runs, predictions, uploaded_datasets)
│   │   ├── schemas/        # Pydantic request/response models
│   │   ├── services/       # analytics per tutorial (independent of HTTP)
│   │   └── ml/pneumonia/   # data, model, metrics, train, evaluate, Grad-CAM, inference
│   └── tests/              # pytest suite
├── frontend/src/
│   ├── components/         # UI kit, charts, image widgets
│   ├── layouts/            # sidebar shell
│   ├── pages/              # one page per module
│   ├── hooks/ services/ types/ utils/ test/
├── scripts/                # download, inspect, train, evaluate, export samples
├── data/                   # raw/ processed/ sample/ uploads/  (raw data is git-ignored)
├── models/                 # checkpoints (git-ignored) + metrics JSON/plots
├── reports/                # reports saved with ?save=true
└── docs/                   # project, ML, tutorial and testing documentation
```

## 7. Key results (from the real runs, see the docs for details)

| Module | Result |
|---|---|
| Pneumonia, test set (624 images), validation-chosen threshold 0.029 | accuracy 0.930, precision 0.900, recall 0.997, specificity 0.816, F1 0.947, ROC-AUC 0.994 |
| Pneumonia, same test set at threshold 0.5 (reported for comparison) | accuracy 0.968, recall 0.979, specificity 0.949, F1 0.974 |
| T10 text classification (2,888 test abstracts, 5 classes) | best macro-F1 0.578 (Complement NB); chance ≈ 0.23 accuracy |
| T12 retrieval (200 test titles as queries) | P@10 0.457 (BM25) vs 0.438 (TF-IDF) |
| T15 BMI regression / obesity classification | R² 0.115 (random forest), ROC-AUC 0.699 |

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| UI shows "Cannot reach the PneumoCare API" | Start the backend on port 8000; check `http://127.0.0.1:8000/api/health` |
| "Dataset not available" (HTTP 409) | Run `scripts/download_datasets.py --only <name>` |
| "Model not available" (HTTP 409) | Train with `scripts/train_pneumonia.py`, then run `evaluate_pneumonia.py` |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm` (the app falls back to a blank pipeline + rules) |
| pip fails on `torch` | Make sure Python is 3.11 64-bit; `requirements.txt` uses the official CPU wheel index |
| Slow or locked files inside OneDrive | Pause OneDrive sync while installing, or move the project outside OneDrive |
| A joblib "could not find the number of physical cores" traceback is printed | Harmless warning from scikit-learn's joblib on some Windows setups |
| Port already in use | Change `--port` for Uvicorn; set `VITE_BACKEND_URL` in `frontend/.env` |

## 9. Documentation

- [docs/project_documentation.md](docs/project_documentation.md): architecture, modules, database, workflows
- [docs/ml_documentation.md](docs/ml_documentation.md): pneumonia pipeline, training, evaluation
- [docs/tutorial_mapping.md](docs/tutorial_mapping.md): T1–T15 → modules, endpoints, files
- [docs/tutorials/T1.md … T15.md](docs/tutorials/): one guide per tutorial
- [docs/model_inventory.md](docs/model_inventory.md): every model and algorithm, where it is stored, what uses it
- [docs/dataset_registry.md](docs/dataset_registry.md), [docs/testing_report.md](docs/testing_report.md), [docs/progress_log.md](docs/progress_log.md)
