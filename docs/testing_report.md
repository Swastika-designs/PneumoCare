# Testing Report

Last full run: **28 Oct 2026** (independent review; every row below was re-run that day except persistence, which is from the 26 Oct audit), Windows 11, Python 3.11.9, Node 24.14.1, all required datasets present,
trained checkpoint present, Kaggle dataset **not** installed (optional). Every command below was actually executed,
and the numbers are copied from its output.

## Summary

| Check | Command | Result |
|---|---|---|
| Backend tests | `backend\.venv\Scripts\python.exe -m pytest -v` | **85 passed**, 0 failed, 0 skipped |
| Backend lint | `ruff check app tests ..\scripts` (from `backend/`) | **All checks passed** |
| Frontend type check | `npx tsc -b --noEmit` | **No errors** |
| Frontend tests | `npx vitest run` | **15 passed** |
| Frontend build | `npx vite build` | **Built** (Vite's >500 kB chunk notice is informational) |
| Model audit | `scripts/audit_pneumonia.py` | **9/9 checks passed**: metrics independently recomputed, see `ml_documentation.md` §10 |
| Tutorial workflows (live API) | `scripts/verify_tutorials.py` | **15/15 tutorials verified, 64/64 checks** (`reports/tutorial_verification.json`; T4 and T6 re-run on 28 Oct) |
| Live API contract | `scripts/check_live_api.py` | **46/46 checks**: error envelope, upload validation, 24 exports parsed (`reports/live_api_checks.json`) |
| Segmentation evaluation | `scripts/evaluate_segmentation.py` | 624 test images, plausibility only (no ground truth): masks > 60% of the image 71 → 2; all checks pass 54.5% → 76.1% (`reports/segmentation_evaluation.json`) |
| Persistence | Restarted Uvicorn and compared counts | 48 runs / 11 predictions before = 48 / 11 after |
| Visual check | Headless Edge screenshots of Overview, Tutorial Library, T3 detail, Registry, Clinical Text, Operations (KPIs, planning), Advanced (PCA, models), Model Performance, EHR | Pages rendered with real data. Three chart bugs found and fixed (below) |

## Independent review (28 Oct 2026)

Run against an isolated backend (scratch copy of the database) so the real history was not changed.

| Check | Result |
|---|---|
| Browser walkthrough (headless Edge): every page and tab, every action button, an X-ray upload, all 24 report downloads, the 404 page | No console errors, no failed requests, no error alerts. Each workflow produced its expected output |
| Documented figures reproduced through the API (T4 restoration/compression, T5 RANSAC on `normal_test_0380`, T10 clustering metrics) | Identical to `docs/tutorials` |
| Static check: every `/api/...` path in the frontend and docs | All map to real routes |
| Retraining during verification (T10, T15) | Metrics identical to the documented values (deterministic seeds) |

Fixed: the Image Lab tab bar scrolled horizontally and hid the **Visualise (T6)** tab at laptop widths (now wraps);
`reports/segmentation_evaluation.json` was excluded by `.gitignore` although the docs cite it (now included).

Not covered: the frontend has no ESLint configuration (TypeScript checking only); full-page screenshots in headless
browsers can show Recharts charts as empty, so charts were checked in viewport screenshots and by counting drawn marks.

## Backend tests by file (85)

| File | Tests | Coverage |
|---|---|---|
| `test_system.py` | 10 | Health, DB tables, `/system/status` (verified count, statuses, required datasets), catalogue, tutorial 404, registry flags, validation envelope, unknown report, **history duplicate guard**, **unknown-route envelope** |
| `test_validation.py` | 7 | Empty / non-image / tiny / oversized uploads, valid PNG, colour detection, NaN → null |
| `test_pneumonia_data.py` | 6 | Missing dataset, patient-ID parsing, patient-grouped split, duplicate detection, hash stability, transforms |
| `test_pneumonia_model.py` | 12 | Model shape, checkpoint round-trip, incomplete checkpoint, metric formulas, missing checkpoint → 409, Grad-CAM, prediction schema, colour warning, any input size, **predictions count as T6 only with `context=T6`** |
| `test_kaggle.py` | 5 | **Archive install de-duplicates nested copies and skips `__MACOSX`**, rejects unrelated zips, clear missing-credentials message, **`auto` never switches to Kaggle**, **external evaluation never overwrites `evaluation.json`** |
| `test_ehr.py` | 11 | T1–T3 calculations and API, ground-truth recovery, CSV upload |
| `test_imaging.py` | 13 | T4–T5 processing, **lung segmentation: Dice > 0.85 on a phantom, dark corner/edge strip ignored, blank image flagged**, RANSAC vs least squares, API with samples, path traversal |
| `test_text.py` | 10 | NER and negation, WSD, classify without model → 409, ranking, training |
| `test_analytics.py` | 11 | PCA, forecasting, KPIs, scenarios, NHANES exploration, leak-free T15 models, reports |

## Frontend tests (15)

Navigation and dashboard (loading → real values incl. "14 / 15" verified and the "optional" dataset badge;
offline error; sidebar navigation + search, with **verified 4/4** and **not verified** badges; 404). Pneumonia
detection (no-model state, client-side file validation, upload → loading → result with Grad-CAM, backend error
shown). Other pages (Model Performance threshold toggle, EHR empty state, Operations dataset error, Reports formats,
Clinical Text tab from URL). Utilities.

## Tutorial verification (live API, real datasets)

| Tutorial | Checks | Key evidence |
|---|---|---|
| T1 | 5/5 | 111 patients profiled; histogram sums to non-missing rows; timeline for dates |
| T2 | 4/4 | 141 → 0 rule violations; 100% recovery of the 6 corrupted columns |
| T3 | 5/5 | 14.64% missing; 5 exact + 3 near duplicates found; LDL RMSE iterative 13.7 vs mean 35.1; quality 78.6 (messy) < 96.4 (original) |
| T4 | 7/7 | Denoising raises PSNR; Otsu; two lung fields; plausible masks on all 4 normal samples; JPEG size falls with quality |
| T5 | 5/5 | RANSAC corner error 0.18 px (< least squares); CNN maps 512×7×7; linear probe 0.93 |
| T6 | 3/3 | 6 visualisation panels; Grad-CAM from the trained model (`context=T6`) |
| T7 | 5/5 | Best model = lowest MAE (random forest 5.71 vs naive 6.62); time-ordered split; FTE formula consistent |
| T8 | 4/4 | Groups add up to the total; Kruskal–Wallis p < 0.05 |
| T9 | 3/3 | 7×24 heat-map; doubling demand doubles utilisation |
| T10 | 4/4 | Best accuracy 0.586 (Complement NB); neoplasm text classified correctly; clustering metrics |
| T11 | 4/4 | 3 negations, medication–dose link, ≥ 4 vitals, corpus extraction for 5 classes |
| T12 | 4/4 | BM25 P@10 0.457 > random 0.232; `RA` → room air; corpus-only WSD 0.58 > baseline 0.42 |
| T13 | 4/4 | Variance sums to 1; unscaled PC1 dominates (scaling matters); second dataset |
| T14 | 3/3 | n = 5,120; weighted obesity 42.5% (CDC ≈ 42%) |
| T15 | 4/4 | No leaky predictors; R² 0.115 > baseline; AUC 0.699 |

## Audit findings and fixes (26 Oct 2026)

| Finding | Severity | Fix | Verified by |
|---|---|---|---|
| Every page-load analysis was recorded **twice** (React StrictMode double effects), inflating T3/T8/T13 run counts 2× | High (misleading dashboard) | `useFetch` key guard + backend 5-s duplicate guard; `repair_history.py` removed 6 duplicates | `test_history_ignores_immediate_duplicates`; dashboard counts |
| All 10 pneumonia predictions were counted as **T6 runs** although no T6 workflow ran | High (misleading) | Only `context=T6` (Image Lab Visualise) counts as T6; 10 rows un-tagged | `test_predictions_count_as_t6_only_from_visualisation` |
| T3 duplicate detection and quality report were **not recorded** as runs | Medium | Both endpoints now record T3 runs | verification run counts |
| Tutorial status was **hard-coded "implemented"** | Medium | Status derived from `verify_tutorials.py` evidence | `test_system_status_shape`, Tutorial Library |
| `load_dataset("auto")` **preferred Kaggle** when present (silent dataset switch); external evaluation would **overwrite** `evaluation.json` | High (latent) | `auto` = PneumoniaMNIST; external evaluation writes `evaluation_<source>.json` | `test_kaggle.py` |
| Kaggle shown as "missing" on the dashboard (counted against availability) | Medium | Registry `required: false`; dashboard shows required datasets 5/5 and Kaggle as "optional"; installer script added | dashboard screenshot, tests |
| T9 stacked demand chart **omitted "wellness" and "virtual"** (alphabetical cap of 8 classes) | High (wrong chart totals) | Top 7 classes by volume + "other" | screenshot |
| Vertical bar charts hid every other category label | Low | `interval={0}` on category axes | screenshot |
| BMI scatter axis fixed at 15–60 while data reached 86 | Low | Auto domain | screenshot |
| Unmatched routes returned `{"detail": ...}` instead of the error envelope | Low | Starlette HTTP exception handler | `test_unknown_route_uses_error_envelope`, live check |
| A dead backend showed "Request failed with status 500" | Low | Clear "Is the backend running?" message for proxy 5xx | screenshot of the offline state |
| T6 doc claimed Grad-CAM concentrates on the lungs (**unverified**) | Medium (documentation) | Measured: CAM mass in lungs 0.16–0.42 ≈ lung area 0.16–0.39 → claim removed, caveat added | measurement in `ml_documentation.md` §8 |
| Tutorial mapping table broke on `|` characters in endpoints | Low | Escaped in the generator | rendered table |

Run counts after the audit are genuine backend executions: two full verification passes plus page loads during
the visual checks. The history also contains the audit's own live-API test inputs (`audit_small.csv` upload,
`photo.png` colour-photo prediction). These are real runs on test inputs.

## Known warnings (not failures)

- joblib/loky may print a "could not find the number of physical cores" traceback on Windows. It is harmless.
- Vite warns about the bundle size (mostly Recharts).

## Not covered

- Full model training (about 40 min) is not repeated by the tests. The audit verifies the trained checkpoint
  instead, and `training_log.txt` records the original run.
- Kaggle download and external evaluation are tested with a synthetic archive only, because there are no credentials.
- Image Lab interactions (button-triggered) were verified through the API and unit tests, not by clicking in a browser.
- No visual-regression, load or security testing.
