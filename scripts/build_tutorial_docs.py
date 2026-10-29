# ruff: noqa: E501  (long prose strings are the documentation content)
"""Regenerate docs/tutorials/T1.md ... T15.md and docs/tutorial_mapping.md.

The content (objective, method, results) is kept here; each guide's status line and its
"Verification" section are read from reports/tutorial_verification.json, which is written by
scripts/verify_tutorials.py. Re-run this after re-verifying:  python scripts/build_tutorial_docs.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
VERIFY = ROOT / "reports" / "tutorial_verification.json"
VERIFIED = json.loads(VERIFY.read_text())["tutorials"] if VERIFY.exists() else {}

T = {}

T["T1"] = dict(
    title="Introduction to EHR and Healthcare Data Analytics",
    where="EHR Data Explorer → **Explore** tab (`/ehr?tab=explore`)",
    objective="Load an electronic health record (EHR) table, understand its structure and describe it with statistics and charts.",
    concepts=[
        ("EHR tables", "Synthea exports one CSV per entity: `patients`, `encounters`, `conditions`, `observations`, `medications`, and so on. Rows are linked by keys: `PATIENT` in encounters refers to `Id` in patients."),
        ("Storage type vs semantic type", "pandas stores a column as `object`, `float64`, `int64` and so on. That is not the same as what it *means*. The app infers a **semantic type** (identifier, numeric, binary, categorical, datetime-like text, free text) from the column name, dtype and cardinality."),
        ("Descriptive statistics", "Numeric columns: count, mean, SD, min, quartiles, max, skewness. Categorical columns: number of levels, most frequent value."),
        ("Distributions", "Histograms for numeric columns (with the number of 1.5×IQR outliers), bar charts of the top categories, and records per year for dates."),
    ],
    dataset="Synthea synthetic EHR (111 patients, 5,059 encounters, 53,961 observations). The derived **patient profile** joins demographics, counts of encounters/conditions/medications and the latest value of 19 vitals/labs per patient. You can also upload any CSV (≤ 10 MB, ≤ 200,000 rows).",
    method=[
        "Read the CSV with pandas (results cached by file modification time).",
        "Per column: dtype, inferred semantic type, non-null / missing counts, unique values, three examples.",
        "`describe()` for numeric columns plus skewness; value counts for categorical columns.",
        "On request, a per-column profile (histogram, bar chart or timeline).",
        "Record the run in `analysis_runs` (tutorial T1).",
    ],
    files="`backend/app/services/ehr_data.py` (loading, patient profile, uploads), `backend/app/services/ehr_analysis.py` (`overview`, `semantic_type`, `column_profile`), `frontend/src/pages/EhrExplorer.tsx` (`ExploreTab`)",
    endpoints="`GET /api/ehr/sources`, `POST /api/ehr/upload`, `GET /api/ehr/overview?source=`, `GET /api/ehr/column?source=&column=`",
    outputs="Stat cards (rows, columns, missing cells, duplicate rows, memory), schema table, column distribution chart, numeric and categorical summaries, preview of the first 10 rows.",
    results="Patient profile: 111 rows × 32 columns. Lab values such as HbA1c, glucose and creatinine are missing for 66–68% of patients, because the simulator only orders those panels for some patients.",
    run="Open **EHR Data Explorer**, keep the *Explore* tab, choose a dataset (for example *Synthea – encounters*), then pick a column in *Column distribution*.",
    tests="`tests/test_ehr.py::test_overview_counts`, `test_column_profile_numeric_histogram`, `test_ehr_api_endpoints`, `test_csv_upload_validation`",
    limits="The data are synthetic, so distributions follow Synthea's disease modules rather than a real population. Semantic-type inference is heuristic and can mislabel unusual columns.",
)

T["T2"] = dict(
    title="EHR Preprocessing: Standardization and Data Cleaning",
    where="EHR Data Explorer → **Standardise & clean** tab (`/ehr?tab=clean`)",
    objective="Turn inconsistent raw EHR values into a clean, typed, validated table, and prove the effect with before/after comparisons.",
    concepts=[
        ("Data-type conversion", "Strings such as `'$12,345.67'` or `'79,577'` become floats. Dates written as `03/15/1990`, `15 Mar 1990` or `1990/03/15` are parsed into one datetime type."),
        ("Category normalisation", "`F`, `f`, `Female`, `female ` → `F`. State names → two-letter codes. Case variants are lower-cased only in columns that actually contain them."),
        ("Null tokens", "Text such as `N/A`, `null`, `unknown` or an empty string means \"missing\" and is converted to a real missing value (NaN)."),
        ("Validation rules", "Domain rules: money ≥ 0, latitude within [-90, 90], gender coded F/M, ZIP code has 5 digits, death date not before birth date, birth date in ISO format. Violating values are set to missing (and counted)."),
        ("Ground-truth recovery", "Because the errors were injected into a copy of clean data, we can measure how much of the original the pipeline restores."),
    ],
    dataset="**Teaching dataset**: Synthea `patients.csv` with 14 documented, deterministic (seed 7) problems: coding variants (GENDER, MARITAL, RACE, STATE), mixed date formats, currency strings, thousands separators, lost ZIP leading zeros, invalid sentinels (-1 income, latitude ×10), text null tokens, extra deleted values, 5 exact duplicates and 3 near-duplicates (119 rows in total). The full corruption log is shown in the UI. Any other source can also be cleaned.",
    method=[
        "Trim and collapse whitespace in text columns.",
        "Convert null tokens to NaN.",
        "Map known category synonyms (GENDER, SEX, MARITAL, STATE); lower-case columns that contain case variants.",
        "Zero-pad ZIP codes to 5 digits.",
        "Convert columns to numbers when ≥ 90% of values parse after removing `$`, `,` and spaces.",
        "Parse date-like columns (name contains DATE/START/STOP/TIME) with `format='mixed'`.",
        "Apply validity rules and set violating values to NaN.",
        "Report: step log, dtype changes, value counts before/after, rule violations before/after, examples, and recovery against the original Synthea values.",
    ],
    files="`backend/app/services/ehr_data.py::make_messy_patients`, `backend/app/services/ehr_analysis.py` (`standardize`, `validation_issues`, `cleaning_report`, `_recovery`), `frontend/src/pages/EhrExplorer.tsx` (`CleanTab`)",
    endpoints="`GET /api/ehr/teaching-log`, `GET /api/ehr/clean?source=`",
    outputs="Cleaning log, bar chart of rule violations before vs after, category value counts before → after, type-conversion table, recovery table, example changes.",
    results="Rule violations fall from 141 to **0** after cleaning. GENDER, RACE, STATE, BIRTHDATE, ZIP and expense columns are recovered **100%**. Remaining mismatches are exactly the corruptions that cannot be undone: deleted MARITAL values (7), income values that were invalid or text-null (11), shifted latitudes that are now missing (2), and one near-duplicate where the upper-case record came first.",
    run="Open **EHR Data Explorer → Standardise & clean**. Read the corruption log, then press **Run cleaning pipeline**.",
    tests="`tests/test_ehr.py::test_standardize_fixes_types_and_categories`, `test_validation_issue_counts`, `test_teaching_dataset_cleaning_recovers_ground_truth`",
    limits="The rules are hand-written for the columns in this dataset. A real project needs a data dictionary and value sets (for example, LOINC/SNOMED). Setting invalid values to missing loses information, so in practice they should be queried with the data source.",
)

T["T3"] = dict(
    title="EHR Preprocessing: Redundancy and Missing Data",
    where="EHR Data Explorer → **Missing data & redundancy** tab (`/ehr?tab=missing`)",
    objective="Quantify missing data and redundancy, compare imputation methods objectively, and summarise data quality.",
    concepts=[
        ("Missingness patterns", "Missingness per column, complete rows, a missingness matrix, and the most common patterns (which columns are missing together)."),
        ("Co-missingness", "The correlation of 0/1 missing indicators. A value of 1 means the columns are always missing together, which happens when they come from the same lab panel. This suggests data that are *missing at random given another variable*, not completely at random."),
        ("Redundancy", "Exact duplicate rows; near-duplicates (equal after case/space normalisation); repeated keys (same `Id`); constant columns; numeric column pairs with |r| ≥ 0.98."),
        ("Imputation", "Mean and median (simple), **KNN** (average of the 5 most similar rows), and **iterative** imputation (MICE-style: each column is regressed on the others in turn)."),
        ("Masking evaluation", "Hide 15% of the *observed* values, impute them, and compare with the truth (MAE, RMSE, RMSE/SD). This is how you choose a method objectively."),
        ("Quality dimensions", "Completeness, uniqueness, validity and consistency, each scored 0–100. The overall score is their mean."),
    ],
    dataset="Synthea patient profile (natural missingness from lab panels) and the teaching dataset (injected duplicates, null tokens and invalid values). Uploaded CSVs also work.",
    method=[
        "Convert null tokens, then compute missingness per column and per row.",
        "Correlate the missing-indicator columns (up to 12 partially missing columns).",
        "Count missingness patterns.",
        "Detect duplicates (exact, near, by key) and redundant or constant columns.",
        "Imputation comparison on numeric columns with ≥ 50% observed values: mask 15%, re-impute with each of the 4 methods, report errors, and plot observed values against values imputed by the best method.",
        "Quality scorecard.",
    ],
    files="`backend/app/services/ehr_analysis.py` (`missingness`, `duplicates`, `_redundant_columns`, `imputation_comparison`, `quality_report`), `frontend/src/pages/EhrExplorer.tsx` (`MissingTab`)",
    endpoints="`GET /api/ehr/missingness`, `GET /api/ehr/duplicates`, `GET /api/ehr/imputation?target=`, `GET /api/ehr/quality`",
    outputs="Quality score cards, missingness bar chart, missingness matrix, co-missingness heat-map, pattern table, imputation RMSE chart and distribution comparison, duplicate report.",
    results="Patient profile: 14.6% of cells missing; only 13 of 111 patients are complete. Glucose, creatinine and HbA1c are perfectly co-missing (r = 1), as are the four lipid values. For **LDL**, iterative imputation has RMSE 13.7 mg/dL versus 33.6 (KNN) and 35.1 (mean), because LDL is strongly predictable from total cholesterol and HDL. Teaching dataset: 5 exact + 3 near duplicates; quality score 78.6 → 97.8 after cleaning and de-duplication.",
    run="Open **EHR Data Explorer → Missing data & redundancy**. Change the target column in *Imputation comparison*, or switch the dataset to the teaching variant to see duplicates.",
    tests="`tests/test_ehr.py::test_missingness_converts_null_tokens`, `test_duplicates_exact_key_and_near`, `test_imputation_comparison_ranks_methods`, `test_quality_report_improves_after_cleaning`",
    limits="Masking evaluation assumes values are missing at random, which is not true for lab tests ordered because of suspected disease. The quality score weights all four dimensions equally, which is a simplification. With 111 patients, imputation errors vary from run to run.",
)

T["T4"] = dict(
    title="Biomedical Image Analysis",
    where="Image Laboratory → **Enhance / Restore / Filter / Threshold / Segment / Compress** tabs (`/image-lab`)",
    objective="Apply classical image processing to chest X-rays and measure its effect.",
    concepts=[
        ("Enhancement", "Contrast stretching (2nd–98th percentile → 0–255), global histogram equalisation, **CLAHE** (contrast-limited adaptive equalisation on 8×8 tiles), gamma correction, unsharp masking (1.5·I − 0.5·blur(I))."),
        ("Restoration", "Add Gaussian, salt-and-pepper or speckle noise, then denoise with Gaussian, median, bilateral or non-local-means filters. Quality is measured against the clean image with **PSNR** (dB) and **SSIM** (structural similarity, 1 = identical)."),
        ("Filtering", "Low-pass (Gaussian, median), first derivative (Sobel gradient magnitude), second derivative (Laplacian), Canny edges (gradient + non-maximum suppression + hysteresis), high-pass (I − blur)."),
        ("Thresholding", "Global, **Otsu** (maximises between-class variance), triangle, adaptive mean and adaptive Gaussian (local thresholds)."),
        ("Segmentation", "Classical lung-field segmentation (no learned model): CLAHE → coarse Gaussian blur so the ribs fade → mediastinum located as the brightest central column band → inverse Otsu, cut along the midline → anatomical rules reject tiny regions, background air, black edge strips, air above the shoulders and bowel gas → **marker-controlled watershed** (eroded lung candidates as seeds; bright tissue, mediastinum and outside air as background) → opening, hole filling, clipping at the midline. Five **plausibility checks** (two fields, total area 15–55%, left/right ratio ≥ 0.4, height ≥ 35%, edge contact < 50% of rows) flag implausible masks. Until 28 Oct 2026 this tab also showed k-means intensity clustering, which groups grey levels and does not identify lungs; it was removed."),
        ("Compression", "PNG (lossless DEFLATE) vs JPEG (lossy DCT + quantisation) at qualities 95–10, compared by size, compression ratio, PSNR and SSIM."),
    ],
    dataset="8 real test-split X-rays from PneumoniaMNIST (`data/sample/xray`, 224×224), or any uploaded image (resized so the longest side is at most 512 px).",
    method=["Every operation runs on the backend with OpenCV/NumPy, and the images are returned as PNG data URLs with statistics (mean, SD, entropy) and histograms."],
    files="`backend/app/services/imaging.py` (`enhance`, `restore`, `filters`, `threshold`, `lung_fields`, `lung_mask`, `segment`, `compress`, `psnr`, `ssim`), `frontend/src/pages/ImageLab.tsx`, `scripts/evaluate_segmentation.py`",
    endpoints="`POST /api/imaging/{enhance|restore|filters|threshold|segment|compress}` (multipart `file` or `sample`, parameters as query strings)",
    outputs="Image panels with captions and metrics, histograms, a restoration metrics table, segmentation steps, a compression table and a PSNR-vs-ratio chart.",
    results="Sample `normal_test_0380`: Gaussian noise σ = 20 gives PSNR 22.1 dB / SSIM 0.34. A 5×5 Gaussian blur restores 31.9 dB / 0.83 (median 31.1 dB). JPEG quality 50 is 10.8× smaller than raw at 38.6 dB; quality 10 is 27.7× smaller at 31.3 dB. Segmentation (`scripts/evaluate_segmentation.py`, 624 test images, **no ground-truth masks, so plausibility only, not accuracy**): masks covering > 60% of the image fell from 71 with the previous heuristic (a corner flood-fill bug) to 2; all plausibility checks pass on 76.1% of images (normal 97.0%, pneumonia 63.6%) versus 54.5% before. On the 8 samples, all 4 normals and 2 of 4 pneumonia cases pass; the two failures are correctly flagged because consolidated lung is not dark.",
    run="Open **Image Laboratory**, choose a sample thumbnail or upload an image, open a tab, adjust the sliders and press **Run**.",
    tests="`tests/test_imaging.py` (PSNR/SSIM identity, median beats noise, Otsu threshold, two-lung segmentation on a phantom with Dice > 0.85 against its known lungs, dark corner and edge strip ignored, blank image flagged, compression monotonicity, API with samples and uploads, path-traversal protection)",
    limits="The segmentation is classical image processing, not a trained or validated lung segmentation model. It finds dark lung fields, so consolidated (white) lung in pneumonia is often missed, and it can fail on rotated or very low-contrast images. The plausibility checks were designed together with the method, so they flag obvious failures but cannot measure accuracy. A pretrained anatomical model (for example torchxrayvision's ChestX-Det PSPNet) was considered but not installed. MedMNIST images are already down-sampled, so fine detail is limited.",
)

T["T5"] = dict(
    title="SIFT, RANSAC, and CNN Feature Extraction",
    where="Image Laboratory → **SIFT · RANSAC · CNN** tab (`/image-lab?tab=features`)",
    objective="Detect and match local features, estimate geometry robustly, and extract deep CNN features.",
    concepts=[
        ("SIFT", "Keypoints are detected as extrema in a Difference-of-Gaussians scale space. Each has a position, scale and dominant orientation, plus a 128-D descriptor (4×4 cells × 8 gradient-orientation bins). This makes them invariant to scale and rotation."),
        ("Matching", "Brute-force L2 nearest neighbours with **Lowe's ratio test**: keep a match only if the best distance is < ratio × the second-best distance."),
        ("Homography and RANSAC", "A 3×3 homography maps one image plane to another. RANSAC repeatedly fits it to 4 random matches and keeps the model with the most inliers (reprojection error < threshold), so wrong matches (outliers) are ignored."),
        ("Known ground truth", "Image 2 is a synthetic warp of image 1 (rotation, scale, translation, perspective) with a known homography. The estimate is scored by the mean error of the four image corners in pixels."),
        ("CNN features", "An ImageNet-pretrained ResNet-18 used as a fixed feature extractor: feature maps from layer1–layer4 (64×56×56 … 512×7×7) and a 512-D global-average-pooled embedding."),
        ("Linear probe", "If a simple logistic regression on frozen features separates the classes, the features carry the information, even without fine-tuning."),
    ],
    dataset="PneumoniaMNIST samples or uploads. For the embedding map: 150 NORMAL + 150 PNEUMONIA test images.",
    method=[
        "SIFT detection and visualisation of the strongest 300 keypoints; histograms of scale and orientation.",
        "Synthetic warp → SIFT on both images → kNN matching with the ratio test → `findHomography` with RANSAC, compared with plain least squares → corner error → visualisation of inliers, outliers and the alignment.",
        "Forward hooks on ResNet-18 → the 12 most active channels of the chosen layer as images.",
        "Embeddings → standardise → PCA 2-D map; 5-fold cross-validated logistic-regression probe (cached in `models/imaging/feature_embedding.json`).",
    ],
    files="`backend/app/services/imaging.py` (`sift_keypoints`, `synthetic_transform`, `match_ransac`, `cnn_features`, `embed`, `feature_embedding`)",
    endpoints="`POST /api/imaging/sift`, `POST /api/imaging/match`, `POST /api/imaging/cnn-features?layer=`, `GET /api/imaging/feature-embedding`",
    outputs="Keypoint image, descriptor size, match visualisations, homography error table, alignment overlay, feature-map grid, embedding scatter, probe accuracy.",
    results="`normal_test_0380` with a 15° / 0.9× / perspective-0.05 warp: 94 ratio-test matches, 92 RANSAC inliers, corner error **0.07 px** with RANSAC versus **15.5 px** with least squares. That gap is why RANSAC is needed. Linear probe on frozen ImageNet features: **93.0% ± 2.4%** accuracy, ROC-AUC 0.975 (5-fold CV, 300 images).",
    run="Image Laboratory → **SIFT · RANSAC · CNN**. Press **Run** in each card. Try a larger rotation or a lower ratio to see the inliers change.",
    tests="`tests/test_imaging.py::test_ransac_recovers_synthetic_homography`, `test_sift_keypoints_reported`, `test_imaging_api_with_sample` (RANSAC error < least-squares error)",
    limits="X-rays have repetitive rib texture, so SIFT matching between *different* patients is unreliable. Synthetic warps are a controlled test, not real registration. ImageNet features are not specific to X-rays, and the probe is evaluated on only 300 images.",
)

T["T6"] = dict(
    title="Biomedical Image Visualization",
    where="Image Laboratory → **Visualise** tab (`/image-lab?tab=visualize`) and **Pneumonia Detection**",
    objective="Visualise images and model behaviour: comparisons, histograms, overlays, feature maps and Grad-CAM.",
    concepts=[
        ("Comparison slider", "The original and CLAHE-enhanced images overlaid, with a draggable divider."),
        ("Histogram and CDF", "Intensity distribution; the cumulative distribution is what histogram equalisation flattens."),
        ("Pseudo-colour", "Mapping grey levels to a colour map (bone, viridis, inferno, hot, jet). Perceptually uniform maps such as viridis avoid false edges."),
        ("Overlays", "Canny edges and the lung-field contour drawn on the X-ray; the |CLAHE − original| difference map."),
        ("Intensity profile", "Grey level along a chosen row: lungs appear as valleys, and the mediastinum and spine as a central peak."),
        ("Grad-CAM", "Class-activation heat-map from the trained pneumonia model (see `ml_documentation.md`)."),
    ],
    dataset="PneumoniaMNIST samples or uploads, plus the trained pneumonia checkpoint for Grad-CAM.",
    method=["The backend renders every panel with OpenCV. Grad-CAM calls the screening endpoint with `context=T6`, so the run counts as T6 (plain predictions on the Pneumonia Detection page do not count as T6 runs)."],
    files="`backend/app/services/imaging.py::visualize`, `backend/app/ml/pneumonia/gradcam.py`, `frontend/src/components/images.tsx` (`CompareSlider`, `ImageGrid`), `frontend/src/pages/ImageLab.tsx` (`VisualizeTab`)",
    endpoints="`POST /api/imaging/visualize?colormap=&row=`, `POST /api/screening/predict?context=T6`",
    outputs="Six visualisation panels, a comparison slider, a profile line chart, a histogram, and Grad-CAM raw and overlay images.",
    results="Grad-CAM maps are 7×7 (ResNet layer4) upsampled to the image. Measured with the classical lung-field segmentation (`scripts/evaluate_segmentation.py`, 156 test images whose masks pass the plausibility checks), the mean share of Grad-CAM mass inside the lung fields is 0.304, about the same as the lung fields' share of the image (0.316; median ratio 1.11). So the maps do **not** concentrate on the lung fields more than their area would predict. The model may be relying partly on regions outside the lungs, which is an important caveat when you present Grad-CAM. The feature maps from T5 are also visualisations for this tutorial.",
    run="Image Laboratory → **Visualise** → **Run**, then **Compute Grad-CAM**. Or upload on the **Pneumonia Detection** page and toggle *show raw map*.",
    tests="`tests/test_pneumonia_model.py::test_gradcam_heatmap_shape_and_range`, `test_prediction_response_schema_with_untrained_checkpoint`; `tests/test_imaging.py::test_imaging_api_with_sample` (visualize)",
    limits="Grad-CAM is coarse and explains the model, not the disease. Colour maps such as jet can create apparent boundaries that are not in the data.",
)

T["T7"] = dict(
    title="Patient Prediction for Improved Healthcare Staffing",
    where="Operations Analytics → **Volume & staffing** tab (`/operations?tab=forecast`)",
    objective="Forecast patient volume and translate it into a staffing estimate.",
    concepts=[
        ("Why a workload-based alternative", "Synthea has no staff rosters or staffing labels, so staff numbers cannot be predicted directly. Instead we predict **patient volume** (which has real labels: the observed encounter counts) and convert it to **workload** (encounters × median observed encounter duration) and full-time equivalents (FTE)."),
        ("Time-series features", "Lags 1–3 (monthly) or 1–4 (weekly), a rolling mean, seasonal sine/cosine of the month or week, and a trend index."),
        ("Time-based split", "The most recent 20% of periods are the test set. Random splits would leak the future into training."),
        ("Baselines", "Naive (next = last) and moving average. A model is useful only if it beats them."),
        ("Metrics", "MAE, RMSE, MAPE (with denominator max(y, 1))."),
        ("FTE", "Required FTE = workload hours / (hours per FTE per period × target utilisation, 0.85)."),
    ],
    dataset="Synthea encounters from 2016 onwards (3,935 encounters, 101 patients). Inpatient, SNF and hospice stays are excluded from clinic workload, because they consume beds rather than clinician hours.",
    method=[
        "Aggregate encounters into weekly or monthly counts (optionally for one encounter class); drop the incomplete last period.",
        "Build lag and season features; split by time.",
        "Fit Ridge regression and a random forest (300 trees); compare with the baselines.",
        "Re-fit the best model on all data; forecast recursively for the chosen horizon.",
        "Convert the forecast to workload hours and required FTE.",
    ],
    files="`backend/app/services/operations.py` (`encounters`, `volume_series`, `_features`, `forecast_volume`), `frontend/src/pages/Operations.tsx` (`ForecastTab`)",
    endpoints="`GET /api/operations/forecast?freq=&encounter_class=&horizon=&hours_per_fte=&utilization_target=`",
    outputs="Model comparison table, actual vs predicted chart, history + forecast chart, staffing table, assumptions and limitations.",
    results="Monthly, all classes (129 periods; 100 train / 26 test): MAE naive 6.62, moving average 6.46, Ridge 5.74, **random forest 5.71** (MAPE 19%). Average monthly volume is about 36 clinic encounters, with a median duration of 31 minutes, so only about 0.15 FTE is needed. A real clinic would have far higher volumes.",
    run="Operations Analytics → **Volume & staffing**. Switch between monthly and weekly, pick an encounter class, and move the horizon and FTE-hours sliders.",
    tests="`tests/test_analytics.py::test_forecast_beats_or_matches_naive`",
    limits="Only 111 synthetic patients, so the error estimates are noisy. Recursive forecasts accumulate error. There are no holiday or epidemic features. Durations are simulated. FTE is a planning estimate, not a roster.",
)

T["T8"] = dict(
    title="Healthcare Strategic Planning Part 1",
    where="Operations Analytics → **KPIs & comparisons** tab (`/operations?tab=kpis`)",
    objective="Compute healthcare key performance indicators (KPIs), compare groups statistically and follow trends.",
    concepts=[
        ("KPIs", "Encounters, unique patients, encounters per patient, total and average claim cost, payer coverage %, uninsured share, inpatient admissions, average length of stay (LOS), 30-day readmission rate, ED visits per 1,000 patients."),
        ("30-day readmission", "An inpatient admission followed by another admission of the same patient within 30 days of discharge."),
        ("Group comparison", "Costs are strongly right-skewed, so a rank-based **Kruskal–Wallis H** test (or Mann–Whitney U for 2 groups) is used instead of ANOVA."),
        ("Dashboards", "KPI cards, group bar charts, a monthly trend line and yearly unique patients."),
    ],
    dataset="Synthea encounters joined with patients (gender, age group at encounter), payers and organizations.",
    method=["Filter by start year → compute headline KPIs → group by encounter class, year, gender, age group, payer or organisation → run the rank test on the chosen metric (claim cost, base cost or duration) for groups with ≥ 5 encounters → monthly and yearly aggregates."],
    files="`backend/app/services/operations.py` (`kpis`, `_readmission_rate`), `frontend/src/pages/Operations.tsx` (`KpiTab`)",
    endpoints="`GET /api/operations/kpis?group_by=&metric=&start_year=`",
    outputs="KPI cards, encounters and average cost by group, test result with interpretation, group table, trends.",
    results="2016–2026: 3,935 encounters, 101 patients, average claim cost $3,301, payer coverage 71.1%, 4.9% uninsured encounters, 39 admissions, average LOS 5.0 days, 30-day readmission 0%. Cost differs strongly by encounter class (Kruskal–Wallis H = 307.8, p = 8.9 × 10⁻⁶²); SNF and inpatient stays are the most expensive.",
    run="Operations Analytics → **KPIs & comparisons**. Change *Compare groups by* and *Metric tested*.",
    tests="`tests/test_analytics.py::test_kpis_and_group_test`",
    limits="Synthetic costs and only 39 admissions, so the readmission rate is unstable. Statistical significance does not mean practical importance.",
)

T["T9"] = dict(
    title="Healthcare Strategic Planning Part 2",
    where="Operations Analytics → **Utilisation & scenarios** tab (`/operations?tab=planning`)",
    objective="Analyse demand patterns and resource utilisation, and compare what-if scenarios.",
    concepts=[
        ("Demand patterns", "Day × hour heat-map; demand by year and encounter class; year-over-year growth for the two most recent complete years."),
        ("Resource utilisation", "Provider workload (encounters per provider per active year, share handled by the busiest 10). Bed utilisation = occupied bed-days / (beds × days in month), with an 85% target line."),
        ("Scenario analysis", "Baseline = clinic demand over the last 12 months. Each scenario sets a demand change %, staff FTE and optionally minutes per encounter. Outputs: workload, capacity, utilisation, unmet hours, FTE needed at 85% utilisation, and projected cost."),
    ],
    dataset="Synthea encounters (2016+). The bed capacity is a user-editable assumption.",
    method=["Crosstab of start times → yearly pivot by class → provider-year averages → daily bed-occupancy series from long-stay intervals → deterministic scenario formulas."],
    files="`backend/app/services/operations.py` (`operations`, `scenarios`), `frontend/src/pages/Operations.tsx` (`PlanningTab`)",
    endpoints="`GET /api/operations/utilization?beds=`, `POST /api/operations/scenarios`",
    outputs="Heat-map, stacked demand bars, growth table, workload distribution, bed-utilisation chart, scenario chart and table with status badges.",
    results="215 providers average 2.4 encounters per provider-year, and the top 10 handle 25% of the workload. Ambulatory visits fell 23% between the last two complete years. Doubling demand exactly doubles workload and utilisation (checked in the tests).",
    run="Operations Analytics → **Utilisation & scenarios**. Move the beds slider, edit or add scenarios, and press **Compare scenarios**.",
    tests="`tests/test_analytics.py::test_utilization_and_scenarios`, `test_scenarios_require_input`; `tests/test_system.py::test_request_validation_uses_error_envelope`",
    limits="Synthea timestamps are simulated and stored in UTC, so the time-of-day pattern does not reflect real opening hours. The scenarios are linear and deterministic (no queueing or variability).",
)

T["T10"] = dict(
    title="Clinical Text Classification and Clustering",
    where="Clinical Text → **Classification** and **Clustering** tabs (`/clinical-text?tab=classify`)",
    objective="Preprocess clinical text, classify it with supervised models, and discover structure with clustering.",
    concepts=[
        ("Preprocessing", "spaCy tokenisation → lower-casing → removal of stop words, punctuation and numbers → lemmatisation (`treated` → `treat`). The UI shows every step."),
        ("TF-IDF", "Term frequency × inverse document frequency. Common words get small weights and distinctive words large ones. Settings: unigrams + bigrams, sublinear TF, min_df = 2, max_df = 0.9."),
        ("Classifiers", "Logistic regression, linear SVM and Complement Naive Bayes (Naive Bayes adapted to imbalanced text)."),
        ("Evaluation", "Accuracy, **macro-F1** (each class counts equally), weighted F1, per-class precision/recall/F1, confusion matrix."),
        ("Clustering", "TF-IDF → Truncated SVD (100 dimensions, the basis of latent semantic analysis) → L2 normalisation → k-means. Internal quality: silhouette. External agreement with the labels: adjusted Rand index (ARI), normalised mutual information (NMI), purity. Elbow plot for choosing k."),
    ],
    dataset="Medical Abstracts TC Corpus: 11,550 training and 2,888 test abstracts in 5 classes (neoplasms, digestive, nervous, cardiovascular, general pathological conditions).",
    method=["Fit each pipeline on the official training split; evaluate once on the test split; save the models (`models/text/classifiers.joblib`) and metrics. Clustering uses a random sample of 3,000 training abstracts."],
    files="`backend/app/services/text_analytics.py` (`preprocess_demo`, `train_classifiers`, `classify`, `cluster`), `frontend/src/pages/ClinicalText.tsx`",
    endpoints="`POST /api/text/preprocess`, `POST /api/text/train`, `GET /api/text/metrics`, `POST /api/text/classify`, `GET /api/text/cluster?k=`",
    outputs="Class distribution, model table, confusion matrix, per-class metrics, top terms per class, class probabilities and per-term contributions for your own text, cluster scatter, elbow plot, cluster descriptions, contingency heat-map.",
    results="Test set: Complement NB accuracy **0.586** / macro-F1 **0.578**; linear SVM 0.511 / 0.505; logistic regression 0.499 / 0.491 (chance ≈ 0.23). The *general pathological conditions* class overlaps with all the others, and the published baselines for this corpus are also around 0.6. Clustering with k = 5: silhouette 0.031, ARI 0.055, NMI 0.108, purity 0.41. The clusters capture themes (oncology, cardiology, case reports) rather than the official labels.",
    run="Clinical Text → **Classification** → **Train classifiers**, then paste text and press **Classify** / **Show preprocessing**. In **Clustering**, choose k and press **Cluster**.",
    tests="`tests/test_text.py::test_preprocess_pipeline`, `test_train_and_classify` (accuracy > 0.45, a breast-carcinoma text is classified as neoplasms), `test_classify_without_model_is_409`",
    limits="Hyperparameters are defaults, not tuned (tuning should use cross-validation on the training set, never the test set). Abstracts are not clinical notes. Labels are noisy, and the class 'general pathological conditions' is ambiguous.",
)

T["T11"] = dict(
    title="Clinical Information Extraction and Entity Recognition",
    where="Clinical Text → **Entity extraction** tab (`/clinical-text?tab=ner`)",
    objective="Extract structured clinical information (problems, medications with doses, vitals, labs, procedures) from free text, including negation.",
    concepts=[
        ("spaCy pipeline", "Tokenizer → tagger/parser → statistical NER (`en_core_web_sm`: DATE, QUANTITY, …) → our **EntityRuler**."),
        ("Statistical vs rule-based NER", "General models do not know clinical entities, so the EntityRuler adds lexicon phrases (≈80 problems, 44 medications, 23 labs, 20 procedures, 15 anatomy terms) and token patterns for DOSAGE (`500 mg bid`), VITAL (`BP 138/84`, `HR 112`, `SpO2 91%`), LAB values (`WBC 15.2`) and AGE."),
        ("Negation (NegEx-style)", "A problem is negated if a trigger (`no`, `denies`, `without`, `negative for`, `ruled out`, …) appears within the 6 preceding tokens of the same sentence, unless a terminator (`but`, `however`, …) comes in between. The window lets lists inherit the trigger (\"denies chest pain or hemoptysis\")."),
        ("Context", "*historical* (`history of`, `s/p`) and *family* (`family history`, `mother`) flags."),
        ("Structured output", "Problems present/negated, family history, medication–dosage pairs (nearest following dosage in the same sentence), vitals, labs, procedures, as JSON."),
    ],
    dataset="Three **synthetic** sample notes written for the project (ED note, outpatient follow-up, discharge summary), any pasted text, and 200 corpus abstracts for statistics at scale.",
    method=["Load the best available model (`en_ner_bc5cdr_md` if installed, else `en_core_web_sm`, else a blank pipeline + sentencizer) → add the ruler → run → `filter_spans` → context flags → build the structured summary."],
    files="`backend/app/services/clinical_ner.py`, `frontend/src/pages/ClinicalText.tsx` (`NerTab`, `Highlighted`)",
    endpoints="`GET /api/text/sample-notes`, `POST /api/text/entities`, `GET /api/text/entity-stats?per_class=`",
    outputs="Colour-highlighted text (negated entities struck through), structured summary + JSON, entity table, top entities per disease class in the corpus.",
    results="ED sample note: 29 entities. Chest pain, hemoptysis and pleural effusion are negated correctly; hypertension and diabetes are flagged historical; all 4 medications are linked to their doses (for example ceftriaxone → 1 g daily); 5 vitals are extracted. Follow-up note: \"no fever\" after \"but\" is negated, while the earlier \"wheezing\" is not; COPD is flagged as family history.",
    run="Clinical Text → **Entity extraction**. Click a sample note → **Extract entities**. Press **Run on corpus** for statistics.",
    tests="`tests/test_text.py::test_ner_labels_and_negation`, `test_ner_family_and_structured_output`, `test_ner_rejects_empty_text`",
    limits="The lexicon is small and hand-made, so recall is limited to listed terms. Negation rules miss complex cases such as double negation or uncertainty (\"cannot rule out\"). `en_core_web_sm` is not a clinical model. To add statistical clinical NER, install scispaCy's `en_ner_bc5cdr_md` 0.5.x; it is detected automatically.",
)

T["T12"] = dict(
    title="Clinical Text Ranking and Word Disambiguation",
    where="Clinical Text → **Ranking & disambiguation** tab (`/clinical-text?tab=ranking`)",
    objective="Rank documents by relevance to a query, and disambiguate clinical abbreviations from their context. These are two separate demonstrations.",
    concepts=[
        ("Vector space model", "Documents and queries become TF-IDF vectors. Relevance = **cosine similarity**."),
        ("BM25", "A probabilistic ranking function with term-frequency saturation (k1 = 1.5) and document-length normalisation (b = 0.75)."),
        ("Evaluation", "Precision@k and mean average precision (MAP@k). Here a test abstract's title is the query, and a retrieved training abstract counts as relevant if it has the same disease label (a proxy for topical relevance)."),
        ("Word-sense disambiguation (WSD)", "Abbreviations such as MS (multiple sclerosis / mitral stenosis / morphine sulfate), RA, PE, CVA, PT and CP are ambiguous."),
        ("Simplified Lesk", "Choose the sense whose signature (bag of words) overlaps most with the sentence context, after stop-word removal and light stemming. There are three signature variants: hand-written gloss (`lesk`), gloss + corpus (`corpus_lesk`), and corpus only (`corpus_only`: words that appear near the full form, e.g. \"mitral stenosis\", in the medical abstracts, weighted by specificity)."),
    ],
    dataset="Ranking: the 11,550 training abstracts as the collection, test titles as queries. WSD: 43 **hand-written synthetic** labelled sentences (`wsd_data.py`) for 6 abbreviations and 15 senses.",
    method=["Build TF-IDF and BM25 indexes over the same vocabulary → score → top-k with matched terms → evaluate 200 queries per method. WSD: tokenise the context → score every sense → pick the maximum (ties go to the first-listed sense) → evaluate all variants and a first-sense baseline."],
    files="`backend/app/services/text_analytics.py` (`_Index`, `rank`, `evaluate_ranking`), `backend/app/services/wsd.py`, `backend/app/services/wsd_data.py`",
    endpoints="`POST /api/text/rank`, `GET /api/text/rank-evaluation`, `POST /api/text/wsd`, `GET /api/text/wsd-evaluation`",
    outputs="Ranked list with scores, labels and matched terms; P@10 / MAP@10 table; sense score table with overlapping words; evaluation table.",
    results="Retrieval (200 queries, k = 10): BM25 P@10 **0.457**, MAP 0.535; TF-IDF P@10 0.438, MAP 0.532; random baseline 0.232. WSD accuracy: `lesk` 1.00 and `corpus_lesk` 1.00 (optimistic, because the same person wrote the glosses and the test sentences), **`corpus_only` 0.58**, first-sense baseline 0.42.",
    run="Enter a query → **Search** (switch between TF-IDF and BM25) → **Evaluate both methods**. Enter a sentence with an abbreviation → **Disambiguate** → **Evaluate on labelled examples**.",
    tests="`tests/test_text.py::test_ranking_returns_relevant_terms`, `test_wsd_uses_context`, `test_wsd_evaluation_beats_baseline`",
    limits="Label-based relevance is only a proxy for real relevance. The WSD examples are few and synthetic. The corpus contains research abstracts, so senses such as \"physical examination\" or \"room air\" are rare there.",
)

T["T13"] = dict(
    title="Dimensionality Reduction for EDA",
    where="Advanced Analytics → **PCA & EDA** tab (`/advanced?tab=pca`)",
    objective="Reduce many correlated variables to a few components and use them for exploratory analysis.",
    concepts=[
        ("Feature scaling", "PCA maximises variance, so unscaled variables with large units (for example kcal) dominate. z-scoring gives every feature variance 1. Min-max scaling and centring only are offered for comparison."),
        ("PCA", "Eigen-decomposition of the covariance matrix. Components are orthogonal directions of decreasing variance."),
        ("Explained variance and scree plot", "The share of total variance per component, and the cumulative curve (with an 80% reference line)."),
        ("Loadings", "Each feature's weight in a component; they show what a component *means*."),
        ("Scores", "The data projected onto PC1–PC3, shown as a 2-D scatter and a rotatable 3-D scatter, coloured by a categorical variable."),
    ],
    dataset="NHANES adults (15 numeric variables, n = 5,120), the Synthea patient profile (26 numeric variables), or an uploaded CSV.",
    method=["Select numeric columns with > 2 distinct values and ≥ 30% observed → median impute or drop rows → scale → full PCA for explained variance → scores for up to 1,500 sampled points → correlation matrix."],
    files="`backend/app/services/dimensionality.py`, `frontend/src/pages/AdvancedAnalytics.tsx` (`PcaTab`), `frontend/src/components/charts/Scatter3D.tsx`",
    endpoints="`GET /api/analytics/pca/sources`, `GET /api/analytics/pca?source=&scaling=&missing=&color_by=`",
    outputs="Scree + cumulative chart, 2-D and 3-D score plots, top loadings per component, loading table, correlation heat-map.",
    results="NHANES, standardised: **8 of 15** components explain 80%. PC1 = diet quantity (energy +0.49, fat +0.45, protein +0.42), PC2 = body size (waist, BMI, weight ≈ +0.54), PC3 = activity vs age/sedentary time. Synthea profile: 12 of 26 components for 80%; PC1 = age and morbidity (conditions, age, QALY, encounters). Without scaling, the first component is almost entirely the largest-unit variable.",
    run="Advanced Analytics → **PCA & EDA**. Change the scaling, colour by `bmi_category` or `sex`, and rotate the 3-D plot.",
    tests="`tests/test_analytics.py::test_pca_on_correlated_data` (PC1 > 60% on data with one latent factor), `test_pca_validation`, `test_pca_api`",
    limits="PCA is linear and sensitive to outliers. Median imputation shrinks variance. Components are not causal factors.",
)

T["T14"] = dict(
    title="Behavioral Factors Influencing Adult Weight Part 1",
    where="Advanced Analytics → **Adult weight: exploration** tab (`/advanced?tab=weight-eda`)",
    objective="Describe adult BMI and lifestyle factors and how they relate.",
    concepts=[
        ("Recoding survey data", "NHANES uses codes such as 7/9/77/99/777/7777/9999 for refused or don't know, and these become missing. Activity minutes = days × minutes for work, transport and recreation; vigorous minutes count double (moderate-equivalent minutes per week). Alcohol frequency codes are converted to drinking days per year, then multiplied by drinks per day. Smoking: never / former / current. Diet values come from day-1 recall and are kept only if the recall was reliable."),
        ("BMI categories (WHO)", "< 18.5 underweight, 18.5–25 normal, 25–30 overweight, ≥ 30 obese."),
        ("Spearman correlation", "Rank correlation, robust to skewed variables such as alcohol and activity."),
        ("Survey weights", "NHANES over-samples some groups. The MEC exam weight `WTMEC2YR` gives a population estimate (a point estimate only here)."),
        ("Group comparisons", "Mean BMI and obesity % by activity level, sleep band, smoking, education, sex and race/ethnicity, each with a Kruskal–Wallis test."),
    ],
    dataset="NHANES 2017–2018: DEMO_J, BMX_J, PAQ_J, SLQ_J, SMQ_J, ALQ_J and DR1TOT_J merged on SEQN; adults ≥ 20 years, not pregnant, BMI measured. **n = 5,120**.",
    method=["Merge → filter → recode → descriptive table → histogram (values above the 99th percentile grouped) → BMI categories → weighted and unweighted obesity prevalence → Spearman matrix → group tables with tests."],
    files="`backend/app/services/lifestyle.py` (`adults`, `explore`), `frontend/src/pages/AdvancedAnalytics.tsx` (`WeightEdaTab`)",
    endpoints="`GET /api/analytics/weight/explore?variable=`",
    outputs="Stat cards, distribution and BMI-category charts, mean BMI by group with a test, correlation table and heat-map, descriptive statistics, summary by sex.",
    results="Obesity prevalence is 41.8% unweighted and **42.5% survey-weighted**, consistent with the CDC's published estimate of about 42% for 2017–2018. That is a useful validation of the recoding. The strongest BMI correlate is sedentary minutes per day (ρ = +0.10); alcohol (−0.09), fibre (−0.07) and sleep (−0.06) are weaker. Mean BMI is 30.3 for inactive adults and 29.6 for those meeting the activity guideline (Kruskal–Wallis p = 0.008).",
    run="Advanced Analytics → **Adult weight: exploration**. Change the distribution variable and the grouping.",
    tests="`tests/test_analytics.py::test_weight_exploration` (n > 4,000, prevalence between 30% and 55%)",
    limits="Cross-sectional data, so no causal claims. Activity, sleep and alcohol are self-reported. Diet comes from a single 24-hour recall. Weighted estimates have no design-based standard errors.",
)

T["T15"] = dict(
    title="Behavioral Factors Influencing Adult Weight Part 2",
    where="Advanced Analytics → **Adult weight: models** tab (`/advanced?tab=weight-model`)",
    objective="Model BMI (regression) and obesity (classification) from behavioural and demographic factors, evaluate the models, interpret them, and state their limitations.",
    concepts=[
        ("Target leakage", "Weight, height and waist circumference are **excluded**, because BMI = weight / height² and waist is almost a proxy for BMI. Including them would give misleadingly high scores."),
        ("Pipeline", "ColumnTransformer: median imputation + standardisation for numeric features, most-frequent imputation + one-hot encoding (first level dropped) for categorical features."),
        ("Models", "Linear regression and a random forest regressor (BMI); logistic regression and a random forest classifier (obese = BMI ≥ 30)."),
        ("Evaluation", "80/20 stratified split. Regression: R², MAE, RMSE versus a predict-the-mean baseline. Classification: accuracy, precision, recall, F1, ROC-AUC versus a majority-class baseline. Plus 5-fold cross-validation on the training part."),
        ("Interpretation", "Linear coefficients (BMI change per 1 SD), logistic odds ratios, and permutation importance (the drop in test R² when a feature is shuffled)."),
    ],
    dataset="NHANES adults from T14 (n = 5,120): 11 numeric and 4 categorical predictors.",
    method=["Split → fit the 4 pipelines → test metrics + cross-validation → coefficients and odds ratios → permutation importance (5 repeats) → save to `models/lifestyle/`."],
    files="`backend/app/services/lifestyle.py` (`train_models`, `model_results`), `frontend/src/pages/AdvancedAnalytics.tsx` (`WeightModelTab`)",
    endpoints="`POST /api/analytics/weight/train`, `GET /api/analytics/weight/results`",
    outputs="Regression and classification tables, actual vs predicted scatter, ROC curves, permutation-importance chart, coefficient and odds-ratio tables, limitations.",
    results="BMI regression on the test set: random forest **R² 0.115** (CV 0.100 ± 0.013), MAE 5.2; linear regression R² 0.091; mean baseline R² ≈ 0. Obesity: random forest ROC-AUC **0.699**, accuracy 0.650 (majority baseline 0.582); logistic regression AUC 0.670. Most important features: race/ethnicity, age, sedentary time, alcohol, sex. The non-Hispanic Asian group has much lower odds of obesity (OR 0.20). **Conclusion:** the measured lifestyle factors explain only about 10% of BMI variation. They are associated with weight, but are weak individual predictors.",
    run="Advanced Analytics → **Adult weight: models** → **Train models** (about 1 minute). The results persist across restarts.",
    tests="`tests/test_analytics.py::test_weight_models_exclude_leaky_features` (no leaky features; models beat the baselines)",
    limits="Associations, not causes. Survey weights are not used in fitting, so results describe the sample. Self-report bias. Single-day diet. Race/ethnicity reflects social and structural factors, not biology.",
)


def status_line(tid):
    v = VERIFIED.get(tid)
    if v is None:
        return "**Status:** not verified (run `python scripts/verify_tutorials.py`)"
    n_ok = sum(c["passed"] for c in v["checks"])
    word = "verified" if v["passed"] else "VERIFICATION FAILED"
    return (f"**Status:** {word} - {n_ok}/{len(v['checks'])} workflow checks passed against the live API "
            f"({v['verified_at'][:10]}, `scripts/verify_tutorials.py`)")


def verification_section(tid):
    v = VERIFIED.get(tid)
    if v is None:
        return ["## Verification", "", "Not verified yet.", ""]
    lines = ["## Verification", "", "Checks run by `scripts/verify_tutorials.py` on the real dataset through the API:", ""]
    lines += [f"- {'✓' if c['passed'] else '✗'} {c['check']}" + (f" ({c['detail']})" if c["detail"] else "")
              for c in v["checks"]]
    if v.get("error"):
        lines.append(f"- error: {v['error']}")
    return lines + [""]


def tutorial_doc(tid, d):
    lines = [f"# {tid} — {d['title']}", "", f"**Where in the app:** {d['where']}  ", status_line(tid), "",
             "## Objective", "", d["objective"], "", "## Key concepts", ""]
    for name, text in d["concepts"]:
        lines.append(f"- **{name}.** {text}")
    lines += ["", "## Dataset", "", d["dataset"], "", "## Methodology", ""]
    lines += [f"{i}. {m}" for i, m in enumerate(d["method"], 1)] if len(d["method"]) > 1 else [d["method"][0]]
    lines += ["", "## Implementation", "", f"- Code: {d['files']}", f"- API: {d['endpoints']}", "",
              "## Outputs", "", d["outputs"], "", "## Results (actual run, 26 Oct 2026)", "", d["results"], "",
              "## How to run it", "", d["run"], "",
              "Prerequisites: the backend and frontend are running (see README) and the dataset has been downloaded.", "",
              "## Tests", "", d["tests"], ""] + verification_section(tid) + ["## Limitations", "", d["limits"], ""]
    return "\n".join(lines)


(DOCS / "tutorials").mkdir(parents=True, exist_ok=True)
for tid, d in T.items():
    (DOCS / "tutorials" / f"{tid}.md").write_text(tutorial_doc(tid, d), encoding="utf-8")

rows = ["| Tutorial | Title | App module / tab | Main backend code | Endpoints | Data | Verified |", "|---|---|---|---|---|---|---|"]
data_of = {"T1": "Synthea", "T2": "Synthea (teaching variant)", "T3": "Synthea", "T4": "PneumoniaMNIST", "T5": "PneumoniaMNIST",
           "T6": "PneumoniaMNIST + trained model", "T7": "Synthea encounters", "T8": "Synthea encounters", "T9": "Synthea encounters",
           "T10": "Medical Abstracts", "T11": "Synthetic notes + Medical Abstracts", "T12": "Medical Abstracts + synthetic WSD set",
           "T13": "NHANES / Synthea profile", "T14": "NHANES 2017-2018", "T15": "NHANES 2017-2018"}
import re  # noqa: E402

for tid, d in T.items():
    code = re.search(r"`([^`]+)`", d["files"]).group(1).split("::")[0]
    endpoint = re.search(r"`([^`]+)`", d["endpoints"]).group(1).replace("|", r"\|")
    more = " …" if d["endpoints"].count("`") > 2 else ""
    rows.append(f"| [{tid}](tutorials/{tid}.md) | {d['title']} | {d['where'].split(' (`')[0]} | `{code}` | `{endpoint}`{more} | "
                f"{data_of[tid]} | {'yes' if VERIFIED.get(tid, {}).get('passed') else 'no'} |")
verified_count = sum(1 for t in T if VERIFIED.get(t, {}).get("passed"))
mapping = f"""# Tutorial Mapping (T1–T15)

All 15 tutorials are implemented as working modules in one application. {verified_count} of 15 passed their
workflow verification (`scripts/verify_tutorials.py`, results in `reports/tutorial_verification.json`). They share navigation, the FastAPI
backend, the SQLite analysis history, the UI components and the reports service. The data domains stay separate.

{chr(10).join(rows)}

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

- `AsyncView` loading/error/empty states and the `{{error: {{code, message}}}}` envelope (`dataset_unavailable`,
  `model_unavailable`, `validation_failed`, `invalid_request`, `not_found`).
- `history.record()` writes to `analysis_runs`, which drives the dashboard and the *Analysis history* report.
- `reports.py` exports EHR quality (T2–T3), operations KPIs (T8), text classification (T10) and weight models (T15)
  as CSV, JSON or PDF.
"""
(DOCS / "tutorial_mapping.md").write_text(mapping, encoding="utf-8")
print("written", len(T))
