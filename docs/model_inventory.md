# Model and Algorithm Inventory

Audit of 28 Oct 2026. Every entry was checked against the code (`backend/app`), the files on disk (`models/`,
the torchvision weight cache, the backend virtual environment) and the API model registry (`GET /api/models`).

Types used below:
- **Trained here**: fitted by this project's scripts or UI buttons; the fitted model is saved under `models/`.
- **Pretrained**: weights produced by someone else and downloaded; used as-is or as a starting point.
- **Fitted on demand**: a statistical/ML model refitted on every request; nothing is saved.
- **Classical**: a fixed algorithm or rule set with no learning.

## 1. Saved models (files on disk)

| Model | Type | File(s) | Trained on | Used by | Status |
|---|---|---|---|---|---|
| Pneumonia classifier: ResNet-18, all layers fine-tuned, head `Dropout(0.3) → Linear(512, 2)` | Trained here (transfer learning from ImageNet) | `models/pneumonia/pneumonia_resnet18.pt` (44.8 MB) + `evaluation.json`, `audit.json`, `training_history.json`, `training_log.txt`, 3 PNG plots | PneumoniaMNIST train split (4,708 images), 6 epochs, 26 Oct 2026 | Pneumonia Detection, Model Performance, T6 Grad-CAM, Reports | Trained; test AUC 0.994; audit 9/9 |
| ImageNet ResNet-18 (`IMAGENET1K_V1`) | Pretrained (torchvision) | `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` (46.8 MB), outside the project | ImageNet | T5 feature maps, 512-D embedding and linear probe; starting weights for the pneumonia classifier | In use |
| Clinical text classifiers: TF-IDF + logistic regression / linear SVM / Complement Naive Bayes | Trained here | `models/text/classifiers.joblib` (17.2 MB), `classifier_metrics.json` | Medical Abstracts, 11,550 training abstracts | T10 classification, Reports | Trained; best macro-F1 0.578 |
| Adult-weight models: linear regression + random forest (BMI); logistic regression + random forest (obesity) | Trained here | `models/lifestyle/weight_models.joblib` (18.2 MB), `weight_models.json` | NHANES 2017–2018 adults, 4,096 training rows | T15, Reports | Trained; RF R² 0.115, AUC 0.699 |
| spaCy `en_core_web_sm` 3.7.1 (tok2vec, tagger, parser, NER) | Pretrained (general English) | `backend/.venv/Lib/site-packages/en_core_web_sm` | OntoNotes-style web text (spaCy) | T10 preprocessing (tokenise, lemmatise), T11 entity extraction | In use; **not listed** in `GET /api/models` |
| T5 embedding results | Cache of results, not a model | `models/imaging/feature_embedding.json` | 300 PneumoniaMNIST test images | T5 embedding map | Cached results only |

## 2. Models fitted on demand (nothing saved)

| Model | Code | Tutorial |
|---|---|---|
| Mean/median imputation, `KNNImputer(k=5)`, `IterativeImputer` | `services/ehr_analysis.py` | T3 |
| `Ridge(alpha=1)` and `RandomForestRegressor(300 trees)` with naive and moving-average baselines | `services/operations.py` | T7 |
| `TruncatedSVD(100)` + `KMeans` (text clustering) | `services/text_analytics.py` | T10 |
| TF-IDF and BM25 indexes (kept in memory) | `services/text_analytics.py` | T12 |
| `PCA` with standard / min-max scaling | `services/dimensionality.py` | T13 |
| `PCA` (2-D map) + logistic-regression linear probe on ResNet-18 embeddings | `services/imaging.py` | T5 |
| Permutation importance | `services/lifestyle.py` | T15 |

## 3. Classical algorithms and rules (no learning)

| Algorithm | Code | Tutorial / feature |
|---|---|---|
| Contrast stretching, histogram equalisation, CLAHE, gamma, unsharp masking; noise + Gaussian/median/bilateral/non-local-means denoising; Sobel, Laplacian, Canny; global/Otsu/triangle/adaptive thresholds; PNG/JPEG compression; PSNR, SSIM | `services/imaging.py` | T4 |
| **Lung-field segmentation**: coarse-scale candidates, anatomical rules, marker-controlled watershed, 5 plausibility checks | `services/imaging.py` (`lung_fields`, `segment`) | T4, T6 contour overlay, Grad-CAM localisation check |
| SIFT keypoints, Lowe ratio-test matching, RANSAC homography vs least squares | `services/imaging.py` | T5 |
| Grad-CAM (applied to the trained classifier), pseudo-colour, overlays, intensity profiles | `ml/pneumonia/gradcam.py`, `services/imaging.py` | Pneumonia Detection, T6 |
| Average-hash + pixel-confirmed duplicate detection | `ml/pneumonia/data.py` | Pneumonia data pipeline |
| Clinical EntityRuler (lexicon + token patterns), NegEx-style negation, historical/family context | `services/clinical_ner.py` | T11 |
| Simplified Lesk word-sense disambiguation (3 signature variants) | `services/wsd.py` | T12 |
| Kruskal–Wallis, Mann–Whitney U, Spearman correlation, survey-weighted prevalence, deterministic scenario formulas | `services/operations.py`, `services/lifestyle.py` | T8, T9, T14 |
| Profiling, standardisation and rule-based cleaning | `services/ehr_data.py`, `services/ehr_analysis.py` | T1, T2 |

## 4. Missing, optional or unused

| Item | Finding | Effect |
|---|---|---|
| scispaCy `en_ner_bc5cdr_md` | **Missing** (optional). Installing it was not permitted in the build session. The code uses it automatically when installed | T11 relies on `en_core_web_sm` + rules |
| Pretrained lung segmentation (torchxrayvision ChestX-Det PSPNet) | **Not installed.** Considered on 28 Oct 2026 as a replacement for the classical method; the package install and weight download were not permitted in this session | T4 uses the classical method, labelled as such |
| ImageNet ResNet-50 weights `resnet50-11ad3fa6.pth` (102.5 MB) | **Unused.** In the torchvision cache (outside the project). `train_pneumonia.py --arch resnet50` is supported, but no ResNet-50 checkpoint was trained | None; the cache file can be deleted |
| Kaggle external evaluation (`evaluation_kaggle.json`) | **Not produced**, because the Kaggle dataset is not installed | No full-resolution check of the classifier |
| k-means intensity clustering in the Segment tab | **Removed** on 28 Oct 2026. It grouped grey levels and did not identify lungs | Replaced by the lung-field segmentation above |
| Interrupted-training files (`*.partial.pt`) | None present | — |

## 5. Segmentation: what changed on 28 Oct 2026

The previous Segment tab showed a lung-mask heuristic (inverse Otsu + largest dark component per image half), a
k-means intensity-cluster panel and a watershed panel. On the 624 PneumoniaMNIST test images, 71 masks covered more
than 60% of the image, mostly because the hole-filling flood fill started from the image corner and filled
everything when a dark region touched that corner.

The new method (`imaging.lung_fields`) and its evaluation (`scripts/evaluate_segmentation.py` →
`reports/segmentation_evaluation.json`):

| 624 test images | Previous heuristic | Current method |
|---|---|---|
| Masks covering > 60% of the image | 71 | 2 |
| All 5 plausibility checks pass (all / normal / pneumonia) | 54.5% / 76.5% / 41.3% | 76.1% / 97.0% / 63.6% |
| Median mask area | 24.8% | 30.0% |
| Time per image | — | 15 ms |

**There are no ground-truth lung masks for PneumoniaMNIST, so these figures are plausibility and failure signs, not
accuracy (no Dice or IoU).** The plausibility checks were written together with the new method, which favours it in
that comparison. The method finds dark lung fields, so consolidated (white) lung in pneumonia is often missed; those
cases are typically flagged by the checks. On a synthetic phantom with known lungs, the unit test requires Dice > 0.85.
