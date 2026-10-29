# Pneumonia Detection: ML Documentation

All numbers below were produced by the scripts in this repository on 26 Oct 2026 (CPU: AMD Ryzen 7 5700U).
The raw outputs are in `models/pneumonia/` (`dataset_inspection.json`, `training_history.json`, `evaluation.json`,
`training_log.txt`, PNG plots).

## 1. Task

Binary image classification: **NORMAL** vs **PNEUMONIA** on paediatric chest X-rays.

## 2. Data

### Source used: PneumoniaMNIST (MedMNIST+, 224×224)

- Derived from Kermany et al. (2018), *Cell* (the same images as the Kaggle dataset), resized to 224×224 and
  centre-cropped by the MedMNIST authors. Licence CC BY 4.0.
- Downloaded with `scripts/download_datasets.py --only pneumoniamnist`. The file's SHA-256 is recorded in
  `data/raw/pneumoniamnist/_manifest.json`.
- Official splits: train and validation come from the Kaggle training folder (9:1); test is the Kaggle test folder.

| Split | Total | NORMAL | PNEUMONIA | Pneumonia share |
|---|---|---|---|---|
| train | 4,708 | 1,214 | 3,494 | 74.2% |
| val | 524 | 135 | 389 | 74.2% |
| test | 624 | 234 | 390 | 62.5% |

### Optional source: Kaggle "Chest X-Ray Images (Pneumonia)"

This source needs a Kaggle login, and no credentials are available on this machine, so it has **not** been
downloaded. It is **optional**: the shipped model was trained on PneumoniaMNIST, which is built from the same images.
Install it with `python scripts/download_kaggle.py` (using your `~/.kaggle/kaggle.json`) or
`python scripts/download_kaggle.py --zip <archive>` (after downloading it in the browser). The installer
de-duplicates the archive's nested `chest_xray/chest_xray` copy and skips `__MACOSX` files.

Installing it changes **nothing** automatically: `source="auto"` always means PneumoniaMNIST, so nothing is silently
replaced. Kaggle is used only when requested with `--source kaggle`:

- `scripts/evaluate_pneumonia.py --source kaggle`: **external evaluation without retraining**, on the full-resolution
  Kaggle test images. It writes `evaluation_kaggle.json` and `*_kaggle.png` and never overwrites the official
  `evaluation.json`. Note that these are the same 624 patients as the MedMNIST test split at a different
  resolution, not an independent population.
- `scripts/train_pneumonia.py --source kaggle`: optional retraining. The official Kaggle `val` folder has only 16 images, so the loader merges train
and val and **re-splits them by patient ID** (parsed from names like `person123_bacteria_456.jpeg`) using
`GroupShuffleSplit`. The test folder is kept unchanged. A unit test checks that train and validation share no patients.

## 3. Inspection and leakage checks (`scripts/inspect_pneumonia_dataset.py`)

1. Class counts, image size and intensity statistics per split.
2. Duplicate detection across splits:
   - stage 1: 64-bit **average hash** (8×8 thumbnail thresholded at its mean);
   - stage 2: each hash collision is confirmed by comparing 32×32 thumbnails (mean absolute difference ≤ 4 grey levels).

| Pair | Hash collisions | Confirmed duplicates |
|---|---|---|
| train–val | 16 | **8** (pixel difference 0.0) |
| train–test | 14 | 0 |
| val–test | 2 | 0 |

**Interpretation.** The test set is free of confirmed duplicates, so the test metrics are not inflated by leakage.
The 8 validation images that duplicate training images are **removed from validation during training**
(`drop_val_duplicates=True`), because they would otherwise bias model selection and threshold choice. Hash
collisions without pixel confirmation are just similar-looking X-rays. MedMNIST does not include patient IDs, so
patient-level leakage cannot be checked for this source. This is a known limitation.

## 4. Preprocessing and augmentation (`app/ml/pneumonia/data.py`)

| Step | Train | Val / test / inference |
|---|---|---|
| Grayscale → 3 channels (matches ImageNet weights) | ✓ | ✓ |
| Resize to 224×224 | ✓ | ✓ |
| Random affine: ±10° rotation, ±5% translation, 0.9–1.1 scale | ✓ | – |
| Brightness/contrast jitter ±15% | ✓ | – |
| ToTensor + ImageNet mean/std normalisation | ✓ | ✓ |

Horizontal flips are deliberately **not** used: they would move the heart to the wrong side, which never happens
in real frontal X-rays. Uploaded images go through exactly the same validation transform as the test set.

## 5. Model (`app/ml/pneumonia/model.py`)

- **ResNet-18** with ImageNet weights (torchvision `IMAGENET1K_V1`). The final layer is replaced by
  `Dropout(0.3) → Linear(512, 2)`. ResNet-50 is also supported (`--arch resnet50`).
- All layers are fine-tuned.
- Why ResNet-18: it is small enough (11 M parameters) to train on a laptop CPU in under an hour, and residual
  connections make fine-tuning stable.

## 6. Training (`app/ml/pneumonia/train.py`)

| Setting | Value |
|---|---|
| Loss | Cross-entropy with **class weights** (inverse frequency: NORMAL 1.94, PNEUMONIA 0.67) |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4 |
| Schedule | Cosine annealing over the epochs |
| Batch size | 32 |
| Epochs | 6 (early stopping with patience 3 on validation ROC-AUC) |
| Model selection | Best validation ROC-AUC. Best-so-far weights are saved to `*.partial.pt`; the final checkpoint is written only when training finishes, so the API never loads a half-trained file |
| Threshold | **Youden J** (max sensitivity + specificity − 1) on the validation split |
| Seed | 42 |
| Time | 2,347 s (~39 min) on CPU |

| Epoch | Train loss | Val loss | Val acc | Val ROC-AUC |
|---|---|---|---|---|
| 1 | 0.147 | 0.075 | 0.977 | 0.9960 |
| 2 | 0.067 | 0.104 | 0.963 | 0.9978 |
| 3 | 0.047 | 0.108 | 0.969 | 0.9979 |
| 4 | 0.037 | 0.079 | 0.979 | 0.9974 |
| 5 | 0.035 | 0.116 | 0.961 | 0.9982 |
| 6 | 0.029 | 0.108 | 0.967 | **0.9983** (best) |

The training loss keeps falling while the validation loss stays roughly flat. Validation AUC is near its ceiling
after the first epoch.

## 7. Evaluation (`scripts/evaluate_pneumonia.py`)

The test split (624 images) was used **once**, after training and threshold selection were complete.

| Metric | Threshold 0.029 (chosen on validation) | Threshold 0.5 (for comparison) |
|---|---|---|
| Accuracy | 0.930 | 0.968 |
| Precision (PPV) | 0.900 | 0.970 |
| Recall / sensitivity | **0.997** | 0.979 |
| Specificity | 0.816 | 0.949 |
| F1-score | 0.947 | 0.974 |
| ROC-AUC (threshold-independent) | 0.994 | 0.994 |
| Confusion matrix (TN, FP, FN, TP) | 191, 43, 1, 389 | 222, 12, 8, 382 |

### Discussion points for your presentation

- **Ranking is excellent (AUC 0.994), but calibration does not transfer.** On validation (same source as
  training), Youden J picked a very low threshold (0.029). On the test set (a separately collected folder), this
  threshold gives almost perfect sensitivity (1 missed case) but 43 false positives. At 0.5, the trade-off is more
  balanced.
- We **did not** switch to 0.5 after seeing the test results. Choosing a threshold on test data would leak test
  information. The app uses the threshold chosen before testing and reports both numbers openly. A proper fix
  would be to choose the threshold on a validation set that resembles the deployment data, or to recalibrate
  probabilities (for example with temperature scaling).
- In screening, high sensitivity is usually preferred, because missing pneumonia is worse than a false alarm.
  The cost is lower specificity.
- **Demo check:** of the 8 bundled sample X-rays (real test images), all 4 pneumonia cases are detected. Two of
  the 4 normals are false positives at the low threshold (P = 0.79 and P = 0.033). The second would be classified
  correctly at 0.5.

## 8. Explainability: Grad-CAM (`app/ml/pneumonia/gradcam.py`)

For the PNEUMONIA logit *y*, the gradients ∂y/∂A_k of the last convolutional block (`layer4`, 512 × 7 × 7) are
averaged over space to give weights α_k. The map is ReLU(Σ α_k A_k), upsampled to the image and normalised to [0, 1].
A unit test checks its shape and range.

**Measured localisation.** First measured on 26 Oct 2026 on the 8 sample X-rays with the earlier lung-mask heuristic
(CAM share inside the mask 0.16–0.42, about equal to the mask's share of the image, 0.16–0.39). Re-measured on
28 Oct 2026 with the improved classical lung-field segmentation (`scripts/evaluate_segmentation.py` →
`reports/segmentation_evaluation.json`): on 200 random test images, 156 had masks passing the plausibility checks;
for those, the mean share of Grad-CAM mass inside the lung fields was **0.304** versus a lung-field area share of
**0.316** (median ratio 1.11). The maps therefore do **not** concentrate on the lung fields more than their area
would predict. The classifier may be relying partly on non-pulmonary regions (borders, mediastinum,
image-acquisition cues), which is a known risk with this dataset. The lung masks come from classical image
processing without ground truth, and they tend to miss consolidated (white) lung, so this is indicative rather than
conclusive.

Limitations: the map has 7×7 resolution, so it is coarse. It explains the model's decision, not the location of
disease, and it can highlight artefacts such as text markers or borders.

## 9. Inference and API (`app/ml/pneumonia/inference.py`)

- The checkpoint is loaded once and cached. It reloads automatically if the file changes.
- An incomplete or missing checkpoint gives HTTP 409 `model_unavailable` with instructions.
- Upload checks: non-empty, ≤ 10 MB, decodable PNG/JPEG/BMP/TIFF, at least 32×32 pixels. The response includes a
  warning for colour photos (mean HSV saturation > 0.15) or extreme aspect ratios.
- The response contains the label, P(pneumonia), both class probabilities, the threshold, the model name, a
  Grad-CAM overlay and raw map, warnings, and a disclaimer.

## 10. Independent audit (`scripts/audit_pneumonia.py`)

The audit recomputes everything with plain NumPy/scikit-learn instead of the project's metric code, and writes
`models/pneumonia/audit.json`. All 9 checks passed (26 Oct 2026):

| Check | Result |
|---|---|
| Dataset file SHA-256 equals the download manifest | pass |
| Checkpoint metadata complete (arch, classes, img_size, threshold; source `pneumoniamnist`) | pass |
| `train_samples` = 4,708 = size of the training split | pass |
| `val_samples` = 516 = 524 − 8 confirmed duplicates | pass |
| No confirmed duplicates between test and train/val | pass |
| Threshold re-derived from validation (Youden J) = stored 0.028754 | pass |
| Test metrics at 0.0288 match `evaluation.json` (acc 0.9295, prec 0.9005, rec 0.9974, spec 0.8162, F1 0.9465, AUC 0.9939; TN 191, FP 43, FN 1, TP 389) | pass |
| Test metrics at 0.5 match (acc 0.9679, prec 0.9695, rec 0.9795, spec 0.9487, F1 0.9745; TN 222, FP 12, FN 8, TP 382) | pass |
| Test split = official test set (624 images: 390 pneumonia, 234 normal) | pass |

## 11. Reproducing

```powershell
backend\.venv\Scripts\python.exe scripts\download_datasets.py --only pneumoniamnist
backend\.venv\Scripts\python.exe scripts\inspect_pneumonia_dataset.py
backend\.venv\Scripts\python.exe scripts\train_pneumonia.py --epochs 6
backend\.venv\Scripts\python.exe scripts\evaluate_pneumonia.py
backend\.venv\Scripts\python.exe scriptsudit_pneumonia.py
```

Results can differ slightly between machines and PyTorch versions, because CPU kernels are not bit-for-bit deterministic.

## 12. Limitations

Paediatric patients only (ages 1–5, Guangzhou). Down-sampled, pre-cropped images. No external validation. No
patient-level leakage check for MedMNIST. Not calibrated. Not a medical device.
