# Dataset Registry

The application computes availability from the files on disk (`GET /api/datasets`, page **Datasets & Models**).
Each scripted download writes `data/raw/<id>/_manifest.json` with the URL, size, SHA-256 and download time.

| ID | Dataset | Domain | Real / synthetic | Access | Status (26 Oct 2026) |
|---|---|---|---|---|---|
| `pneumoniamnist` | PneumoniaMNIST, MedMNIST+ 224×224 | Imaging | Real (de-identified) | Public, scripted | Downloaded (214 MB) |
| `kaggle_chest_xray` | Chest X-Ray Images (Pneumonia), Kaggle | Imaging | Real (de-identified) | **Optional**; your own Kaggle account | Not installed (no credentials on this machine) |
| `synthea` | Synthea sample data, CSV | EHR / operations | **Synthetic** | Public, scripted | Downloaded (5.4 MB zip, 18 CSVs) |
| `medical_abstracts` | Medical Abstracts TC Corpus | Clinical text | Real (published abstracts) | Public, scripted | Downloaded (17.8 MB) |
| `nhanes` | NHANES 2017–2018 (7 files) | Lifestyle | Real (public-use survey) | Public, scripted | Downloaded (20.6 MB) |
| `wsd_examples` | Abbreviation sense examples | Clinical text | **Synthetic, hand-written** | Bundled in code | Available |

## Sources and licences

| ID | URL | Licence / terms |
|---|---|---|
| pneumoniamnist | https://zenodo.org/records/10519652 (file `pneumoniamnist_224.npz`); https://medmnist.com/ | CC BY 4.0 |
| kaggle_chest_xray | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia | CC BY 4.0 |
| synthea | https://synthea.mitre.org/downloads (`synthea_sample_data_csv_latest.zip`) | Synthea: Apache 2.0; data are fully synthetic |
| medical_abstracts | https://github.com/sebischair/Medical-Abstracts-TC-Corpus | See the repository |
| nhanes | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ (`DEMO_J`, `BMX_J`, `PAQ_J`, `SLQ_J`, `SMQ_J`, `ALQ_J`, `DR1TOT_J`) | U.S. public-use data (CDC/NCHS) |
| wsd_examples | `backend/app/services/wsd_data.py` | Written for this project |

## Download commands

```powershell
backend\.venv\Scripts\python.exe scripts\download_datasets.py --list
backend\.venv\Scripts\python.exe scripts\download_datasets.py                 # everything (~285 MB)
backend\.venv\Scripts\python.exe scripts\download_datasets.py --only synthea nhanes
```

## Optional: Kaggle chest X-ray dataset

**Status of the dependency:** nothing in the application requires it. The registry marks it `required: false`; the
dashboard counts only required datasets (5/5 available) and shows Kaggle as "optional". The shipped model was
trained on PneumoniaMNIST (the same Kermany et al. images), and this is stated in the registry and the docs.

To install it (about 1.2 GB):

1. **With an API token:** at kaggle.com → Settings → API → *Create New Token*, save `kaggle.json` to
   `%USERPROFILE%\.kaggle\` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY`), then run
   `backend\.venv\Scripts\python.exe scripts\download_kaggle.py`.
2. **Or in the browser:** download the dataset zip from the Kaggle page, then run
   `backend\.venv\Scripts\python.exe scripts\download_kaggle.py --zip C:/Users/you/Downloads/archive.zip`.

The installer keeps only `train|val|test/NORMAL|PNEUMONIA` images, removes the archive's duplicate nested copy and
`__MACOSX` files, and writes `data/raw/chest_xray/_manifest.json` with per-split counts. Without credentials it
exits with instructions (exit code 2). This behaviour was verified in this audit.

After installation nothing changes automatically (`source="auto"` stays PneumoniaMNIST). Optional next steps:
- `scripts/evaluate_pneumonia.py --source kaggle`: external check of the **existing** model on the full-resolution
  test images. Writes `evaluation_kaggle.json`; the official results are not overwritten.
- `scripts/inspect_pneumonia_dataset.py --source kaggle` / `scripts/train_pneumonia.py --source kaggle`: retraining.
  The loader re-splits train/val by patient ID.

## Derived and teaching datasets (computed on demand, not stored)

| Source ID | Built from | Notes |
|---|---|---|
| `synthea:patient_profile` | patients + encounters + conditions + medications + latest of 19 observations | One row per patient. Lab panels are naturally missing for 40–68% of patients |
| `teaching:patients_messy` | Synthea `patients.csv` | Deterministic (seed 7) injection of 14 documented problems: coding variants, mixed date formats, currency strings, lost ZIP zeros, invalid sentinels, text null tokens, exact and near duplicates. Used for T2/T3 with a ground-truth comparison |
| `nhanes_adults` | 7 NHANES files merged on SEQN | Adults ≥ 20 years, not pregnant, BMI measured (n = 5,120). Survey codes 7/9/77/99/777/7777/9999 recoded as missing |

## Data handling rules

- Image, EHR, text and lifestyle data are never merged with each other.
- Raw data and uploads are git-ignored. Only 8 small demo X-rays exported from the public test split are kept in
  `data/sample/xray/`. Never use them for training.
