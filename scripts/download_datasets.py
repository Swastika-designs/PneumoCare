"""Download the public datasets used by PneumoCare.

Every dataset is downloaded from its original public source into ``data/raw/<name>/``
and a ``_manifest.json`` file records the source URL, file sizes, SHA-256 hashes and
download time. The application's dataset registry reads these manifests to decide
whether a dataset is available.

Usage (from the project root):
    python scripts/download_datasets.py                 # all datasets
    python scripts/download_datasets.py --only synthea nhanes
    python scripts/download_datasets.py --list

The Kaggle "Chest X-Ray Images (Pneumonia)" dataset needs a Kaggle account and is NOT
downloaded by this script. See docs/dataset_registry.md for manual instructions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
NHANES_FILES = ["DEMO_J", "BMX_J", "PAQ_J", "SLQ_J", "SMQ_J", "ALQ_J", "DR1TOT_J"]

DATASETS: dict[str, dict] = {
    "pneumoniamnist": {
        "description": "PneumoniaMNIST (MedMNIST+ 224x224) chest X-rays, derived from Kermany et al. 2018",
        "files": {"pneumoniamnist_224.npz": "https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz"},
    },
    "synthea": {
        "description": "Synthea synthetic EHR sample data (CSV export)",
        "files": {
            "synthea_sample_data_csv_latest.zip":
                "https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip"
        },
        "extract": True,
    },
    "medical_abstracts": {
        "description": "Medical Abstracts Text Classification Corpus (Schopf et al. 2022), 5 disease classes",
        "files": {
            name: f"https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/{name}"
            for name in ["medical_tc_train.csv", "medical_tc_test.csv", "medical_tc_labels.csv"]
        },
    },
    "nhanes": {
        "description": "NHANES 2017-2018 demographics, body measures and lifestyle questionnaires (CDC)",
        "files": {f"{name}.xpt": f"{NHANES_BASE}{name}.xpt" for name in NHANES_FILES},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "PneumoCare-course-project/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {dest.name}: {done / total:6.1%} of {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def _safe_extract(archive: Path, target: Path) -> list[str]:
    """Extract CSV files from a zip, flattening folders and refusing path traversal."""
    extracted = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.lower().endswith(".csv"):
                continue
            with zf.open(member) as src, (target / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(name)
    return extracted


def download_dataset(key: str, force: bool = False) -> None:
    spec = DATASETS[key]
    target = RAW / key
    target.mkdir(parents=True, exist_ok=True)
    print(f"[{key}] {spec['description']}")
    files = []
    for filename, url in spec["files"].items():
        dest = target / filename
        if dest.exists() and not force:
            print(f"    {filename}: already present, skipping (use --force to re-download)")
        else:
            _download(url, dest)
        files.append({"name": filename, "url": url, "bytes": dest.stat().st_size, "sha256": _sha256(dest)})
    extracted = []
    if spec.get("extract"):
        for item in files:
            extracted += _safe_extract(target / item["name"], target)
        print(f"    extracted {len(extracted)} CSV files")
    manifest = {
        "dataset": key,
        "description": spec["description"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "extracted": extracted,
    }
    (target / "_manifest.json").write_text(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(DATASETS), help="download only these datasets")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--list", action="store_true", help="list datasets and exit")
    args = parser.parse_args(argv)
    if args.list:
        for key, spec in DATASETS.items():
            print(f"{key:18s} {spec['description']}")
        return 0
    failures = []
    for key in args.only or DATASETS:
        try:
            download_dataset(key, force=args.force)
        except Exception as exc:  # keep going so one outage does not block the others
            print(f"    FAILED: {exc}")
            failures.append(key)
    if failures:
        print(f"Failed datasets: {', '.join(failures)}")
        return 1
    print("All requested datasets downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
