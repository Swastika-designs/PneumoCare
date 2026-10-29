"""Install the optional Kaggle "Chest X-Ray Images (Pneumonia)" dataset (needs YOUR Kaggle account).

Two ways:
    python scripts/download_kaggle.py                      # uses ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
    python scripts/download_kaggle.py --zip archive.zip    # zip you downloaded in the browser

Afterwards nothing changes automatically: the app keeps using PneumoniaMNIST and the existing model.
To evaluate the existing model on the full-resolution Kaggle test images (no retraining):
    python scripts/evaluate_pneumonia.py --source kaggle   # writes models/pneumonia/evaluation_kaggle.json
"""
import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from app.core.config import settings
from app.ml.pneumonia.kaggle import KaggleCredentialsMissing, download_archive, install_archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, help="install from an already downloaded archive")
    args = parser.parse_args()
    archive = args.zip
    if archive is None:
        archive = settings.raw_dir / "kaggle_download" / "chest-xray-pneumonia.zip"
        if not archive.exists():
            try:
                print("Downloading from the Kaggle API (about 1.2 GB) ...")
                download_archive(archive)
            except KaggleCredentialsMissing as exc:
                print(exc)
                return 2
    counts = install_archive(archive)
    for split, by_class in counts.items():
        print(f"  {split:5s} {by_class}")
    print(f"Installed into {settings.raw_dir / 'chest_xray'}. Use --source kaggle to evaluate or train on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
