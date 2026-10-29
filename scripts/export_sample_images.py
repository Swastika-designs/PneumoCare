"""Export a few PneumoniaMNIST test images as PNG files for demos (data/sample/xray/).

These are real images from the public PneumoniaMNIST test split (CC BY 4.0). Because they
come from the *test* split, never use them for training.

Usage: python scripts/export_sample_images.py [--per-class 4]
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np

from app.core.config import settings
from app.ml.pneumonia.data import CLASSES, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=4)
    args = parser.parse_args()
    split = load_dataset("pneumoniamnist").splits["test"]
    out = settings.sample_dir / "xray"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for label, name in enumerate(CLASSES):
        for idx in rng.choice(np.flatnonzero(split.labels == label), size=args.per_class, replace=False):
            path = out / f"{name.lower()}_test_{idx:04d}.png"
            split.load(int(idx)).save(path)
            print(f"saved {path.name}")


if __name__ == "__main__":
    main()
