"""Evaluate the trained checkpoint on the held-out test split.

Usage: python scripts/evaluate_pneumonia.py [--checkpoint models/pneumonia/pneumonia_resnet18.pt]
Writes models/pneumonia/evaluation.json, confusion_matrix.png and roc_curve.png.
With --source kaggle (a different source than the model was trained on) it writes
evaluation_kaggle.json and *_kaggle.png instead, leaving the official results untouched.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from app.ml.pneumonia.evaluate import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    a = p.parse_args()
    evaluate(a.checkpoint, a.source, max_samples=a.max_samples)


if __name__ == "__main__":
    main()
