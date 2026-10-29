"""Inspect the chest X-ray dataset: class balance, image statistics and split leakage.

Usage: python scripts/inspect_pneumonia_dataset.py [--source auto|kaggle|pneumoniamnist]
Writes models/pneumonia/dataset_inspection.json
"""
import argparse
import json

import _bootstrap  # noqa: F401

from app.ml.pneumonia.data import inspect_dataset, leakage_report, load_dataset
from app.ml.pneumonia.model import checkpoint_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto")
    parser.add_argument("--sample", type=int, default=300, help="images per split for size/intensity stats")
    args = parser.parse_args()
    dataset = load_dataset(args.source)
    report = inspect_dataset(dataset, sample_per_split=args.sample)
    print(f"Source: {report['source']} ({report['root']})")
    for name, split in report["splits"].items():
        print(f"  {name:5s} total={split['total']:5d} {split['class_counts']} "
              f"pneumonia={split['pneumonia_fraction']:.1%} size~{split['width']['mean']}x{split['height']['mean']}")
    print("Checking for duplicate images across splits ...")
    report["leakage"] = leakage_report(dataset)
    leak = report["leakage"]
    print(f"  hash collisions: {leak['hash_collisions_between_splits']} confirmed: {leak['confirmed_near_duplicates']}")
    if "patient_overlap_between_splits" in report["leakage"]:
        print(f"  patient overlap: {report['leakage']['patient_overlap_between_splits']}")
    out = checkpoint_dir() / "dataset_inspection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
