"""Chest X-ray dataset loading, inspection, leakage checks and preprocessing.

Two real data sources are supported:

1. **Kaggle "Chest X-Ray Images (Pneumonia)"** (Kermany et al.) unpacked at
   ``data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}``. Requires a manual,
   authenticated download. The official ``val`` folder holds only 16 images, so we
   merge train+val and re-split by *patient id* (parsed from file names) to avoid
   patient leakage. The official ``test`` folder is kept untouched.
2. **PneumoniaMNIST (MedMNIST+, 224x224)** at ``data/raw/pneumoniamnist/pneumoniamnist_224.npz``.
   It is derived from the same Kermany et al. images, with official train/val/test
   splits (test = Kaggle test set). Downloadable without authentication, so it is the
   default source used for the checkpoint shipped by ``scripts/train_pneumonia.py``.

``source="auto"`` always means PneumoniaMNIST when it is present; Kaggle must be requested
explicitly (``--source kaggle``).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import settings

CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".bmp"}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when no supported chest X-ray dataset is present on disk."""


@dataclass
class SplitData:
    """One split. ``images`` holds either uint8 arrays (npz) or file paths (Kaggle)."""

    images: list | np.ndarray
    labels: np.ndarray
    groups: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    def load(self, idx: int) -> Image.Image:
        item = self.images[idx]
        if isinstance(item, (str, Path)):
            return Image.open(item).convert("L")
        return Image.fromarray(np.asarray(item, dtype=np.uint8), mode="L")


@dataclass
class XrayDataset:
    source: str
    root: Path
    splits: dict[str, SplitData]
    notes: list[str] = field(default_factory=list)


def kaggle_root() -> Path:
    return settings.raw_dir / "chest_xray"


def medmnist_path() -> Path:
    return settings.raw_dir / "pneumoniamnist" / "pneumoniamnist_224.npz"


def available_sources() -> list[str]:
    """Available sources, default first. PneumoniaMNIST (the source of the shipped model) is
    always preferred by ``source="auto"``; Kaggle is only used when requested explicitly, so
    downloading it never silently changes which data training or the app uses."""
    sources = []
    if medmnist_path().exists():
        sources.append("pneumoniamnist")
    if (kaggle_root() / "train").is_dir() and (kaggle_root() / "test").is_dir():
        sources.append("kaggle")
    return sources


def _patient_id(path: Path) -> str:
    """Kaggle file names look like ``person1_bacteria_1.jpeg`` or ``IM-0115-0001.jpeg``."""
    name = path.stem
    match = re.match(r"(person\d+)", name)
    if match:
        return match.group(1)
    match = re.match(r"((?:NORMAL2-)?IM-\d+)", name)
    return match.group(1) if match else name


def _scan_kaggle_folder(folder: Path) -> tuple[list[Path], list[int]]:
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        cls_dir = folder / cls
        if not cls_dir.is_dir():
            continue
        for path in sorted(cls_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
                labels.append(label)
    return paths, labels


def _load_kaggle(val_fraction: float, seed: int) -> XrayDataset:
    from sklearn.model_selection import GroupShuffleSplit

    root = kaggle_root()
    train_paths, train_labels = _scan_kaggle_folder(root / "train")
    val_paths, val_labels = _scan_kaggle_folder(root / "val")
    test_paths, test_labels = _scan_kaggle_folder(root / "test")
    pool_paths = train_paths + val_paths
    pool_labels = np.array(train_labels + val_labels)
    groups = [f"{lbl}:{_patient_id(p)}" for p, lbl in zip(pool_paths, pool_labels)]
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, va_idx = next(splitter.split(pool_paths, pool_labels, groups))
    splits = {
        "train": SplitData([pool_paths[i] for i in tr_idx], pool_labels[tr_idx], [groups[i] for i in tr_idx]),
        "val": SplitData([pool_paths[i] for i in va_idx], pool_labels[va_idx], [groups[i] for i in va_idx]),
        "test": SplitData(test_paths, np.array(test_labels),
                          [f"{lbl}:{_patient_id(p)}" for p, lbl in zip(test_paths, test_labels)]),
    }
    notes = [
        f"Official val folder ({len(val_paths)} images) merged with train and re-split "
        f"{1 - val_fraction:.0%}/{val_fraction:.0%} by patient id (GroupShuffleSplit, seed={seed}).",
        "Official test folder kept unchanged.",
    ]
    return XrayDataset("kaggle", root, splits, notes)


def _load_medmnist() -> XrayDataset:
    path = medmnist_path()
    data = np.load(path, allow_pickle=False)
    splits = {
        split: SplitData(data[f"{split}_images"], data[f"{split}_labels"].ravel().astype(np.int64))
        for split in ("train", "val", "test")
    }
    notes = [
        "PneumoniaMNIST official splits: train/val come from the Kaggle training folder (9:1), "
        "test is the Kaggle test folder.",
        "Patient identifiers are not distributed with MedMNIST, so leakage is checked with image hashes only.",
    ]
    return XrayDataset("pneumoniamnist", path.parent, splits, notes)


def load_dataset(source: str = "auto", val_fraction: float = 0.1, seed: int = 42) -> XrayDataset:
    sources = available_sources()
    if not sources:
        raise DatasetNotFoundError(
            "No chest X-ray dataset found. Run `python scripts/download_datasets.py --only pneumoniamnist` "
            f"or unpack the Kaggle dataset into {kaggle_root()}."
        )
    if source == "auto":
        source = sources[0]
    if source not in sources:
        raise DatasetNotFoundError(f"Requested source '{source}' is not available. Available: {sources}")
    return _load_kaggle(val_fraction, seed) if source == "kaggle" else _load_medmnist()


def average_hash(image: Image.Image, hash_size: int = 8) -> int:
    """64-bit perceptual average hash. Identical/near-identical images share a hash."""
    small = np.asarray(image.convert("L").resize((hash_size, hash_size), Image.BILINEAR), dtype=np.float32)
    bits = (small > small.mean()).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def _thumb(image: Image.Image, size: int = 32) -> np.ndarray:
    return np.asarray(image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)


def leakage_report(dataset: XrayDataset, max_pixel_diff: float = 4.0) -> dict:
    """Check for images (and, for Kaggle, patients) that appear in more than one split.

    Stage 1 groups images by a 64-bit average hash. Stage 2 confirms each cross-split
    hash collision by comparing 32x32 thumbnails: a mean absolute difference below
    ``max_pixel_diff`` (on a 0-255 scale) counts as a confirmed near-duplicate.
    """
    index: dict[str, dict[int, list[int]]] = {}
    for name, split in dataset.splits.items():
        buckets: dict[int, list[int]] = {}
        for i in range(len(split)):
            buckets.setdefault(average_hash(split.load(i)), []).append(i)
        index[name] = buckets
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    collisions, confirmed, duplicate_pairs = {}, {}, []
    for a, b in pairs:
        shared = set(index[a]) & set(index[b])
        collisions[f"{a}-{b}"] = len(shared)
        n_confirmed = 0
        for h in shared:
            thumbs_a = [_thumb(dataset.splits[a].load(i)) for i in index[a][h]]
            for j in index[b][h]:
                tb = _thumb(dataset.splits[b].load(j))
                diffs = [float(np.abs(ta - tb).mean()) for ta in thumbs_a]
                if min(diffs) <= max_pixel_diff:
                    n_confirmed += 1
                    duplicate_pairs.append({"splits": f"{a}-{b}", "index_a": index[a][h][int(np.argmin(diffs))],
                                            "index_b": j, "mean_abs_diff": round(min(diffs), 2)})
        confirmed[f"{a}-{b}"] = n_confirmed
    report = {
        "method": "Stage 1: 64-bit average hash; stage 2: 32x32 thumbnail mean absolute difference "
                  f"<= {max_pixel_diff} grey levels confirms a near-duplicate.",
        "hash_collisions_between_splits": collisions,
        "confirmed_near_duplicates": confirmed,
        "duplicate_pairs": duplicate_pairs,
        "images_per_split": {k: len(v) for k, v in dataset.splits.items()},
    }
    if dataset.source == "kaggle":
        groups = {k: set(v.groups) for k, v in dataset.splits.items()}
        report["patient_overlap_between_splits"] = {f"{a}-{b}": len(groups[a] & groups[b]) for a, b in pairs}
    report["note"] = (
        "Hash collisions alone are weak evidence because chest X-rays look alike at 8x8 pixels; only confirmed "
        "near-duplicates indicate likely leakage. MedMNIST does not ship patient IDs, so patient-level overlap "
        "cannot be checked for that source."
    )
    return report


def inspect_dataset(dataset: XrayDataset, sample_per_split: int = 200, seed: int = 0) -> dict:
    """Class distribution, image size and intensity statistics per split."""
    rng = np.random.default_rng(seed)
    summary = {"source": dataset.source, "root": str(dataset.root), "classes": CLASSES, "notes": dataset.notes,
               "splits": {}}
    for name, split in dataset.splits.items():
        counts = Counter(int(x) for x in split.labels)
        idx = rng.choice(len(split), size=min(sample_per_split, len(split)), replace=False)
        sizes, means, stds = [], [], []
        for i in idx:
            img = split.load(int(i))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            sizes.append(img.size)
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        summary["splits"][name] = {
            "total": len(split),
            "class_counts": {CLASSES[k]: counts.get(k, 0) for k in range(len(CLASSES))},
            "pneumonia_fraction": round(counts.get(1, 0) / max(len(split), 1), 4),
            "sampled_images": len(idx),
            "width": {"min": int(min(widths)), "max": int(max(widths)), "mean": round(float(np.mean(widths)), 1)},
            "height": {"min": int(min(heights)), "max": int(max(heights)), "mean": round(float(np.mean(heights)), 1)},
            "intensity_mean": round(float(np.mean(means)), 4),
            "intensity_std": round(float(np.mean(stds)), 4),
        }
    return summary


# --------------------------------------------------------------------------- torch helpers


def build_transforms(img_size: int, train: bool):
    from torchvision import transforms as T

    common = [T.Grayscale(num_output_channels=3), T.Resize((img_size, img_size))]
    augment = [
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ] if train else []
    return T.Compose(common + augment + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class TorchXrayDataset:
    """Thin ``torch.utils.data.Dataset``-compatible wrapper around a :class:`SplitData`."""

    def __init__(self, split: SplitData, transform, indices: np.ndarray | None = None):
        self.split = split
        self.transform = transform
        self.indices = np.arange(len(split)) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        return self.transform(self.split.load(idx)), int(self.split.labels[idx])
