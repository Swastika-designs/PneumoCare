"""Install the Kaggle "Chest X-Ray Images (Pneumonia)" archive into data/raw/chest_xray.

The Kaggle zip contains the same folders more than once (``chest_xray/...``, ``chest_xray/chest_xray/...``
and ``__MACOSX`` metadata). Only real images under ``<split>/<CLASS>/`` are kept, each file once.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from app.core.config import settings
from app.ml.pneumonia.data import CLASSES, IMAGE_SUFFIXES, kaggle_root

DATASET_REF = "paultimothymooney/chest-xray-pneumonia"
API_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"
SPLITS = ("train", "val", "test")


class KaggleCredentialsMissing(RuntimeError):
    pass


def find_credentials() -> tuple[str, str]:
    """KAGGLE_USERNAME/KAGGLE_KEY environment variables, or ~/.kaggle/kaggle.json."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data["username"], data["key"]
    raise KaggleCredentialsMissing(
        "No Kaggle credentials found. Create an API token at https://www.kaggle.com/settings (API -> Create New "
        "Token) and save kaggle.json to ~/.kaggle/, or set KAGGLE_USERNAME and KAGGLE_KEY. Alternatively download "
        "the zip in your browser and run: python scripts/download_kaggle.py --zip <path-to-archive.zip>")


def download_archive(dest: Path) -> Path:
    user, key = find_credentials()
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    request = urllib.request.Request(API_URL, headers={"Authorization": f"Basic {token}",
                                                      "User-Agent": "PneumoCare-course-project/1.0"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=600) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    return dest


def _target_for(member: str) -> tuple[str, str, str] | None:
    parts = PurePosixPath(member).parts
    if "__MACOSX" in parts or len(parts) < 3:
        return None
    split, cls, name = parts[-3].lower(), parts[-2].upper(), parts[-1]
    if split not in SPLITS or cls not in CLASSES or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        return None
    if name.startswith("."):
        return None
    return split, cls, name


def install_archive(zip_path: Path, target: Path | None = None) -> dict:
    """Extract images into ``target/<split>/<CLASS>/`` and write a manifest. Returns per-split counts."""
    target = target or kaggle_root()
    counts: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    seen: set[tuple[str, str, str]] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            dest = _target_for(info.filename)
            if dest is None or dest in seen:
                continue
            seen.add(dest)
            out = target.joinpath(*dest)
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out.open("wb") as fh:
                shutil.copyfileobj(src, fh)
            counts[dest[0]][dest[1]] += 1
    if not counts["train"]["NORMAL"] or not counts["test"]["NORMAL"]:
        raise ValueError(f"{zip_path.name} does not look like the Kaggle chest X-ray archive (no train/test images).")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest() if zip_path.stat().st_size < 2_000_000_000 else None
    manifest = {"dataset": "kaggle_chest_xray", "source": f"https://www.kaggle.com/datasets/{DATASET_REF}",
                "archive": zip_path.name, "archive_sha256": digest,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "counts": counts, "files": [{"name": zip_path.name, "bytes": zip_path.stat().st_size}]}
    (target / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    return counts


def manifest_path() -> Path:
    return settings.raw_dir / "chest_xray" / "_manifest.json"
