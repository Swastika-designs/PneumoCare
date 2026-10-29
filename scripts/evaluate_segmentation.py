"""Evaluate the classical lung-field segmentation (T4) on the PneumoniaMNIST test split.

There are no ground-truth lung masks for PneumoniaMNIST, so this does NOT measure accuracy (no Dice/IoU).
It reports failure signs and anatomical plausibility for:
  - the current method (`imaging.lung_fields`: coarse-scale candidates + marker-controlled watershed), and
  - the previous heuristic (inverse Otsu + largest dark component per half + corner flood fill), kept here only
    as a baseline;
and, for images whose mask passes the plausibility checks, the share of Grad-CAM mass inside the lung fields.

Usage: python scripts/evaluate_segmentation.py   (writes reports/segmentation_evaluation.json, ~1 min on CPU)
"""

import json
import time
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.ml.pneumonia.data import medmnist_path
from app.ml.pneumonia.inference import get_predictor
from app.services import imaging

OUT = settings.reports_dir / "segmentation_evaluation.json"


def previous_heuristic(img: np.ndarray) -> dict[str, np.ndarray]:
    """The segmentation used before 28 Oct 2026, reproduced for comparison."""
    h, w = img.shape
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(img, (5, 5), 0))
    _, dark = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, h // 60) | 1,) * 2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=2)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    best = {"left": (0, None), "right": (0, None)}
    for i in range(1, n):
        if stats[i][2] > 0.6 * w or stats[i][4] < 0.01 * h * w:
            continue
        side = "left" if centroids[i][0] < w / 2 else "right"
        if stats[i][4] > best[side][0]:
            best[side] = (stats[i][4], i)
    mask = np.zeros_like(img)
    for _, i in best.values():
        if i is not None:
            mask[labels == i] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    mask = mask | cv2.bitwise_not(flood)
    fields = {}
    for side, cols in (("left", np.s_[:, : w // 2]), ("right", np.s_[:, w // 2:])):
        half = np.zeros_like(mask)
        half[cols] = mask[cols]
        half = imaging._largest_component(half)
        if half.any():
            fields[side] = half
    return fields


def summarise(rows: list[tuple[float, bool, int]], labels: np.ndarray) -> dict:
    a = np.array(rows, dtype=float)
    out = {}
    for name, sel in (("all", np.ones(len(a), bool)), ("normal", labels == 0), ("pneumonia", labels == 1)):
        b = a[sel]
        out[name] = {
            "images": int(len(b)),
            "masks_over_60pct_of_image": int((b[:, 0] > 0.6).sum()),
            "two_fields_pct": round(float((b[:, 2] == 2).mean() * 100), 1),
            "all_plausibility_checks_pass_pct": round(float(b[:, 1].mean() * 100), 1),
            "median_mask_area_pct": round(float(np.median(b[:, 0]) * 100), 1),
        }
    return out


def main() -> None:
    data = np.load(medmnist_path())
    images, labels = data["test_images"], data["test_labels"].ravel()
    current, previous = [], []
    t0 = time.time()
    for img in images:
        r = imaging.lung_fields(img)
        current.append((float((r["mask"] > 0).mean()), r["quality"]["plausible"], len(r["fields"])))
    ms_current = (time.time() - t0) / len(images) * 1000
    for img in images:
        fields = previous_heuristic(img)
        mask = np.zeros_like(img)
        for m in fields.values():
            mask |= m
        q = imaging._plausibility(fields, *img.shape)
        previous.append((float((mask > 0).mean()), q["plausible"], len(fields)))

    predictor = get_predictor()
    rng = np.random.default_rng(1)
    shares = []
    for i in rng.choice(len(images), 200, replace=False):
        r = imaging.lung_fields(images[i])
        if not r["quality"]["plausible"]:
            continue
        cam = predictor.predict(Image.fromarray(images[i]))["gradcam"]["heatmap"]
        cam = cv2.resize(np.asarray(cam, np.float32), images[i].shape[::-1])
        inside = r["mask"] > 0
        shares.append((float(cam[inside].sum() / max(cam.sum(), 1e-9)), float(inside.mean())))
    s = np.array(shares)
    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "PneumoniaMNIST 224 test split",
        "ground_truth": "none - plausibility and failure signs only, not accuracy",
        "plausibility_checks": [c["check"] for c in imaging._plausibility({}, 1, 1)["checks"]],
        "current_method": summarise(current, labels),
        "previous_heuristic": summarise(previous, labels),
        "ms_per_image_current": round(ms_current, 1),
        "gradcam_inside_lung_fields": {
            "images_sampled": 200,
            "images_with_plausible_mask": int(len(s)),
            "mean_cam_share_inside": round(float(s[:, 0].mean()), 3),
            "mean_lung_area_share": round(float(s[:, 1].mean()), 3),
            "median_ratio": round(float(np.median(s[:, 0] / s[:, 1])), 2),
            "note": "A ratio near 1 means Grad-CAM mass is spread in proportion to area, "
                    "not concentrated in the lungs.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
