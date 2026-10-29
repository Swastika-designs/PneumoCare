"""T4-T6: biomedical image analysis, feature extraction and visualisation.

All functions take a uint8 grayscale numpy array and return JSON-ready dicts with images
encoded as PNG data URLs.
"""
from __future__ import annotations

import io
import json
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.errors import NotFound, ValidationFailed
from app.core.utils import png_base64, to_jsonable

# ------------------------------------------------------------------ metrics


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity (Wang et al., 2004) with an 11x11 Gaussian window."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    var_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    var_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def entropy(img: np.ndarray) -> float:
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def image_stats(img: np.ndarray) -> dict:
    return {"mean": float(img.mean()), "std": float(img.std()), "min": int(img.min()), "max": int(img.max()),
            "entropy_bits": entropy(img)}


def histogram(img: np.ndarray, bins: int = 64) -> list[dict]:
    counts, edges = np.histogram(img, bins=bins, range=(0, 256))
    cdf = np.cumsum(counts) / counts.sum()
    return [{"intensity": int(edges[i]), "count": int(c), "cdf": float(cdf[i])} for i, c in enumerate(counts)]


def _panel(title: str, img: np.ndarray, **extra) -> dict:
    gray = img if img.ndim == 2 else None
    panel = {"title": title, "image": png_base64(img)}
    if gray is not None:
        panel["stats"] = image_stats(gray)
    panel.update(extra)
    return to_jsonable(panel)


# ------------------------------------------------------------------ T4: enhancement, restoration, filtering


def enhance(img: np.ndarray, clip_limit: float = 2.0, gamma: float = 0.8) -> dict:
    if not 0.1 <= gamma <= 5:
        raise ValidationFailed("gamma must be between 0.1 and 5")
    lo, hi = np.percentile(img, (2, 98))
    stretched = np.clip((img.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    equalized = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(img)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    gamma_img = cv2.LUT(img, lut)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    unsharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    outputs = [
        ("Original", img, "Input image"),
        ("Contrast stretching (2-98th pct)", stretched, "Linearly maps the 2nd-98th percentile range to 0-255"),
        ("Histogram equalisation", equalized, "Flattens the global intensity histogram"),
        (f"CLAHE (clip {clip_limit:g})", clahe, "Contrast-limited adaptive equalisation on 8x8 tiles"),
        (f"Gamma correction (gamma {gamma:g})", gamma_img, "gamma < 1 brightens dark regions, > 1 darkens"),
        ("Unsharp masking", unsharp, "Adds back high-frequency detail: 1.5*I - 0.5*blur(I)"),
    ]
    return {"panels": [_panel(t, im, description=d, histogram=histogram(im)) for t, im, d in outputs]}


def restore(img: np.ndarray, noise: str = "gaussian", level: float = 20.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    if noise == "gaussian":
        noisy = np.clip(img.astype(np.float32) + rng.normal(0, level, img.shape), 0, 255).astype(np.uint8)
    elif noise == "salt_pepper":
        amount = min(max(level / 100.0, 0.0), 0.5)
        noisy = img.copy()
        mask = rng.random(img.shape)
        noisy[mask < amount / 2] = 0
        noisy[mask > 1 - amount / 2] = 255
    elif noise == "speckle":
        noisy = np.clip(img * (1 + rng.normal(0, level / 100.0, img.shape)), 0, 255).astype(np.uint8)
    else:
        raise ValidationFailed("noise must be gaussian, salt_pepper or speckle")
    filters = {
        "Gaussian blur (5x5)": cv2.GaussianBlur(noisy, (5, 5), 0),
        "Median filter (5x5)": cv2.medianBlur(noisy, 5),
        "Bilateral filter": cv2.bilateralFilter(noisy, 7, 50, 50),
        "Non-local means": cv2.fastNlMeansDenoising(noisy, None, h=max(level * 0.8, 5), templateWindowSize=7,
                                                    searchWindowSize=21),
    }
    rows = [{"method": "Noisy input", "psnr": psnr(img, noisy), "ssim": ssim(img, noisy)}]
    panels = [_panel("Original", img), _panel(f"Noisy ({noise}, level {level:g})", noisy,
                                              psnr=psnr(img, noisy), ssim=ssim(img, noisy))]
    for name, out in filters.items():
        rows.append({"method": name, "psnr": psnr(img, out), "ssim": ssim(img, out)})
        panels.append(_panel(name, out, psnr=psnr(img, out), ssim=ssim(img, out)))
    best = max(rows[1:], key=lambda r: r["psnr"])["method"]
    return to_jsonable({"panels": panels, "metrics": rows, "best_by_psnr": best,
                        "note": "PSNR (dB) and SSIM are measured against the original clean image."})


def filters(img: np.ndarray, low: int = 50, high: int = 150) -> dict:
    sobel = cv2.magnitude(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lap = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F, ksize=3))
    canny = cv2.Canny(img, low, high)
    highpass = cv2.normalize(img.astype(np.float32) - cv2.GaussianBlur(img, (0, 0), 5).astype(np.float32), None, 0,
                             255, cv2.NORM_MINMAX).astype(np.uint8)
    outputs = [
        ("Original", img, "Input"),
        ("Gaussian low-pass (sigma 2)", cv2.GaussianBlur(img, (0, 0), 2), "Smooths noise and fine detail"),
        ("Median (5x5)", cv2.medianBlur(img, 5), "Non-linear; preserves edges, removes impulse noise"),
        ("Sobel gradient magnitude", sobel, "First derivative: strength of edges in any direction"),
        ("Laplacian", lap, "Second derivative: highlights rapid intensity changes"),
        (f"Canny edges ({low}/{high})", canny, "Gradient + non-max suppression + hysteresis thresholds"),
        ("High-pass (I - blur(I))", highpass, "Keeps only fine detail / texture"),
    ]
    return {"panels": [_panel(t, im, description=d) for t, im, d in outputs],
            "edge_pixels_pct": float((canny > 0).mean() * 100)}


def threshold(img: np.ndarray, value: int = 128, block_size: int = 31) -> dict:
    if block_size % 2 == 0 or block_size < 3:
        raise ValidationFailed("block_size must be an odd number >= 3")
    _, global_t = cv2.threshold(img, value, 255, cv2.THRESH_BINARY)
    otsu_value, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tri_value, triangle = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2)
    adaptive_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                           block_size, 2)
    outputs = [
        (f"Global (T = {value})", global_t, value),
        (f"Otsu (T = {otsu_value:.0f})", otsu, otsu_value),
        (f"Triangle (T = {tri_value:.0f})", triangle, tri_value),
        (f"Adaptive mean ({block_size}x{block_size})", adaptive_mean, None),
        (f"Adaptive Gaussian ({block_size}x{block_size})", adaptive_gauss, None),
    ]
    return to_jsonable({
        "panels": [_panel("Original", img, histogram=histogram(img))] + [
            _panel(t, im, threshold=v, foreground_pct=float((im > 0).mean() * 100)) for t, im, v in outputs],
        "otsu_threshold": otsu_value,
        "histogram": histogram(img),
    })


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    # Fill outer contours instead of flood-filling from a corner: a flood seeded inside the mask filled the whole image.
    filled = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask)
    return np.where(labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)


def _midline(img: np.ndarray) -> int:
    """Column of the mediastinum/spine: the brightest vertical band in the central 30% of the image."""
    h, w = img.shape
    profile = img[int(0.15 * h):int(0.85 * h)].mean(axis=0)
    width = max(3, w // 20)
    profile = np.convolve(profile, np.ones(width) / width, mode="same")
    lo, hi = int(0.35 * w), int(0.65 * w)
    return lo + int(np.argmax(profile[lo:hi]))


def lung_fields(img: np.ndarray) -> dict:
    """Classical lung-field segmentation: coarse-scale candidates, anatomical rules and a marker-controlled watershed.

    It finds dark, lung-shaped regions either side of the mediastinum. It has no training data or ground truth,
    so it is not a validated lung segmentation model; consolidated (white) lung is often missed.
    """
    h, w = img.shape
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(img, (5, 5), 0))
    coarse = cv2.GaussianBlur(eq, (0, 0), w / 45)  # ribs blur away, the lung fields stay dark
    mid = _midline(coarse)
    otsu_t, dark = cv2.threshold(coarse, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    band = max(2, int(0.03 * w))
    dark[:, mid - band:mid + band] = 0
    full_dark = dark.copy()
    # Ignore a thin frame while choosing candidates so lungs detach from air around the shoulders and body edge.
    ft, fs = int(0.06 * h), int(0.05 * w)
    dark[:ft] = 0
    dark[:, :fs] = 0
    dark[:, w - fs:] = 0
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    best: dict[str, tuple[int, int | None]] = {"left": (0, None), "right": (0, None)}
    rejected: list[str] = []
    outside = np.zeros((h, w), bool)
    for i in range(1, n):
        x, y, bw, _bh, area = stats[i]
        comp = labels == i
        reason = None
        if area < 0.02 * h * w:
            reason = "smaller than 2% of the image"
        elif bw > 0.6 * w:
            reason = "spans the body (background air)"
        elif img[comp].mean() < 25:
            reason = "empty air (mean grey level < 25)"
        elif bw < 0.1 * w and (x <= fs or x + bw >= w - fs):
            reason = "thin strip along the image edge"
        elif y <= ft and centroids[i][1] < 0.2 * h:
            reason = "air above the shoulders"
        elif centroids[i][1] > 0.8 * h:
            reason = "below the diaphragm (bowel gas)"
        if reason:
            rejected.append(reason)
            if area >= 0.02 * h * w:
                outside |= comp
            continue
        side = "left" if centroids[i][0] < mid else "right"
        if area > best[side][0]:
            best[side] = (area, i)
    chosen = np.zeros((h, w), bool)
    for _, i in best.values():
        if i is not None:
            chosen |= labels == i
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(full_dark)
    for j in range(1, n2):
        x, y, bw, _bh, _area = stats2[j]
        comp = labels2 == j
        if (y <= ft or x <= fs or x + bw >= w - fs) and not (comp & chosen).any():
            outside |= comp
    # Markers: 1 = background (bright tissue, mediastinum, outside air), 2/3 = eroded lung candidates.
    markers = np.zeros((h, w), np.int32)
    markers[coarse > otsu_t + 0.35 * (255 - otsu_t)] = 1
    markers[:, mid - band:mid + band] = 1
    markers[(img < 15) | outside] = 1
    if chosen.any():
        # Near the side edges, pixels much darker than the chosen lungs are air outside the body (e.g. a black border).
        side_band = np.zeros((h, w), bool)
        side_band[:, :int(0.08 * w)] = side_band[:, w - int(0.08 * w):] = True
        markers[side_band & (img < 0.6 * np.median(img[chosen]))] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, h // 45) | 1,) * 2)
    seeds = {}
    for label, (side, (_, i)) in zip((2, 3), best.items()):
        if i is None:
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        core = cv2.erode(comp, kernel)
        markers[(core if core.any() else comp) > 0] = label
        seeds[side] = label
    ws = cv2.watershed(cv2.cvtColor(coarse, cv2.COLOR_GRAY2BGR), markers.copy())
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, w // 18) | 1,) * 2)
    fields = {}
    for side, label in seeds.items():
        m = np.where(ws == label, 255, 0).astype(np.uint8)
        m = _largest_component(cv2.morphologyEx(m, cv2.MORPH_OPEN, opening))  # cuts thin slivers along the edges
        m = _fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2))
        if side == "left":
            m[:, mid:] = 0
        else:
            m[:, :mid] = 0
        if m.any():
            fields[side] = m
    mask = np.zeros((h, w), np.uint8)
    for m in fields.values():
        mask |= m
    return {"mask": mask, "fields": fields, "midline": mid, "markers": markers, "coarse": coarse,
            "rejected": rejected, "quality": _plausibility(fields, h, w)}


def _plausibility(fields: dict[str, np.ndarray], h: int, w: int) -> dict:
    """Anatomical sanity checks. They flag implausible masks; passing them does not prove the mask is correct."""
    areas = {s: float((m > 0).mean() * 100) for s, m in fields.items()}
    total = sum(areas.values())
    heights = {s: float((np.ptp(np.where((m > 0).any(axis=1))[0]) + 1) / h * 100) for s, m in fields.items()}
    edge = max((float(((m[:, :2] > 0).any(axis=1) | (m[:, -2:] > 0).any(axis=1)).mean() * 100)
                for m in fields.values()), default=0.0)
    ratio = min(areas.values()) / max(areas.values()) if len(areas) == 2 and max(areas.values()) > 0 else 0.0
    checks = [
        {"check": "two lung fields (left and right)", "passed": len(fields) == 2, "detail": f"{len(fields)} found"},
        {"check": "total area 15-55% of the image", "passed": 15 <= total <= 55, "detail": f"{total:.1f}%"},
        {"check": "left/right area ratio >= 0.4", "passed": ratio >= 0.4, "detail": f"{ratio:.2f}"},
        {"check": "each field spans >= 35% of the image height",
         "passed": len(heights) == 2 and min(heights.values()) >= 35,
         "detail": ", ".join(f"{s} {v:.0f}%" for s, v in heights.items()) or "-"},
        {"check": "field touches the left/right image edge on < 50% of rows", "passed": edge < 50,
         "detail": f"{edge:.0f}% of rows"},
    ]
    return {"plausible": all(c["passed"] for c in checks), "checks": checks,
            "field_area_pct": areas, "field_height_pct": heights}


def lung_mask(img: np.ndarray) -> tuple[np.ndarray, list[str]]:
    return lung_fields(img)["mask"], SEGMENTATION_STEPS


SEGMENTATION_STEPS = [
    "CLAHE + Gaussian smoothing, then a coarse blur (sigma = width/45) so the ribs fade and the lung fields stay dark",
    "Locate the mediastinum/spine as the brightest vertical band in the central 30% of the image",
    "Inverse Otsu threshold on the coarse image; cut along the midline so the two lungs separate",
    "Reject candidates that are tiny, span the body, are empty air, hug the image edge, sit above the shoulders "
    "or below the diaphragm; keep the largest remaining candidate on each side",
    "Marker-controlled watershed: eroded candidates are lung seeds; bright tissue, the mediastinum and outside "
    "air are background",
    "Opening removes thin slivers, then closing, hole filling and clipping at the midline",
    "Plausibility checks: two fields, total area, left/right balance, height and edge contact",
]


def segment(img: np.ndarray) -> dict:
    r = lung_fields(img)
    mask, fields, quality = r["mask"], r["fields"], r["quality"]
    colours = {"left": np.array([16, 185, 129]), "right": np.array([59, 130, 246])}
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for side, m in fields.items():
        overlay[m > 0] = (0.6 * overlay[m > 0] + 0.4 * colours[side]).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in colours[side]), 2)
    markers_vis = cv2.cvtColor(r["coarse"], cv2.COLOR_GRAY2RGB)
    markers_vis[r["markers"] == 1] = (0.5 * markers_vis[r["markers"] == 1] + [120, 30, 30]).clip(0, 255)
    for label, side in ((2, "left"), (3, "right")):
        markers_vis[r["markers"] == label] = colours[side]
    cv2.line(markers_vis, (r["midline"], 0), (r["midline"], img.shape[0] - 1), (250, 204, 21), 1)
    area_pct = float((mask > 0).mean() * 100)
    return to_jsonable({
        "panels": [
            _panel("Original", img),
            _panel("Coarse scale (ribs smoothed away)", r["coarse"]),
            _panel("Watershed markers (red = background, green/blue = lung seeds, yellow = midline)", markers_vis),
            _panel("Lung-field mask", mask, area_pct=area_pct),
            _panel("Overlay (green = image-left field, blue = image-right field)", overlay),
        ],
        "method": "Anatomically constrained marker-controlled watershed (classical, no learned model)",
        "lung_area_pct": area_pct,
        "components": len(fields),
        "field_area_pct": quality["field_area_pct"],
        "quality": {"plausible": quality["plausible"], "checks": quality["checks"]},
        "rejected_candidates": r["rejected"],
        "steps": SEGMENTATION_STEPS,
        "warning": "Classical image processing, not a trained or validated lung segmentation model. There is no "
                   "ground truth for these images, so the plausibility checks can flag obvious failures but cannot "
                   "measure accuracy. Consolidated (white) lung in pneumonia is often missed.",
    })


def compress(img: np.ndarray, qualities: tuple[int, ...] = (95, 75, 50, 25, 10)) -> dict:
    raw_bytes = img.size
    png_buf = io.BytesIO()
    Image.fromarray(img).save(png_buf, format="PNG", optimize=True)
    rows = [{"format": "Raw 8-bit", "bytes": raw_bytes, "ratio": 1.0, "psnr": None, "ssim": 1.0}]
    rows.append({"format": "PNG (lossless)", "bytes": len(png_buf.getvalue()),
                 "ratio": raw_bytes / len(png_buf.getvalue()), "psnr": None, "ssim": 1.0})
    panels = [_panel("Original", img)]
    for q in qualities:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        rows.append({"format": f"JPEG q={q}", "bytes": int(enc.size), "ratio": raw_bytes / enc.size,
                     "psnr": psnr(img, dec), "ssim": ssim(img, dec)})
        panels.append(_panel(f"JPEG quality {q}", dec, bytes=int(enc.size), psnr=psnr(img, dec)))
    return to_jsonable({"panels": panels, "table": rows,
                        "note": "Compression ratio = raw size / encoded size. JPEG is lossy (DCT + quantisation); "
                                "PNG is lossless (DEFLATE)."})


# ------------------------------------------------------------------ T5: SIFT, RANSAC, CNN features


def sift_keypoints(img: np.ndarray, max_draw: int = 300) -> dict:
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(img, None)
    strongest = sorted(kps, key=lambda k: -k.response)[:max_draw]
    drawn = cv2.drawKeypoints(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), strongest, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    sizes = np.array([k.size for k in kps]) if kps else np.array([0.0])
    angles = np.array([k.angle for k in kps]) if kps else np.array([0.0])
    size_hist, size_edges = np.histogram(sizes, bins=12)
    angle_hist, angle_edges = np.histogram(angles, bins=12, range=(0, 360))
    return to_jsonable({
        "panels": [_panel("Original", img), _panel(f"SIFT keypoints (strongest {len(strongest)})",
                                                   cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))],
        "keypoints": len(kps),
        "descriptor_shape": list(desc.shape) if desc is not None else [0, 128],
        "size_histogram": [{"bin": f"{size_edges[i]:.1f}", "count": int(c)} for i, c in enumerate(size_hist)],
        "orientation_histogram": [{"bin": f"{angle_edges[i]:.0f}", "count": int(c)} for i, c in enumerate(angle_hist)],
        "explanation": "Each circle is a keypoint: radius = scale at which it was detected, line = dominant "
                       "gradient orientation. Each keypoint has a 128-D descriptor (4x4 cells x 8 orientation bins).",
    })


def synthetic_transform(img: np.ndarray, angle: float, scale: float, tx: float, ty: float,
                        perspective: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    hmat = np.vstack([rot, [0, 0, 1]])
    hmat[0, 2] += tx * w
    hmat[1, 2] += ty * h
    hmat[2, 0] = perspective / w
    warped = cv2.warpPerspective(img, hmat, (w, h), borderMode=cv2.BORDER_CONSTANT)
    return warped, hmat


def _corner_error(h_est: np.ndarray | None, h_true: np.ndarray, shape: tuple[int, int]) -> float | None:
    if h_est is None:
        return None
    h, w = shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(corners, h_est)
    b = cv2.perspectiveTransform(corners, h_true)
    return float(np.linalg.norm(a - b, axis=2).mean())


def match_ransac(img: np.ndarray, other: np.ndarray | None = None, angle: float = 15.0, scale: float = 0.9,
                 tx: float = 0.05, ty: float = 0.03, perspective: float = 0.05, ratio: float = 0.75,
                 reproj_threshold: float = 5.0) -> dict:
    h_true = None
    if other is None:
        other, h_true = synthetic_transform(img, angle, scale, tx, ty, perspective)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img, None)
    k2, d2 = sift.detectAndCompute(other, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        raise ValidationFailed("Not enough SIFT keypoints were found to match these images.")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    result = {"keypoints_image_1": len(k1), "keypoints_image_2": len(k2), "candidate_matches": len(pairs),
              "good_matches_after_ratio_test": len(good), "ratio": ratio, "reprojection_threshold": reproj_threshold}
    h_ransac, h_ls, inliers = None, None, np.zeros(len(good), dtype=bool)
    if len(good) >= 4:
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        h_ransac, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj_threshold)
        h_ls, _ = cv2.findHomography(src, dst, 0)
        inliers = mask.ravel().astype(bool) if mask is not None else inliers
    result["ransac_inliers"] = int(inliers.sum())
    result["inlier_ratio"] = float(inliers.mean()) if len(inliers) else 0.0
    result["homography_ransac"] = h_ransac.tolist() if h_ransac is not None else None
    if h_true is not None:
        result["homography_true"] = h_true.tolist()
        result["corner_error_ransac_px"] = _corner_error(h_ransac, h_true, img.shape)
        result["corner_error_least_squares_px"] = _corner_error(h_ls, h_true, img.shape)
    vis = cv2.drawMatches(
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), k1, cv2.cvtColor(other, cv2.COLOR_GRAY2BGR), k2,
        [m for m, ok in zip(good, inliers) if ok][:80], None, matchColor=(0, 200, 0), singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    outliers_vis = cv2.drawMatches(
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), k1, cv2.cvtColor(other, cv2.COLOR_GRAY2BGR), k2,
        [m for m, ok in zip(good, inliers) if not ok][:80], None, matchColor=(0, 0, 230), singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    panels = [_panel("Image 1", img), _panel("Image 2" + (" (synthetic transform)" if h_true is not None else ""),
                                             other),
              _panel(f"RANSAC inlier matches ({int(inliers.sum())})", cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)),
              _panel(f"Rejected outliers ({int((~inliers).sum())})", cv2.cvtColor(outliers_vis, cv2.COLOR_BGR2RGB))]
    if h_ransac is not None:
        aligned = cv2.warpPerspective(img, h_ransac, (other.shape[1], other.shape[0]))
        blend = np.dstack([aligned, other, aligned])
        panels.append(_panel("Alignment check (magenta = warped image 1, green = image 2)", blend))
    result["panels"] = panels
    return to_jsonable(result)


_feature_lock = threading.Lock()


@lru_cache(maxsize=1)
def _feature_extractor():
    import torch
    from torchvision import models

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    torch.set_grad_enabled(False)
    return model


def _to_tensor(img: np.ndarray):
    from app.ml.pneumonia.data import build_transforms

    return build_transforms(224, train=False)(Image.fromarray(img)).unsqueeze(0)


def cnn_features(img: np.ndarray, layer: str = "layer2", channels: int = 12) -> dict:
    import torch

    if layer not in {"layer1", "layer2", "layer3", "layer4"}:
        raise ValidationFailed("layer must be layer1..layer4")
    model = _feature_extractor()
    captured = {}
    hooks = [getattr(model, name).register_forward_hook(lambda m, i, o, n=name: captured.__setitem__(n, o))
             for name in ("layer1", "layer2", "layer3", "layer4")]
    with _feature_lock, torch.no_grad():
        try:
            model(_to_tensor(img))
        finally:
            for h in hooks:
                h.remove()
    fmap = captured[layer][0].numpy()
    energy = fmap.mean(axis=(1, 2))
    top = np.argsort(energy)[::-1][:channels]
    tiles = []
    for c in top:
        m = fmap[c]
        m = ((m - m.min()) / (np.ptp(m) + 1e-8) * 255).astype(np.uint8)
        m = cv2.resize(m, (112, 112), interpolation=cv2.INTER_NEAREST)
        tiles.append({"channel": int(c), "mean_activation": float(energy[c]),
                      "image": png_base64(cv2.applyColorMap(m, cv2.COLORMAP_VIRIDIS)[..., ::-1])})
    pooled = captured["layer4"][0].mean(dim=(1, 2)).numpy()
    shapes = {name: list(captured[name].shape[1:]) for name in captured}
    return to_jsonable({
        "layer": layer,
        "feature_map_shapes": shapes,
        "channels": tiles,
        "embedding_dim": int(pooled.shape[0]),
        "embedding_preview": pooled[:32].tolist(),
        "embedding_stats": {"mean": float(pooled.mean()), "std": float(pooled.std()),
                            "sparsity_pct": float((pooled < 1e-3).mean() * 100)},
        "explanation": "Feature maps from an ImageNet-pretrained ResNet-18 (no X-ray fine-tuning). Early layers "
                       "respond to edges/textures; deeper layers are coarser and more abstract.",
    })


def embed(images: list[np.ndarray], batch: int = 32) -> np.ndarray:
    import torch

    model = _feature_extractor()
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    out = []
    with _feature_lock, torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.cat([_to_tensor(im) for im in images[i:i + batch]])
            out.append(backbone(x).flatten(1).numpy())
    return np.concatenate(out)


def feature_embedding(n_per_class: int = 150, refresh: bool = False) -> dict:
    """ImageNet CNN embeddings of PneumoniaMNIST test images -> PCA map + linear-probe accuracy."""
    cache = settings.models_dir / "imaging" / "feature_embedding.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from app.ml.pneumonia.data import CLASSES, load_dataset

    split = load_dataset("pneumoniamnist").splits["test"]
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.flatnonzero(split.labels == c), size=min(n_per_class,
                                     int((split.labels == c).sum())), replace=False) for c in (0, 1)])
    images = [np.asarray(split.load(int(i))) for i in idx]
    labels = split.labels[idx]
    feats = embed(images)
    coords = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(feats))
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    acc = cross_val_score(probe, feats, labels, cv=5, scoring="accuracy")
    auc = cross_val_score(probe, feats, labels, cv=5, scoring="roc_auc")
    result = to_jsonable({
        "images": len(idx),
        "points": [{"x": float(x), "y": float(y), "label": CLASSES[int(lbl)]} for (x, y), lbl in zip(coords, labels)],
        "linear_probe": {"cv_accuracy_mean": float(acc.mean()), "cv_accuracy_std": float(acc.std()),
                         "cv_roc_auc_mean": float(auc.mean())},
        "method": "512-D global-average-pooled ResNet-18 (ImageNet) features -> standardise -> PCA (2D). "
                  "Linear probe = logistic regression, 5-fold cross-validation on the same images.",
    })
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


# ------------------------------------------------------------------ T6: visualisation


def visualize(img: np.ndarray, colormap: str = "bone", row: float = 0.5) -> dict:
    maps = {"bone": cv2.COLORMAP_BONE, "jet": cv2.COLORMAP_JET, "viridis": cv2.COLORMAP_VIRIDIS,
            "inferno": cv2.COLORMAP_INFERNO, "hot": cv2.COLORMAP_HOT}
    if colormap not in maps:
        raise ValidationFailed(f"colormap must be one of {', '.join(maps)}")
    pseudo = cv2.applyColorMap(img, maps[colormap])[..., ::-1]
    edges = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 40, 120)
    edge_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    edge_overlay[edges > 0] = (250, 204, 21)
    mask, _ = lung_mask(img)
    seg_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_overlay, contours, -1, (16, 185, 129), 2)
    y = int(np.clip(row, 0, 1) * (img.shape[0] - 1))
    profile_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.line(profile_img, (0, y), (img.shape[1] - 1, y), (239, 68, 68), 1)
    step = max(1, img.shape[1] // 200)
    profile = [{"x": int(x), "intensity": int(img[y, x])} for x in range(0, img.shape[1], step)]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    diff = cv2.absdiff(clahe, img)
    diff_vis = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_INFERNO)[..., ::-1]
    return to_jsonable({
        "panels": [
            _panel("Original", img),
            _panel(f"Pseudo-colour ({colormap})", pseudo),
            _panel("Edge overlay (Canny)", edge_overlay),
            _panel("Lung-field contour overlay", seg_overlay),
            _panel(f"Intensity profile line (row {y})", profile_img),
            _panel("Difference map |CLAHE - original|", diff_vis),
        ],
        "histogram": histogram(img),
        "profile": profile,
        "profile_row": y,
        "comparison": {"original": png_base64(img), "enhanced": png_base64(clahe), "label": "CLAHE"},
    })


# ------------------------------------------------------------------ sample images


def sample_dir() -> Path:
    return settings.sample_dir / "xray"


def list_samples() -> list[dict]:
    folder = sample_dir()
    if not folder.exists():
        return []
    return [{"name": p.name, "label": p.name.split("_")[0].upper(), "url": f"/api/imaging/samples/{p.name}"}
            for p in sorted(folder.glob("*.png"))]


def load_sample(name: str) -> np.ndarray:
    path = (sample_dir() / name).resolve()
    if path.parent != sample_dir().resolve() or not path.exists():
        raise NotFound(f"Sample image '{name}' not found.")
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
