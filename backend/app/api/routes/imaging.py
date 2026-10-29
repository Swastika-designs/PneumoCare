"""T4-T6: biomedical image laboratory.

Every POST endpoint accepts either an uploaded ``file`` or the ``sample`` name of a bundled
X-ray (see GET /api/imaging/samples). Tunable parameters are query parameters.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.errors import ValidationFailed
from app.core.utils import to_gray_array, validate_image_bytes
from app.db.database import get_db
from app.services import history, imaging

router = APIRouter(prefix="/api/imaging", tags=["imaging"])


def _image(file: UploadFile | None, sample: str | None) -> tuple[np.ndarray, str]:
    if file is not None and file.filename:
        return to_gray_array(validate_image_bytes(file.file.read(), file.filename)), file.filename
    if sample:
        return imaging.load_sample(sample), sample
    raise ValidationFailed("Provide an image file or choose a sample image.")


def _log(db: Session, tutorial: str, title: str, name: str, summary: dict) -> None:
    history.record(db, "imaging", f"{title}: {name}", summary, tutorial=tutorial, dataset=name)


@router.get("/samples")
def samples() -> list[dict]:
    return imaging.list_samples()


@router.get("/samples/{name}")
def sample_file(name: str):
    imaging.load_sample(name)  # validates the name
    return FileResponse(imaging.sample_dir() / name, media_type="image/png")


@router.post("/enhance")
def enhance(file: UploadFile | None = File(None), sample: str | None = Form(None),
            clip_limit: float = Query(2.0, ge=0.5, le=10), gamma: float = Query(0.8),
            db: Session = Depends(get_db)) -> dict:
    img, name = _image(file, sample)
    result = imaging.enhance(img, clip_limit, gamma)
    _log(db, "T4", "Enhancement", name, {"clip_limit": clip_limit, "gamma": gamma})
    return result


@router.post("/restore")
def restore(file: UploadFile | None = File(None), sample: str | None = Form(None), noise: str = "gaussian",
            level: float = Query(20.0, ge=1, le=60), db: Session = Depends(get_db)) -> dict:
    img, name = _image(file, sample)
    result = imaging.restore(img, noise, level)
    _log(db, "T4", "Restoration", name, {"noise": noise, "level": level, "best": result["best_by_psnr"]})
    return result


@router.post("/filters")
def filters(file: UploadFile | None = File(None), sample: str | None = Form(None),
            low: int = Query(50, ge=0, le=255), high: int = Query(150, ge=0, le=255)) -> dict:
    img, _ = _image(file, sample)
    return imaging.filters(img, low, high)


@router.post("/threshold")
def threshold(file: UploadFile | None = File(None), sample: str | None = Form(None),
              value: int = Query(128, ge=0, le=255), block_size: int = Query(31, ge=3, le=151)) -> dict:
    img, _ = _image(file, sample)
    return imaging.threshold(img, value, block_size)


@router.post("/segment")
def segment(file: UploadFile | None = File(None), sample: str | None = Form(None),
            db: Session = Depends(get_db)) -> dict:
    img, name = _image(file, sample)
    result = imaging.segment(img)
    _log(db, "T4", "Segmentation", name,
         {"lung_area_pct": result["lung_area_pct"], "plausible": result["quality"]["plausible"]})
    return result


@router.post("/compress")
def compress(file: UploadFile | None = File(None), sample: str | None = Form(None)) -> dict:
    img, _ = _image(file, sample)
    return imaging.compress(img)


@router.post("/sift")
def sift(file: UploadFile | None = File(None), sample: str | None = Form(None)) -> dict:
    img, _ = _image(file, sample)
    return imaging.sift_keypoints(img)


@router.post("/match")
def match(file: UploadFile | None = File(None), sample: str | None = Form(None),
          file2: UploadFile | None = File(None), sample2: str | None = Form(None),
          angle: float = Query(15, ge=-180, le=180), scale: float = Query(0.9, ge=0.3, le=3),
          perspective: float = Query(0.05, ge=0, le=0.5), ratio: float = Query(0.75, ge=0.3, le=0.95),
          reproj_threshold: float = Query(5.0, ge=0.5, le=50), db: Session = Depends(get_db)) -> dict:
    img, name = _image(file, sample)
    other = None
    if (file2 is not None and file2.filename) or sample2:
        other, _ = _image(file2, sample2)
        if other.shape != img.shape:
            import cv2

            other = cv2.resize(other, img.shape[::-1])
    result = imaging.match_ransac(img, other, angle=angle, scale=scale, perspective=perspective, ratio=ratio,
                                  reproj_threshold=reproj_threshold)
    _log(db, "T5", "SIFT + RANSAC", name, {k: result.get(k) for k in ("good_matches_after_ratio_test",
                                                                     "ransac_inliers", "corner_error_ransac_px")})
    return result


@router.post("/cnn-features")
def cnn_features(file: UploadFile | None = File(None), sample: str | None = Form(None), layer: str = "layer2",
                 db: Session = Depends(get_db)) -> dict:
    img, name = _image(file, sample)
    result = imaging.cnn_features(img, layer)
    _log(db, "T5", "CNN features", name, {"layer": layer})
    return result


@router.get("/feature-embedding")
def feature_embedding(refresh: bool = False) -> dict:
    return imaging.feature_embedding(refresh=refresh)


@router.post("/visualize")
def visualize(file: UploadFile | None = File(None), sample: str | None = Form(None), colormap: str = "bone",
              row: float = Query(0.5, ge=0, le=1), db: Session = Depends(get_db)) -> dict:
    img, name = _image(file, sample)
    result = imaging.visualize(img, colormap, row)
    _log(db, "T6", "Visualisation", name, {"colormap": colormap})
    return result
