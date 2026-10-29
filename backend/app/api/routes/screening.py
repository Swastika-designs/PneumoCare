"""Pneumonia screening (prediction + Grad-CAM) and model performance endpoints."""
from __future__ import annotations

import json
import re
import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import ModelUnavailable, NotFound
from app.core.utils import color_saturation, png_base64, validate_image_bytes
from app.db.database import get_db
from app.ml.pneumonia.inference import get_predictor, overlay_heatmap
from app.ml.pneumonia.model import checkpoint_dir, default_checkpoint_path
from app.models.entities import Prediction
from app.schemas.api import GradCamPayload, PredictionResponse
from app.services import history

router = APIRouter(prefix="/api", tags=["screening"])
DISCLAIMER = ("Educational prototype trained on a public pediatric chest X-ray dataset. It is not a medical device "
              "and must not be used for diagnosis.")


@router.get("/screening/status")
def screening_status() -> dict:
    path = default_checkpoint_path()
    try:
        predictor = get_predictor()
    except ModelUnavailable as exc:
        return {"model_available": False, "checkpoint": str(path), "message": exc.message}
    return {"model_available": True, "model_name": predictor.name, "threshold": predictor.threshold,
            "img_size": predictor.meta["img_size"], "trained_at": predictor.meta.get("trained_at"),
            "data_source": predictor.meta.get("data_source")}


@router.post("/screening/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...), gradcam: bool = Query(True),
            context: str | None = Query(None, pattern="^T6$",
                                        description="Set to T6 when called from the T6 visualisation workflow"),
            db: Session = Depends(get_db)):
    data = file.file.read()
    image = validate_image_bytes(data, file.filename or "upload")
    predictor = get_predictor()
    warnings = []
    if color_saturation(image) > 0.15:
        warnings.append("This image looks like a colour photograph. Chest X-rays are grayscale, so the prediction "
                        "is probably meaningless for this input.")
    w, h = image.size
    if max(w, h) / min(w, h) > 2:
        warnings.append(f"Unusual aspect ratio ({w}x{h}); the image is resized to a square, which distorts anatomy.")
    result = predictor.predict(image, with_gradcam=gradcam)

    folder = settings.uploads_dir / "xrays"
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename or "upload.png")[-80:]
    stored = folder / f"{uuid.uuid4().hex[:12]}_{safe}"
    stored.write_bytes(data)
    record = Prediction(filename=file.filename or "upload", stored_path=str(stored),
                        predicted_label=result["predicted_label"],
                        probability_pneumonia=result["probability_pneumonia"], threshold=result["threshold"],
                        model_name=result["model_name"])
    db.add(record)
    db.commit()
    history.record(db, "screening", f"Pneumonia screening: {file.filename}",
                   {"prediction": result["predicted_label"], "p_pneumonia": result["probability_pneumonia"]},
                   tutorial=context, dataset="upload")

    gray = image.convert("L")
    gray.thumbnail((512, 512))
    gray_arr = np.asarray(gray)
    cam_payload = None
    if result["gradcam"] is not None:
        cam = result["gradcam"]["heatmap"]
        import cv2

        heat = cv2.applyColorMap((cv2.resize(cam, gray_arr.shape[::-1]) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_payload = GradCamPayload(overlay=png_base64(overlay_heatmap(gray_arr, cam)),
                                     heatmap=png_base64(heat[..., ::-1]),
                                     target_class=result["gradcam"]["target_class"])
    return PredictionResponse(
        id=record.id, filename=record.filename, predicted_label=result["predicted_label"],
        probability_pneumonia=result["probability_pneumonia"], probabilities=result["probabilities"],
        threshold=result["threshold"], model_name=result["model_name"], image=png_base64(gray_arr),
        gradcam=cam_payload, warnings=warnings, disclaimer=DISCLAIMER)


@router.get("/screening/predictions")
def predictions(limit: int = 20, db: Session = Depends(get_db)) -> list[dict]:
    rows = db.execute(select(Prediction).order_by(Prediction.created_at.desc()).limit(min(limit, 200))).scalars()
    return [{"id": r.id, "filename": r.filename, "predicted_label": r.predicted_label,
             "probability_pneumonia": r.probability_pneumonia, "threshold": r.threshold, "model_name": r.model_name,
             "created_at": r.created_at.isoformat()} for r in rows]


def _read_json(name: str) -> dict:
    path = checkpoint_dir() / name
    if not path.exists():
        raise ModelUnavailable(f"{name} not found. Train and evaluate the model first "
                               "(scripts/train_pneumonia.py, scripts/evaluate_pneumonia.py).")
    return json.loads(path.read_text())


@router.get("/models/pneumonia/evaluation")
def evaluation() -> dict:
    return _read_json("evaluation.json")


@router.get("/models/pneumonia/training")
def training_history() -> dict:
    return _read_json("training_history.json")


@router.get("/models/pneumonia/dataset")
def dataset_inspection() -> dict:
    return _read_json("dataset_inspection.json")


PLOTS = {"training_curves.png", "confusion_matrix.png", "roc_curve.png"}


@router.get("/models/pneumonia/plots/{name}")
def plot(name: str):
    if name not in PLOTS or not (checkpoint_dir() / name).exists():
        raise NotFound(f"Plot '{name}' is not available.")
    return FileResponse(checkpoint_dir() / name, media_type="image/png")
