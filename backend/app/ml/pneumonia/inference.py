"""Load the trained checkpoint once and serve predictions + Grad-CAM overlays."""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.errors import ModelUnavailable
from app.ml.pneumonia.data import build_transforms
from app.ml.pneumonia.gradcam import GradCAM
from app.ml.pneumonia.model import IncompleteCheckpointError, default_checkpoint_path, load_checkpoint


class PneumoniaPredictor:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        self.model, self.meta = load_checkpoint(checkpoint)
        self.classes = self.meta["classes"]
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.transform = build_transforms(self.meta["img_size"], train=False)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"{self.meta.get('arch', 'resnet')} ({self.checkpoint.name})"

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("L")).unsqueeze(0)

    def predict(self, image: Image.Image, with_gradcam: bool = True) -> dict:
        x = self._tensor(image)
        with self._lock, torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].numpy()
        p_pneumonia = float(probs[1])
        label_idx = int(p_pneumonia >= self.threshold)
        result = {
            "predicted_label": self.classes[label_idx],
            "probability_pneumonia": p_pneumonia,
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "threshold": self.threshold,
            "model_name": self.name,
            "gradcam": None,
        }
        if with_gradcam:
            with self._lock:
                cam_engine = GradCAM(self.model, self.model.layer4)
                try:
                    cam = cam_engine(x, class_idx=1)
                finally:
                    cam_engine.remove()
            result["gradcam"] = {"heatmap": cam, "target_class": self.classes[1]}
        return result


def overlay_heatmap(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0,1] heat-map (any size) over a grayscale image; returns RGB uint8."""
    h, w = gray.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)


_predictor: PneumoniaPredictor | None = None
_predictor_lock = threading.Lock()


def get_predictor(checkpoint: Path | None = None) -> PneumoniaPredictor:
    """Return the cached predictor, reloading if the checkpoint file changed."""
    global _predictor
    path = checkpoint or default_checkpoint_path()
    if not path.exists():
        raise ModelUnavailable(
            "No trained pneumonia model was found. Train one with `python scripts/train_pneumonia.py` "
            "and evaluate it with `python scripts/evaluate_pneumonia.py`.",
            details={"expected_checkpoint": str(path)},
        )
    with _predictor_lock:
        if _predictor is None or _predictor.checkpoint != path or _stale(_predictor, path):
            try:
                _predictor = PneumoniaPredictor(path)
            except IncompleteCheckpointError as exc:
                _predictor = None
                raise ModelUnavailable(str(exc), details={"checkpoint": str(path)}) from exc
            _predictor.mtime = path.stat().st_mtime
        return _predictor


def _stale(predictor: PneumoniaPredictor, path: Path) -> bool:
    return getattr(predictor, "mtime", None) != path.stat().st_mtime


def reset_predictor() -> None:
    global _predictor
    _predictor = None
