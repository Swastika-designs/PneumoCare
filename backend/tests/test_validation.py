"""File/image validation and JSON helpers."""
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.core.config import settings
from app.core.errors import ValidationFailed
from app.core.utils import color_saturation, to_gray_array, to_jsonable, validate_image_bytes
from tests.conftest import png_bytes


def test_rejects_empty_file():
    with pytest.raises(ValidationFailed, match="empty"):
        validate_image_bytes(b"")


def test_rejects_non_image():
    with pytest.raises(ValidationFailed, match="not a readable image"):
        validate_image_bytes(b"%PDF-1.4 definitely not an image", "report.pdf")


def test_rejects_too_small_image():
    with pytest.raises(ValidationFailed, match="too small"):
        validate_image_bytes(png_bytes(np.zeros((10, 10), dtype=np.uint8)))


def test_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0.001)
    with pytest.raises(ValidationFailed, match="larger than"):
        validate_image_bytes(png_bytes(np.random.default_rng(0).integers(0, 255, (128, 128), dtype=np.uint8)))


def test_accepts_valid_png(phantom):
    img = validate_image_bytes(png_bytes(phantom), "x.png")
    assert img.size == (160, 160)
    assert to_gray_array(img).shape == (160, 160)


def test_color_saturation_distinguishes_grayscale(phantom):
    gray = Image.fromarray(phantom).convert("RGB")
    color = np.zeros((64, 64, 3), dtype=np.uint8)
    color[..., 0] = 255
    assert color_saturation(gray) < 0.01
    assert color_saturation(Image.fromarray(color)) > 0.9


def test_to_jsonable_handles_numpy_and_nan():
    out = to_jsonable({"a": np.int64(3), "b": np.float32(1.5), "c": float("nan"), "d": [np.bool_(True)],
                       "e": pd.Timestamp("2020-01-02"), "f": pd.NaT})
    assert out == {"a": 3, "b": 1.5, "c": None, "d": [True], "e": "2020-01-02T00:00:00", "f": None}
    assert not any(isinstance(v, float) and math.isnan(v) for v in out.values() if v is not None)
