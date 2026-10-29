"""Small shared helpers: JSON-safe conversion and image encoding/validation."""
from __future__ import annotations

import base64
import io
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.core.errors import ValidationFailed

ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "BMP", "TIFF"}


def to_jsonable(value):
    """Recursively convert numpy / pandas values into plain JSON types (NaN -> None)."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NaT or (value is not None and not isinstance(value, str) and pd.api.types.is_scalar(value)
                           and pd.isna(value)):
        return None
    return value


def png_base64(image: np.ndarray | Image.Image) -> str:
    """Encode an image as a ``data:image/png;base64,...`` URL for the frontend."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def validate_image_bytes(data: bytes, filename: str = "upload") -> Image.Image:
    """Validate an uploaded image and return it as a PIL image.

    Checks: non-empty, size limit, decodable, allowed format, minimum resolution.
    """
    if not data:
        raise ValidationFailed("The uploaded file is empty.")
    limit = settings.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise ValidationFailed(f"File is larger than the {settings.max_upload_mb:g} MB limit.")
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValidationFailed(f"'{filename}' is not a readable image (PNG, JPEG, BMP or TIFF).") from exc
    if image.format not in ALLOWED_IMAGE_FORMATS:
        raise ValidationFailed(f"Unsupported image format '{image.format}'. Use PNG, JPEG, BMP or TIFF.")
    if min(image.size) < 32:
        raise ValidationFailed(f"Image is too small ({image.size[0]}x{image.size[1]}); minimum is 32x32 pixels.")
    return image


def color_saturation(image: Image.Image) -> float:
    """Mean HSV saturation in [0, 1]. Chest X-rays are grayscale, so this is ~0 for real X-rays."""
    hsv = np.asarray(image.convert("RGB").convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean() / 255.0)


def to_gray_array(image: Image.Image, max_side: int | None = 512) -> np.ndarray:
    """Convert to a uint8 grayscale numpy array, optionally limiting the longest side."""
    gray = image.convert("L")
    if max_side and max(gray.size) > max_side:
        gray.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(gray, dtype=np.uint8)
