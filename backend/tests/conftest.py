"""Shared fixtures.

Tests run against a temporary SQLite database and a temporary uploads folder, so they never
touch the application's real database. Tests that need a downloaded dataset are skipped
(not failed) when it is missing.
"""
from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.db import database

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _has(path) -> bool:
    return path.exists()


requires_synthea = pytest.mark.skipif(not _has(settings.raw_dir / "synthea" / "patients.csv"),
                                      reason="Synthea dataset not downloaded")
requires_nhanes = pytest.mark.skipif(not _has(settings.raw_dir / "nhanes" / "BMX_J.xpt"),
                                     reason="NHANES dataset not downloaded")
requires_corpus = pytest.mark.skipif(not _has(settings.raw_dir / "medical_abstracts" / "medical_tc_train.csv"),
                                     reason="Medical abstracts corpus not downloaded")
requires_samples = pytest.mark.skipif(not any((settings.sample_dir / "xray").glob("*.png"))
                                      if (settings.sample_dir / "xray").exists() else True,
                                      reason="Sample X-rays not exported (scripts/export_sample_images.py)")


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pneumocare")
    database.configure(f"sqlite:///{(tmp / 'test.db').as_posix()}")
    original_uploads = settings.uploads_dir
    settings.uploads_dir = tmp / "uploads"
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    settings.uploads_dir = original_uploads


@pytest.fixture
def phantom() -> np.ndarray:
    """A synthetic chest-like phantom: bright body, two dark 'lungs', texture and a gradient."""
    rng = np.random.default_rng(0)
    img = np.full((160, 160), 170, dtype=np.float32)
    yy, xx = np.mgrid[:160, :160]
    for cx in (50, 110):
        img[((xx - cx) / 26) ** 2 + ((yy - 85) / 50) ** 2 < 1] = 60
    img += 20 * np.sin(xx / 6.0) * np.cos(yy / 9.0)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def png_bytes(arr: np.ndarray, mode: str | None = None) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()
