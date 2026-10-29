"""Optional Kaggle dataset: installation, opt-in selection and non-destructive external evaluation."""
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.ml.pneumonia import data, kaggle
from app.ml.pneumonia.evaluate import evaluate
from app.ml.pneumonia.model import build_model, save_checkpoint


def _jpeg(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (40, 40), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def fake_archive(tmp_path):
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as zf:
        seed = 0
        for split in ("train", "val", "test"):
            for cls, prefix in (("NORMAL", "IM-"), ("PNEUMONIA", "person")):
                for i in range(3):
                    seed += 1
                    name = f"{prefix}{split}{i}-0001.jpeg" if cls == "NORMAL" else f"{prefix}{seed}_virus_{i}.jpeg"
                    # The real archive contains each image twice (chest_xray/ and chest_xray/chest_xray/).
                    zf.writestr(f"chest_xray/{split}/{cls}/{name}", _jpeg(seed))
                    zf.writestr(f"chest_xray/chest_xray/{split}/{cls}/{name}", _jpeg(seed))
                    zf.writestr(f"__MACOSX/chest_xray/{split}/{cls}/._{name}", b"junk")
        zf.writestr("chest_xray/train/.DS_Store", b"junk")
    return path


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    return tmp_path


def test_install_archive_deduplicates_and_skips_junk(tmp_data, fake_archive):
    counts = kaggle.install_archive(fake_archive)
    assert counts["train"] == {"NORMAL": 3, "PNEUMONIA": 3}
    assert sum(1 for _ in data.kaggle_root().rglob("*.jpeg")) == 18
    assert json.loads(kaggle.manifest_path().read_text())["counts"]["test"]["PNEUMONIA"] == 3


def test_install_rejects_unrelated_zip(tmp_data, tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("readme.txt", "hello")
    with pytest.raises(ValueError):
        kaggle.install_archive(bad)


def test_missing_credentials_message(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    with pytest.raises(kaggle.KaggleCredentialsMissing, match="--zip"):
        kaggle.find_credentials()


def test_auto_never_switches_to_kaggle(tmp_data, fake_archive):
    kaggle.install_archive(fake_archive)
    (settings.raw_dir / "pneumoniamnist").mkdir(parents=True)
    imgs = np.zeros((2, 32, 32), dtype=np.uint8)
    np.savez(data.medmnist_path(), **{f"{s}_images": imgs for s in ("train", "val", "test")},
             **{f"{s}_labels": np.array([[0], [1]]) for s in ("train", "val", "test")})
    assert data.available_sources() == ["pneumoniamnist", "kaggle"]
    assert data.load_dataset("auto").source == "pneumoniamnist"
    assert data.load_dataset("kaggle").source == "kaggle"


def test_external_evaluation_does_not_overwrite_official(tmp_data, fake_archive):
    kaggle.install_archive(fake_archive)
    folder = settings.models_dir / "pneumonia"
    meta = {"arch": "resnet18", "classes": ["NORMAL", "PNEUMONIA"], "img_size": 64, "threshold": 0.5,
            "data_source": "pneumoniamnist"}
    save_checkpoint(folder / "pneumonia_resnet18.pt", build_model("resnet18", pretrained=False), meta)
    (folder / "evaluation.json").write_text('{"official": true}')
    result = evaluate(source="kaggle", log=lambda *_: None)
    assert result["samples"] == 6 and "External evaluation" in result["note"]
    assert json.loads((folder / "evaluation.json").read_text()) == {"official": True}
    assert (folder / "evaluation_kaggle.json").exists() and (folder / "roc_curve_kaggle.png").exists()
