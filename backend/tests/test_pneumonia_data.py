"""Dataset loading, patient-level splitting and leakage checks (using tiny fake folders)."""
import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.ml.pneumonia import data
from app.ml.pneumonia.data import DatasetNotFoundError, SplitData, XrayDataset, average_hash, leakage_report


def _img(seed: int, size: int = 48) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size), dtype=np.uint8), mode="L")


@pytest.fixture
def fake_kaggle(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    root = tmp_path / "raw" / "chest_xray"
    seed = 0
    for split in ("train", "val", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            folder = root / split / cls
            folder.mkdir(parents=True)
            for patient in range(6 if split == "train" else 2):
                for view in range(2):  # two images per patient -> must stay in the same split
                    seed += 1
                    name = (f"IM-{split}{patient:03d}-000{view}.jpeg" if cls == "NORMAL"
                            else f"person{split}{patient}_bacteria_{view}.jpeg")
                    _img(seed).save(folder / name.replace("person", "person9" if split == "test" else "person1"))
    return root


def test_no_dataset_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    assert data.available_sources() == []
    with pytest.raises(DatasetNotFoundError):
        data.load_dataset()


def test_patient_id_parsing():
    from pathlib import Path

    assert data._patient_id(Path("person1234_bacteria_3150.jpeg")) == "person1234"
    assert data._patient_id(Path("IM-0115-0001.jpeg")) == "IM-0115"
    assert data._patient_id(Path("NORMAL2-IM-1427-0001.jpeg")) == "NORMAL2-IM-1427"


def test_kaggle_split_has_no_patient_overlap(fake_kaggle):
    ds = data.load_dataset("kaggle", val_fraction=0.25)
    assert ds.source == "kaggle"
    assert set(ds.splits["train"].groups).isdisjoint(ds.splits["val"].groups)
    # 2 classes x (6 train + 2 val + 2 test patients) x 2 views
    assert sum(len(s) for s in ds.splits.values()) == 40
    assert len(ds.splits["test"]) == 8
    report = leakage_report(ds)
    assert report["patient_overlap_between_splits"]["train-val"] == 0


def test_leakage_report_finds_exact_duplicates():
    imgs = [np.asarray(_img(i)) for i in range(6)]
    train = SplitData(np.stack(imgs[:4]), np.array([0, 1, 0, 1]))
    val = SplitData(np.stack([imgs[1], imgs[4]]), np.array([1, 0]))  # imgs[1] duplicated
    test = SplitData(np.stack([imgs[5]]), np.array([1]))
    report = leakage_report(XrayDataset("pneumoniamnist", settings.data_dir, {"train": train, "val": val,
                                                                              "test": test}))
    assert report["confirmed_near_duplicates"]["train-val"] == 1
    assert report["confirmed_near_duplicates"]["train-test"] == 0
    assert report["duplicate_pairs"][0]["index_b"] == 0


def test_average_hash_stable_under_resize():
    img = _img(3, 128)
    assert average_hash(img) == average_hash(img.resize((256, 256)))


def test_transforms_produce_normalised_tensor():
    t = data.build_transforms(64, train=False)(_img(1, 100))
    assert tuple(t.shape) == (3, 64, 64)
    assert abs(float(t.mean())) < 3
