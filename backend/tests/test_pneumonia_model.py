"""Model building, checkpoints, metrics, missing-checkpoint behaviour and prediction schema.

A small *untrained* checkpoint is written to a temporary folder so the prediction endpoint
and Grad-CAM can be tested without the real (large, slow-to-train) model.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from app.core.config import settings
from app.core.errors import ModelUnavailable
from app.ml.pneumonia import inference
from app.ml.pneumonia.gradcam import GradCAM
from app.ml.pneumonia.metrics import binary_metrics, youden_threshold
from app.ml.pneumonia.model import IncompleteCheckpointError, build_model, load_checkpoint, save_checkpoint
from app.schemas.api import PredictionResponse
from tests.conftest import png_bytes

META = {"arch": "resnet18", "classes": ["NORMAL", "PNEUMONIA"], "img_size": 64, "threshold": 0.4}


@pytest.fixture
def tmp_models(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    inference.reset_predictor()
    yield tmp_path
    inference.reset_predictor()


def test_build_model_outputs_two_logits():
    model = build_model("resnet18", pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 2)


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError):
        build_model("vgg99", pretrained=False)


def test_checkpoint_roundtrip(tmp_path):
    model = build_model("resnet18", pretrained=False)
    save_checkpoint(tmp_path / "m.pt", model, META)
    loaded, meta = load_checkpoint(tmp_path / "m.pt")
    assert meta["threshold"] == 0.4
    key = "conv1.weight"
    assert torch.equal(model.state_dict()[key], loaded.state_dict()[key])


def test_incomplete_checkpoint_detected(tmp_path):
    save_checkpoint(tmp_path / "partial.pt", build_model("resnet18", pretrained=False), {"arch": "resnet18"})
    with pytest.raises(IncompleteCheckpointError):
        load_checkpoint(tmp_path / "partial.pt")


def test_metrics_known_values():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.2, 0.9, 0.8, 0.3, 0.7, 0.95])
    m = binary_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 4}
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(4 / 5)
    assert m["recall"] == pytest.approx(4 / 5)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(0.8)
    assert 0 <= youden_threshold(y, p) <= 1


def test_missing_checkpoint_raises_model_unavailable(tmp_models):
    with pytest.raises(ModelUnavailable):
        inference.get_predictor()


def test_predict_endpoint_409_without_model(client, tmp_models, phantom):
    r = client.post("/api/screening/predict", files={"file": ("x.png", png_bytes(phantom), "image/png")})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "model_unavailable"
    status = client.get("/api/screening/status").json()
    assert status["model_available"] is False


def test_gradcam_heatmap_shape_and_range():
    model = build_model("resnet18", pretrained=False).eval()
    cam = GradCAM(model, model.layer4)
    try:
        heat = cam(torch.randn(1, 3, 64, 64), class_idx=1)
    finally:
        cam.remove()
    assert heat.shape == (64, 64)
    assert heat.min() >= 0 and heat.max() <= 1


def test_prediction_response_schema_with_untrained_checkpoint(client, tmp_models, phantom):
    torch.manual_seed(0)
    save_checkpoint(tmp_models / "pneumonia" / "pneumonia_resnet18.pt",
                    build_model("resnet18", pretrained=False), META)
    r = client.post("/api/screening/predict", files={"file": ("chest.png", png_bytes(phantom), "image/png")})
    assert r.status_code == 200, r.text
    body = PredictionResponse.model_validate(r.json())
    assert body.predicted_label in {"NORMAL", "PNEUMONIA"}
    assert body.probabilities["NORMAL"] + body.probabilities["PNEUMONIA"] == pytest.approx(1, abs=1e-5)
    assert (body.probability_pneumonia >= body.threshold) == (body.predicted_label == "PNEUMONIA")
    assert body.gradcam is not None and body.gradcam.overlay.startswith("data:image/png;base64,")
    assert "not a medical device" in body.disclaimer
    assert client.get("/api/screening/predictions").json()[0]["filename"] == "chest.png"


def test_color_photo_triggers_warning(client, tmp_models):
    save_checkpoint(tmp_models / "pneumonia" / "pneumonia_resnet18.pt",
                    build_model("resnet18", pretrained=False), META)
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    r = client.post("/api/screening/predict?gradcam=false",
                    files={"file": ("photo.png", png_bytes(rgb, "RGB"), "image/png")})
    assert r.status_code == 200
    assert any("colour photograph" in w for w in r.json()["warnings"])
    assert r.json()["gradcam"] is None


def test_predictor_preprocessing_accepts_any_size(tmp_models):
    save_checkpoint(tmp_models / "pneumonia" / "pneumonia_resnet18.pt",
                    build_model("resnet18", pretrained=False), META)
    predictor = inference.get_predictor()
    result = predictor.predict(Image.new("L", (300, 200), 128), with_gradcam=False)
    assert 0 <= result["probability_pneumonia"] <= 1


def test_predictions_count_as_t6_only_from_visualisation(client, tmp_models, phantom):
    save_checkpoint(tmp_models / "pneumonia" / "pneumonia_resnet18.pt",
                    build_model("resnet18", pretrained=False), META)
    before = {t["id"]: t["runs"] for t in client.get("/api/tutorials").json()}["T6"]
    files = {"file": ("plain.png", png_bytes(phantom), "image/png")}
    assert client.post("/api/screening/predict?gradcam=false", files=files).status_code == 200
    assert {t["id"]: t["runs"] for t in client.get("/api/tutorials").json()}["T6"] == before
    files = {"file": ("viz.png", png_bytes(phantom), "image/png")}
    assert client.post("/api/screening/predict?gradcam=false&context=T6", files=files).status_code == 200
    assert {t["id"]: t["runs"] for t in client.get("/api/tutorials").json()}["T6"] == before + 1
    assert client.post("/api/screening/predict?context=T9", files=files).status_code == 422
