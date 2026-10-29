"""Transfer-learning classifier and checkpoint helpers."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from app.core.config import settings

SUPPORTED_ARCHS = {"resnet18", "resnet50"}


def checkpoint_dir() -> Path:
    return settings.models_dir / "pneumonia"


def default_checkpoint_path() -> Path:
    return checkpoint_dir() / "pneumonia_resnet18.pt"


def build_model(arch: str = "resnet18", pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """ImageNet-pretrained ResNet with a new 2-class head."""
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {sorted(SUPPORTED_ARCHS)}")
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    else:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    return model


def save_checkpoint(path: Path, model: nn.Module, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


REQUIRED_META = ("arch", "classes", "img_size", "threshold")


class IncompleteCheckpointError(ValueError):
    """The file exists but lacks the metadata written at the end of training."""


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise IncompleteCheckpointError(f"Checkpoint {path.name} is missing metadata {missing}; "
                                        "training may still be running or was interrupted.")
    model = build_model(meta.get("arch", "resnet18"), pretrained=False, num_classes=len(meta["classes"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, meta
