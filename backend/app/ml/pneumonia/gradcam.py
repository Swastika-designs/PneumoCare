"""Grad-CAM (Selvaraju et al., 2017) for ResNet models.

For the chosen class c, the gradient of the class score y_c with respect to the
feature maps A_k of the last convolutional block is average-pooled to give one weight
per channel (alpha_k). The heat-map is ReLU(sum_k alpha_k * A_k), upsampled to the
input size. Regions with high values increased the score for class c.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        return cam / cam.max() if cam.max() > 0 else cam
