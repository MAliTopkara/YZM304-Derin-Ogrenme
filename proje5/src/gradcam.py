"""Minimal Grad-CAM implementation for 2D CNN classifiers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Standard Grad-CAM (Selvaraju et al., 2017) for a single target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.zero_grad()
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        pred_class = int(logits.argmax(dim=1).item()) if target_class is None else int(target_class)
        score = logits[0, pred_class]
        score.backward(retain_graph=False)

        assert self.activations is not None and self.gradients is not None
        # Weights: GAP over gradients -> one scalar per channel.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np, pred_class, float(probs[0, pred_class].item())


def load_and_preprocess(path: Path, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    img_arr = np.asarray(img).astype(np.float32) / 255.0
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tf(img).unsqueeze(0)
    return x, img_arr


def overlay_cam(img_rgb01: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[..., :3]  # drop alpha
    overlay = (1 - alpha) * img_rgb01 + alpha * heat
    return np.clip(overlay, 0.0, 1.0)


def save_gradcam_panel(
    image_paths: list[Path],
    images: list[np.ndarray],
    cams: list[np.ndarray],
    preds: list[int],
    probs: list[float],
    trues: list[int],
    out_path: Path,
    title: str = "Grad-CAM",
) -> None:
    n = len(image_paths)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
    if n == 1:
        axes = axes[:, None]
    for i in range(n):
        axes[0, i].imshow(images[i])
        tag = "TP/TN" if preds[i] == trues[i] else "ERR"
        axes[0, i].set_title(
            f"{image_paths[i].name}\n"
            f"true={CLASS_NAMES[trues[i]]} | pred={CLASS_NAMES[preds[i]]} "
            f"({probs[i]:.2f}) [{tag}]",
            fontsize=8,
        )
        axes[0, i].axis("off")
        axes[1, i].imshow(overlay_cam(images[i], cams[i]))
        axes[1, i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
