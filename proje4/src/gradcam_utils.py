"""Grad-CAM yardımcıları — 3 model (ResNet50, EfficientNetB0, ViT-Base/16) için.

Public API:
    generate_gradcam(model, model_name, image_tensor, target_class=None) -> np.ndarray
        H×W heatmap, [0, 1] aralığında.
    overlay_heatmap(image_pil, heatmap, alpha=0.45) -> PIL.Image
        Heatmap'i orijinal görselin üstüne overlay eder (jet colormap).
    image_to_input_tensor(pil_image, device) -> torch.Tensor
        webapp/notebook'larda hazır kullanım için: PIL → eval transform → cihaza gönder.

Notlar:
    * `pytorch_grad_cam` (PyPI: `grad-cam`) bağımlılığı gerekir.
    * ViT için reshape_transform: CLS token at, (B, 196, 768) → (B, 768, 14, 14).
      timm `vit_base_patch16_224` için patch grid 14×14, hidden 768.
    * Target layer seçimi mimarinin son anlamlı feature map katmanıdır.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.transforms import build_eval_transform


# ----- target layer seçimi --------------------------------------------------

def get_target_layers(model: torch.nn.Module, model_name: str) -> list[torch.nn.Module]:
    """Mimariye göre Grad-CAM target layer'ı döner.

    ResNet50 → layer4'ün son bloğu (en derin spatial feature map).
    EfficientNetB0 → blocks'un son MBConv stage'i.
    ViT-Base/16 → son transformer block'unun ön norm'u (LayerNorm).
    """
    if model_name == "resnet50":
        return [model.layer4[-1]]
    if model_name == "efficientnet_b0":
        # blocks: ModuleList of stages; son stage'in son bloğunu seç
        last_stage = model.blocks[-1]
        if isinstance(last_stage, torch.nn.Sequential) or hasattr(last_stage, "__len__"):
            return [last_stage[-1]]
        return [last_stage]
    if model_name == "vit_base":
        return [model.blocks[-1].norm1]
    raise ValueError(f"Bilinmeyen model adı: {model_name}")


# ----- ViT için reshape transform -------------------------------------------

def _vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """timm ViT çıktısı (B, N, C) → (B, C, H, W).

    N = 1 (CLS) + height*width. CLS token at, kalanı 2D grid'e reshape.
    """
    # tensor: (B, 197, 768)
    no_cls = tensor[:, 1:, :]                                # (B, 196, 768)
    result = no_cls.reshape(tensor.size(0), height, width, tensor.size(2))  # (B, 14, 14, 768)
    return result.permute(0, 3, 1, 2)                        # (B, 768, 14, 14)


def get_reshape_transform(model_name: str):
    if model_name == "vit_base":
        return _vit_reshape_transform
    return None


# ----- ana API --------------------------------------------------------------

def generate_gradcam(
    model: torch.nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Tek görsel için heatmap üretir.

    Mimariye göre yöntem seçilir:
        * CNN'ler (resnet50, efficientnet_b0): GradCAM (sınıf-özelidir).
        * ViT (vit_base): EigenCAM. Vanilla GradCAM ViT'te softmax saturation
          (conf≈1.0) durumunda gradient'leri sıfırlar; EigenCAM gradient'ten
          bağımsız (SVD tabanlı) olduğu için tutarlı sonuç verir.

    Args:
        model: eval moduna alınmış model (cihaz üzerinde).
        model_name: 'resnet50' | 'efficientnet_b0' | 'vit_base'.
        image_tensor: (1, 3, H, W) — eğitim transform'undan geçmiş tensor.
        target_class: int veya None. CNN'ler için None ise Top-1 kullanılır.
                      ViT için yoksayılır (EigenCAM class-specific değil).

    Returns:
        heatmap: H×W float32, [0, 1] aralığında.
    """
    if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
        raise ValueError(f"image_tensor (1,3,H,W) olmalı, got {tuple(image_tensor.shape)}")

    target_layers = get_target_layers(model, model_name)
    reshape_transform = get_reshape_transform(model_name)

    if model_name == "vit_base":
        cam = EigenCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )
        targets = None                                       # EigenCAM class-agnostic
    else:
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(int(target_class))]

    # pytorch_grad_cam (1, H, W) numpy döner
    grayscale = cam(input_tensor=image_tensor, targets=targets)
    return grayscale[0]                                      # (H, W)


def overlay_heatmap(image_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Heatmap'i PIL görselin üstüne jet colormap ile overlay eder.

    Args:
        image_pil: orijinal görsel (RGB veya RGBA).
        heatmap: H×W, [0, 1] aralığında. image_pil ile aynı boyutta olmasa da
                 otomatik resize edilir.
        alpha: heatmap karışım oranı (0=sadece görsel, 1=sadece heatmap).
    """
    import cv2  # opencv-python — grad-cam ile birlikte kuruluyor

    img_rgb = image_pil.convert("RGB")
    img_np = np.array(img_rgb)                               # (H, W, 3) uint8
    h, w = img_np.shape[:2]

    # heatmap'i görselin boyutuna resize
    if heatmap.shape != (h, w):
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap

    heatmap_uint8 = np.uint8(np.clip(heatmap_resized, 0.0, 1.0) * 255)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    blended = (alpha * colored + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(blended)


# ----- yardımcılar ----------------------------------------------------------

def image_to_input_tensor(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL görseli → (1, 3, 224, 224) tensor (eval transform). Gradient gerektirir
    (Grad-CAM backward yapar) — bu yüzden requires_grad=True set edilir."""
    transform = build_eval_transform()
    tensor = transform(pil_image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    return tensor


def cleanup_cache() -> None:
    """Notebook/uzun servislerde Grad-CAM hook'larını temizlemek için."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----- standalone smoke test ------------------------------------------------

def _smoke_test(image_path: Optional[Path] = None) -> None:
    """Komut satırından çalıştırılabilir hızlı kontrol:
        python -m src.gradcam_utils
    """
    import sys
    from src.config import CLASS_NAMES, MODELS_DIR, FIGURES_DIR
    from src.models import create_model
    from src.utils import get_device

    device = get_device()
    print(f"Device: {device}")

    if image_path is None:
        # data/processed/test'ten ilk sınıfın ilk görselini bul
        from src.config import TEST_DIR
        candidates = sorted(TEST_DIR.glob("*/*"))
        if not candidates:
            print(f"[ERROR] Test görsel bulunamadı: {TEST_DIR}")
            sys.exit(1)
        image_path = candidates[0]
    print(f"Image: {image_path}")

    pil = Image.open(image_path).convert("RGB")

    out_dir = FIGURES_DIR / "gradcam_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ["resnet50", "efficientnet_b0", "vit_base"]:
        weights = MODELS_DIR / f"{name}.pth"
        if not weights.exists():
            print(f"[skip] {name}: ağırlık yok ({weights})")
            continue
        print(f"\n--- {name} ---")
        model = create_model(name, pretrained=False).to(device)
        state = torch.load(weights, map_location=device)
        model.load_state_dict(state)
        model.eval()

        tensor = image_to_input_tensor(pil, device)
        with torch.no_grad():
            logits = model(tensor)
        pred_idx = int(logits.argmax(dim=1).item())
        pred_class = CLASS_NAMES[pred_idx]
        print(f"Top-1: {pred_class} ({pred_idx})")

        heatmap = generate_gradcam(model, name, tensor, target_class=pred_idx)
        print(f"heatmap shape: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

        overlay = overlay_heatmap(pil, heatmap, alpha=0.45)
        out_path = out_dir / f"{name}_smoke.png"
        overlay.save(out_path)
        print(f"saved: {out_path}")

        del model
        cleanup_cache()


if __name__ == "__main__":
    _smoke_test()
