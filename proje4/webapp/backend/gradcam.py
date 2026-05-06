"""Backend Grad-CAM logic — `/gradcam` endpoint için.

inference.py'deki model cache + transform'u yeniden kullanır (memory efficient).
Çıktı: base64 PNG + tahmin bilgisi (frontend tek istekte hem heatmap hem etiket alır).
"""
from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

# Eğitim kodunu (src/...) bu dosya cwd ne olursa olsun bulabilsin diye repo
# kökünü path'e ekle. inference.py de aynısını yapar; bu modül onu inference'tan
# önce import etse bile çalışsın diye burada da kopyalandı.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch                                                        # noqa: E402
from PIL import Image                                               # noqa: E402

from src.config import CLASS_NAMES                                  # noqa: E402
from src.gradcam_utils import (                                     # noqa: E402
    generate_gradcam,
    overlay_heatmap,
)

from inference import (                                             # noqa: E402
    _DEVICE,
    _TRANSFORM,
    _display_name,
    load_model,
)


def _prepare_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _make_tensor(pil: Image.Image, requires_grad: bool = False) -> torch.Tensor:
    tensor = _TRANSFORM(pil).unsqueeze(0).to(_DEVICE)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def gradcam_overlay(model_name: str, image_bytes: bytes, alpha: float = 0.45) -> dict:
    """Tek görsel + tek model için heatmap overlay PNG ve tahmin döner.

    Returns:
        {
            "model": str,
            "display_name": str,
            "predicted_class": str,
            "predicted_index": int,
            "confidence": float,
            "overlay_png_b64": str,        # base64-encoded PNG bytes
            "inference_ms": float,
            "alpha": float,
        }

    Raises:
        FileNotFoundError: model ağırlığı yok (load_model'den).
        ValueError: bilinmeyen model adı (load_model'den).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha 0..1 aralığında olmalı, got {alpha}")

    model = load_model(model_name)                                  # cached
    pil = _prepare_pil(image_bytes)

    t0 = time.perf_counter()

    # 1) tahmin (gradient'siz, hızlı)
    with torch.no_grad():
        pred_tensor = _make_tensor(pil, requires_grad=False)
        logits = model(pred_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])

    # 2) heatmap (cam library kendi forward+backward'ını yapar)
    cam_tensor = _make_tensor(pil, requires_grad=True)
    heatmap = generate_gradcam(model, model_name, cam_tensor, target_class=pred_idx)
    overlay = overlay_heatmap(pil, heatmap, alpha=alpha)

    # 3) PNG base64
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    inference_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "model": model_name,
        "display_name": _display_name(model_name),
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": confidence,
        "overlay_png_b64": b64,
        "inference_ms": round(inference_ms, 2),
        "alpha": alpha,
    }


def gradcam_overlay_all(image_bytes: bytes, alpha: float = 0.45) -> list[dict]:
    """3 modelin heatmap'ini sıralı üretir (tek GPU üzerinde paralel batch riskli)."""
    from src.config import MODEL_CONFIGS  # noqa: WPS433 (local import, avoid cycle)

    out: list[dict] = []
    for name in MODEL_CONFIGS:
        try:
            out.append(gradcam_overlay(name, image_bytes, alpha=alpha))
        except FileNotFoundError as e:
            out.append({
                "model": name,
                "display_name": _display_name(name),
                "error": str(e),
                "overlay_png_b64": "",
                "predicted_class": "",
                "predicted_index": -1,
                "confidence": 0.0,
                "inference_ms": 0.0,
                "alpha": alpha,
            })
    return out
