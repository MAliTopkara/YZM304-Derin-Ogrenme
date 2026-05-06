"""Inference yardımcıları — eğitilmiş modelleri lazy-load eder ve tek görselden tahmin üretir.

Eğitim repo'sundaki `src.transforms` ve `src.config` modüllerini yeniden kullanır
(eval transformları eğitimle birebir aynı olmak zorunda, aksi halde drift olur).
"""
from __future__ import annotations

import io
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CLASS_NAMES,
    MODEL_CONFIGS,
    MODELS_DIR,
)
from src.models import create_model  # noqa: E402
from src.transforms import build_eval_transform  # noqa: E402
from src.utils import get_device, model_size_mb  # noqa: E402


_DEVICE = get_device()
_TRANSFORM = build_eval_transform()
_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_MODEL_META: dict[str, dict] = {}


def list_models() -> list[dict]:
    """webapp UI için 3 modelin metadatası."""
    out = []
    for name, cfg in MODEL_CONFIGS.items():
        weights = MODELS_DIR / f"{name}.pth"
        out.append({
            "name": name,
            "display_name": _display_name(name),
            "paradigm": _paradigm(name),
            "params_m": _approx_params_m(name),
            "trained": weights.exists(),
            "weights_path": str(weights) if weights.exists() else None,
        })
    return out


def _display_name(name: str) -> str:
    return {
        "resnet50": "ResNet50",
        "efficientnet_b0": "EfficientNetB0",
        "vit_base": "ViT-Base/16",
    }.get(name, name)


def _paradigm(name: str) -> str:
    return {
        "resnet50": "Klasik CNN",
        "efficientnet_b0": "Modern CNN",
        "vit_base": "Transformer",
    }.get(name, "?")


def _approx_params_m(name: str) -> float:
    return {"resnet50": 25.0, "efficientnet_b0": 5.0, "vit_base": 86.0}.get(name, 0.0)


def load_model(name: str) -> torch.nn.Module:
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(MODEL_CONFIGS)}")

    weights = MODELS_DIR / f"{name}.pth"
    if not weights.exists():
        raise FileNotFoundError(
            f"Eğitilmiş ağırlık bulunamadı: {weights}. "
            "Önce `python -m src.train --model {name}` çalıştır."
        )

    model = create_model(name, pretrained=False).to(_DEVICE)
    state = torch.load(weights, map_location=_DEVICE)
    model.load_state_dict(state)
    model.eval()

    _MODEL_CACHE[name] = model
    _MODEL_META[name] = {
        "size_mb": model_size_mb(model),
        "weights_path": str(weights),
    }
    return model


def _prepare_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes))
    tensor = _TRANSFORM(img).unsqueeze(0).to(_DEVICE)
    return tensor


def _compute_uncertainty(probs: np.ndarray) -> dict:
    """Tüm sınıf olasılıklarından kalibrasyon metrikleri:

    - max_prob: top-1 güveni (zaten predictions[0].confidence ile aynı, kolaylık)
    - margin:   top-1 ile top-2 olasılıkları arasındaki fark — küçükse "ikisi yarışıyor"
    - entropy_normalized: 0..1 arası shannon entropy'nin log(K)'ya bölünmüş hali —
                          0 = kesin, 1 = uniform (model hiçbir şey bilmiyor)
    - level: "high" (model emin) / "medium" / "low" (muhtemelen OOD görsel)

    Eşikler kalibre edilmiş değil — heuristic. Test set'te %99+ doğruluk olduğu için
    'high' eşiği yüksek tutuldu.
    """
    sorted_probs = np.sort(probs)[::-1]
    max_prob = float(sorted_probs[0])
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

    # numerical safe entropy
    eps = 1e-12
    entropy = float(-np.sum(probs * np.log(probs + eps)))
    entropy_norm = entropy / math.log(len(probs))   # 0..1

    if max_prob >= 0.85 and entropy_norm <= 0.20 and margin >= 0.50:
        level = "high"
    elif max_prob < 0.50 or entropy_norm > 0.50:
        level = "low"
    else:
        level = "medium"

    return {
        "max_prob": round(max_prob, 6),
        "margin": round(margin, 6),
        "entropy_normalized": round(entropy_norm, 6),
        "level": level,
    }


@torch.no_grad()
def predict_single(model_name: str, image_bytes: bytes, top_k: int = 3) -> dict:
    model = load_model(model_name)
    tensor = _prepare_image(image_bytes)

    t0 = time.perf_counter()
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    inference_ms = (time.perf_counter() - t0) * 1000

    top_idx = probs.argsort()[::-1][:top_k]
    predictions = [
        {"class": CLASS_NAMES[i], "confidence": float(probs[i])}
        for i in top_idx
    ]
    uncertainty = _compute_uncertainty(probs)
    return {
        "model": model_name,
        "display_name": _display_name(model_name),
        "predictions": predictions,
        "uncertainty": uncertainty,
        "inference_ms": round(inference_ms, 2),
        "size_mb": round(_MODEL_META.get(model_name, {}).get("size_mb", 0.0), 1),
    }


@torch.no_grad()
def predict_all(image_bytes: bytes, top_k: int = 3) -> list[dict]:
    out = []
    for name in MODEL_CONFIGS:
        try:
            out.append(predict_single(name, image_bytes, top_k=top_k))
        except FileNotFoundError as e:
            out.append({
                "model": name,
                "display_name": _display_name(name),
                "error": str(e),
                "predictions": [],
            })
    return out


def warmup(names: Iterable[str] | None = None) -> None:
    """İlk istekte gecikmeyi azaltmak için modelleri önceden yükle."""
    targets = list(names) if names else list(MODEL_CONFIGS)
    for n in targets:
        try:
            load_model(n)
        except FileNotFoundError:
            pass
