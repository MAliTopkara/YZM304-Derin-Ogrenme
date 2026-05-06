"""FastAPI entry-point for the gameplay classifier demo.

Endpoints:
    GET  /health                 — sunucu durumu
    GET  /models                 — 3 modelin metadatası (eğitildi mi?)
    POST /predict?model=<name>   — tek model tahmini (multipart 'file')
    POST /predict/all            — 3 modelin paralel (sıralı) tahmini
    POST /gradcam?model=<name>   — Grad-CAM overlay PNG (base64) + tahmin
    POST /gradcam/all            — 3 modelin Grad-CAM overlay'i tek upload'la
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from gradcam import gradcam_overlay, gradcam_overlay_all
from inference import (
    list_models as _list_models,
    predict_all,
    predict_single,
    warmup as _warmup,
)

app = FastAPI(title="Gameplay Classifier API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    """Sunucu açılışında 3 modeli eagerly yükle — ilk istekte gecikme olmasın."""
    print("[startup] Warming up models...")
    _warmup()
    print("[startup] Ready.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> list[dict]:
    return _list_models()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("resnet50"),
    top_k: int = Query(3, ge=1, le=10),
) -> dict:
    image_bytes = await file.read()
    try:
        return predict_single(model, image_bytes, top_k=top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/all")
async def predict_all_endpoint(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10),
) -> list[dict]:
    image_bytes = await file.read()
    return predict_all(image_bytes, top_k=top_k)


@app.post("/gradcam")
async def gradcam_endpoint(
    file: UploadFile = File(...),
    model: str = Query("resnet50"),
    alpha: float = Query(0.45, ge=0.0, le=1.0),
) -> dict:
    """Grad-CAM (CNN) / EigenCAM (ViT) overlay PNG'sini base64 olarak döner.

    Response:
        model, display_name, predicted_class, predicted_index, confidence,
        overlay_png_b64, inference_ms, alpha
    """
    image_bytes = await file.read()
    try:
        return gradcam_overlay(model, image_bytes, alpha=alpha)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/gradcam/all")
async def gradcam_all_endpoint(
    file: UploadFile = File(...),
    alpha: float = Query(0.45, ge=0.0, le=1.0),
) -> list[dict]:
    """3 modelin Grad-CAM overlay'ini sıralı üretir (eğitilmemiş model için error alanı dolu)."""
    image_bytes = await file.read()
    return gradcam_overlay_all(image_bytes, alpha=alpha)
