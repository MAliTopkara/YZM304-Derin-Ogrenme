# Web Demo — Gameplay Classifier

Lokal interaktif demo: yüklediğiniz oyun ekran görüntüsünü 3 modelden biriyle (veya hepsiyle) sınıflandırır, tahminleri ve Grad-CAM heatmap'ini gösterir.

## Mimari

```
[ React (Vite) :5173 ]  ─HTTP─►  [ FastAPI :8000 ]  ─►  [ 3 PyTorch model + Grad-CAM ]
```

## Kurulum

### Backend
```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### Frontend
```bash
cd webapp/frontend
npm install
npm run dev
```
UI: http://localhost:5173

## Geliştirme Durumu

- [x] Hafta 1: Backend iskelet (`/health`, `/models`)
- [ ] Hafta 2: `/predict?model=resnet50` (tek model inference)
- [ ] Hafta 3: `/predict/all` (3 model paralel) + `/gradcam` + frontend MVP
- [ ] Hafta 4: Frontend polish + demo screencast

Detay: bkz. proje kökündeki [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md).
