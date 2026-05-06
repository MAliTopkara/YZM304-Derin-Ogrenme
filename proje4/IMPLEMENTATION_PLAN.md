# YZM304 Proje 4 — Implementation Plan

**Proje:** Oyun Ekran Görüntülerinden Oyun Tespiti — Transfer Learning Tabanlı CNN ve Vision Transformer Mimarilerinin Karşılaştırmalı Analizi
**Ek Bileşen:** Lokal Web Demo (FastAPI + React) — model karşılaştırma & Grad-CAM görselleştirme

> Bu doküman, `YZM304_Proje4_Taslak_OyunSiniflandirma.md.pdf` dosyasındaki orijinal taslağı baz alır ve web demo bileşenini akademik gereksinimleri **bozmadan** entegre eder. Orijinal taslaktaki tüm zorunlu maddeler (IMRAD sunum, blog, GitHub, 3 model karşılaştırması) korunur; web demo bunların **üzerine** eklenir.

---

## 0. Yönetici Özeti

| Bileşen | Durum | Hedef |
|---|---|---|
| 3 model eğitimi (ResNet50, EfficientNetB0, ViT-B/16) | Zorunlu | Hafta 2-3 sonu |
| IMRAD sunum + blog + GitHub | Zorunlu | Hafta 4 |
| **Web demo (yeni)** | **Sunum gücü için** | **Hafta 3 sonu - 4 başı** |
| Grad-CAM görselleştirme | Opsiyonel → **Demo'da zorunlu** | Hafta 3 |

**Demo'nun katma değeri:** Sunumda *real-world deployment* argümanı, README'de canlı GIF, "tüm modelleri yan yana karşılaştır" modu projenin tezini somut hale getirir.

---

## 1. Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────┐
│                       Eğitim Pipeline                       │
│  Dataset → preprocessing → 3 model (timm) → .pth ağırlıklar │
└─────────────────────────────────────────────────────────────┘
                            │
                  results/models/*.pth
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌──────────────────┐                  ┌──────────────────────┐
│  Akademik Çıktı  │                  │     Web Demo (Yeni)  │
│  - IMRAD slides  │                  │  ┌────────────────┐  │
│  - Blog yazısı   │                  │  │ React Frontend │  │
│  - Confusion mtx │                  │  └────────┬───────┘  │
│  - metrics.csv   │                  │           │ HTTP     │
│                  │                  │  ┌────────▼───────┐  │
└──────────────────┘                  │  │ FastAPI Server │  │
                                      │  │  + 3 model     │  │
                                      │  │  + Grad-CAM    │  │
                                      │  └────────────────┘  │
                                      └──────────────────────┘
```

---

## 2. Teknoloji Stack

### Eğitim (orijinal taslakla aynı)
- PyTorch ≥2.0, torchvision, **timm**, albumentations
- scikit-learn, pandas, numpy, matplotlib, seaborn, Pillow, tqdm

### Web Demo (yeni)
| Katman | Seçim | Gerekçe |
|---|---|---|
| Backend | **FastAPI** + uvicorn | Async, otomatik OpenAPI docs, PyTorch ile sorunsuz |
| Inference | torch + timm (eğitimle aynı) | Model formatı uyumu |
| Görselleştirme | **pytorch-grad-cam** | Grad-CAM heatmap üretimi |
| Frontend | **Vite + React + TypeScript + TailwindCSS** | Hızlı build, modern, sunumda profesyonel görünüm |
| UI bileşenleri | **shadcn/ui** + lucide-react | Hazır temiz kart/buton/upload bileşenleri |
| Grafikler | **recharts** veya chart.js | Güven barları, karşılaştırma grafikleri |
| Dosya yükleme | react-dropzone | Drag-drop UX |

### Plan B (zaman dararsa)
- **Gradio** ile tek dosyalık demo: 1 günde hazır, daha az "wow" ama yine etkileyici. Backend/frontend ayrımı yok, ama Grad-CAM ve model seçimi yine yapılabilir.

---

## 3. Güncellenmiş Klasör Yapısı

Orijinal taslak yapısına `webapp/` ve birkaç yardımcı dosya eklenmiştir:

```
derinogrenme_proje4/
├── README.md                          # Proje tanıtımı + demo GIF + kurulum
├── IMPLEMENTATION_PLAN.md             # (bu dosya)
├── requirements.txt                   # Eğitim bağımlılıkları
├── .gitignore                         # data/raw, *.pth, node_modules vb.
│
├── Dataset/                           # ✅ Mevcut - 10 sınıf klasörü
│   ├── Among Us/  Apex Legends/  ...  (her biri ~1000 PNG)
│
├── data/
│   ├── processed/                     # Stratified split sonrası
│   │   ├── train/  val/  test/
│   └── EDA.ipynb                      # Keşifsel analiz
│
├── src/                               # Orijinal taslakla aynı
│   ├── __init__.py
│   ├── config.py                      # Hiperparametreler, sınıf isimleri
│   ├── dataset.py                     # PyTorch Dataset
│   ├── transforms.py                  # Augmentation pipeline
│   ├── models.py                      # 3 modelin factory'si (timm)
│   ├── train.py                       # Eğitim entry-point
│   ├── evaluate.py                    # Test set metrikleri
│   ├── visualize.py                   # Confusion matrix, loss plots
│   ├── gradcam_utils.py               # 🆕 Web demo için Grad-CAM yardımcısı
│   └── utils.py
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_training_resnet50.ipynb
│   ├── 03_training_efficientnet.ipynb
│   ├── 04_training_vit.ipynb
│   └── 05_comparison.ipynb            # 3 modelin yan yana karşılaştırması
│
├── results/
│   ├── models/                        # resnet50.pth, efficientnet_b0.pth, vit_base.pth
│   ├── figures/                       # Confusion matrix, accuracy/loss eğrileri
│   ├── logs/                          # Training logs (tensorboard veya csv)
│   └── metrics.csv                    # Tüm metrikleri tek tabloda toplayan özet
│
├── webapp/                            # 🆕 Lokal web demo
│   ├── README.md                      # Kurulum + ekran görüntüleri
│   ├── docker-compose.yml             # Tek komutla çalıştırma (opsiyonel)
│   │
│   ├── backend/
│   │   ├── main.py                    # FastAPI app (endpoints)
│   │   ├── inference.py               # Model yükleme + tahmin (3 model lazy-load)
│   │   ├── gradcam.py                 # Grad-CAM PNG üretimi
│   │   ├── schemas.py                 # Pydantic request/response
│   │   ├── config.py                  # Model yolları, sınıf etiketleri
│   │   └── requirements.txt
│   │
│   └── frontend/
│       ├── package.json
│       ├── vite.config.ts
│       ├── tailwind.config.ts
│       ├── public/
│       │   └── sample_images/         # Her sınıftan 1 örnek (hızlı demo için)
│       └── src/
│           ├── main.tsx
│           ├── App.tsx                # Router
│           ├── pages/
│           │   ├── UploadPage.tsx     # 1️⃣ Görsel yükleme
│           │   ├── ModelSelectPage.tsx# 2️⃣ Model seçimi (tek/tümü)
│           │   └── ResultsPage.tsx    # 3️⃣ Sonuçlar + Grad-CAM
│           ├── components/
│           │   ├── Dropzone.tsx
│           │   ├── ModelCard.tsx      # Parametre, beklenen doğruluk, eğitim süresi
│           │   ├── PredictionBar.tsx  # Top-3 güven barları
│           │   ├── GradCamOverlay.tsx # Heatmap görselleştirme
│           │   └── ComparisonView.tsx # 3 model yan yana
│           ├── lib/
│           │   └── api.ts             # FastAPI istemcisi (axios)
│           └── styles/
│
├── presentation/
│   ├── IMRAD_slides.pptx              # Sunum
│   └── demo.gif                       # Web demo screencast
│
└── blog/
    └── blog_post.md
```

---

## 4. Web Demo — Detaylı Tasarım

### 4.1. Kullanıcı Akışı (3 ekran)

#### Ekran 1: Upload
- Drag-drop alanı (react-dropzone) + "Dosya seç" butonu
- "Veya örnek görsellerden birini dene" — `public/sample_images/` altından her sınıftan 1 görsel
- Yüklendiğinde önizleme + "Devam et" butonu
- Doğrulamalar: PNG/JPG, ≤5MB, 224×224'e resize edilebilir

#### Ekran 2: Model Seçimi
3 model kartı (her biri parametre sayısı, mimari türü, eğitim süresi, beklenen test doğruluğu rozetli):
- ResNet50 — Klasik CNN — 25M params
- EfficientNetB0 — Modern CNN — 5M params
- ViT-Base/16 — Transformer — 86M params

🔥 **+ Dördüncü buton: "Tümünü Karşılaştır"** — projenin asıl tezini canlı sergiler.

#### Ekran 3: Sonuçlar
**Tek model modu:**
- Sol panel: orijinal görsel + Grad-CAM heatmap overlay (slider ile şeffaflık ayarı)
- Sağ panel:
  - Top-3 tahmin (animasyonlu güven barları %)
  - Inference süresi (ms) + model boyutu (MB) rozetleri
  - "Modeli değiştir" / "Yeni görsel" butonları

**Karşılaştırma modu:**
- 3 sütun yan yana, her biri kendi Top-1 tahmini + güven %'si
- En altta: hangi model en yüksek güvenle hangi sınıfı seçti karşılaştırma çubuğu
- Eğer sample_images'tan seçildiyse "doğru cevap" rozeti gösterilir → kim bildi/bilemedi anında görünür

### 4.2. Backend API Endpoints

```python
GET  /health                              # Sunucu durumu
GET  /models                              # Mevcut 3 modelin metadata listesi
POST /predict?model=resnet50              # Tek model tahmini (multipart/form-data)
POST /predict/all                         # 3 modelin paralel tahmini
POST /gradcam?model=resnet50              # Grad-CAM PNG bytes (overlay için)
GET  /samples                             # Demo görsel listesi
```

**Response şeması (`/predict`):**
```json
{
  "model": "resnet50",
  "predictions": [
    {"class": "Fortnite", "confidence": 0.92},
    {"class": "Apex Legends", "confidence": 0.05},
    {"class": "Free Fire", "confidence": 0.02}
  ],
  "inference_ms": 18.4,
  "model_size_mb": 98.2
}
```

### 4.3. Performans Notları
- 3 model **lazy-load** edilir (ilk istek ~1-2sn, sonrası <50ms RTX 5070 Ti'de)
- `torch.no_grad()` + `model.eval()` mod
- Grad-CAM için son conv katmanı seçimi: ResNet50 → `layer4`, EfficientNetB0 → `conv_head`, ViT → attention rollout (özel implementasyon)
- Karşılaştırma modunda 3 model **sıralı** çalışır (paralel batch için tek GPU yeterli ve daha güvenli)

---

## 5. Haftalık Plan (Web Demo Entegre Edilmiş)

### Hafta 1 — Veri Hazırlama ve Altyapı
- [x] Dataset indirildi (10 sınıf klasörü mevcut)
- [ ] Veri uygunluk kontrolü scripti (`notebooks/01_EDA.ipynb`):
  - Sınıf başına dosya sayısı, görsel boyut tutarlılığı, RGB/RGBA, bozuk dosya
  - Her sınıftan 3 örnek görsel grid (zaten `dataset_sample_grid.png` var)
- [ ] Stratified train/val/test split (70/15/15) → `data/processed/`
- [ ] GitHub repo + README iskelet
- [ ] `src/config.py`, `src/dataset.py`, `src/transforms.py` taslakları
- [ ] **Web demo için:** FastAPI ve frontend için boş scaffold (Vite + Tailwind kurulumu) — eğitim sürerken paralel geliştirilebilsin diye

### Hafta 2 — Baseline (ResNet50) ve EfficientNetB0
- [ ] `src/models.py`: 3 modelin factory'si
- [ ] `src/train.py`: training loop, checkpointing, early stopping
- [ ] ResNet50 eğitimi (~45 dk) → `results/models/resnet50.pth`
- [ ] EfficientNetB0 eğitimi (~30 dk) → `results/models/efficientnet_b0.pth`
- [ ] İlk metriklerin `results/metrics.csv`'a yazılması
- [ ] **Web demo için:** `webapp/backend/inference.py` — ResNet50 üzerinden `/predict` endpoint'i çalışıyor olsun (1 model yeter, sonra çoğaltılır)

### Hafta 3 — ViT, Karşılaştırma ve Demo Backend
- [ ] ViT-Base/16 eğitimi (~90 dk, batch size 16) → `results/models/vit_base.pth`
- [ ] `notebooks/05_comparison.ipynb`: 3 model side-by-side metrikler
- [ ] Confusion matrix, yanlış sınıflandırılan örnekler analizi
- [ ] Grad-CAM görselleştirmeleri (notebook + `src/gradcam_utils.py`)
- [ ] **Web demo backend:**
  - 3 model lazy-load
  - `/predict/all` endpoint'i (karşılaştırma modu)
  - `/gradcam` endpoint'i
- [ ] **Web demo frontend:** UploadPage + ModelSelectPage + ResultsPage MVP

### Hafta 4 — Demo Polish, IMRAD ve Teslim
**Pazartesi-Salı:**
- [ ] Frontend polish: Tailwind ile temiz tasarım, animasyonlar, loading state'ler
- [ ] Karşılaştırma görünümü (ComparisonView)
- [ ] `webapp/README.md` — kurulum ve ekran görüntüleri
- [ ] Demo screencast (~30 sn) → `presentation/demo.gif`

**Çarşamba-Perşembe:**
- [ ] **IMRAD PowerPoint sunumu** (bkz. §6)
- [ ] Blog yazısı (`blog/blog_post.md`)
- [ ] GitHub README finalize: kurulum, sonuçlar, demo GIF, blog/sunum linki
- [ ] Sunum provası (10-12 dk hedef)

**Cuma / 13-14. hafta:** Sınıfta sunum.

---

## 6. IMRAD Sunum İskeleti (Web Demo Entegre)

| Slayt | İçerik | Web demo etkisi |
|---|---|---|
| 1. Title | Başlık + ad + ders | — |
| 2. Introduction | Streaming pazar büyüklüğü, otomatik kategorizasyon ihtiyacı, problem tanımı, araştırma sorusu | — |
| 3. Methods (1/3) | Dataset tanıtımı, EDA grafikleri, preprocessing/augmentation | — |
| 4. Methods (2/3) | 3 model mimarisi (ResNet50 / EfficientNetB0 / ViT) — diyagramlar | — |
| 5. Methods (3/3) | Eğitim protokolü + **mimari diyagram (eğitim → API → frontend)** | 🆕 Demo'yu burada tanıt |
| 6. Results (1/2) | Accuracy/F1 tablosu, eğitim süresi vs doğruluk grafiği | — |
| 7. Results (2/2) | Confusion matrix (en iyi model) + karıştırılan sınıf örnekleri | — |
| 8. **Live Demo** | **Web demo screencast (GIF)** + Grad-CAM örnekleri | 🔥 **Sunumun zirvesi** |
| 9. Discussion | Hangi model kazandı/neden, model boyutu vs doğruluk trade-off, sınırlılıklar | — |
| 10. Conclusion | Ana bulgular, real-world deployment kanıtı (demo) | — |
| 11. References | timm, ImageNet, ViT/ResNet/EfficientNet papers | — |

**Sunum süresi hedefi:** 10-12 dk → demo slaydı 1.5-2 dk almalı.

---

## 7. Blog Yazısı Akışı

`blog/blog_post.md` — orijinal taslakla aynı plan, **+ "Modeli kendin dene" bölümü**:
1. Giriş (gaming + AI kancası)
2. Problem (benzer grafik stilleri zorluğu)
3. Yöntem (3 model basit dilde)
4. Sonuçlar (grafikler, confusion matrix)
5. Şaşırtıcı gözlemler (hangi oyunlar karıştırıldı)
6. **🆕 Lokal demoyu çalıştır:** GitHub kurulum komutları + ekran görüntüleri
7. Sonuç ve gelecek çalışmalar
8. GitHub linki + iletişim

---

## 8. Risk Yönetimi ve Plan B

| Risk | Olasılık | Plan B |
|---|---|---|
| ViT eğitimi VRAM yetersizliği | Orta | `swin_tiny_patch4_window7_224` veya batch size 8 + gradient accumulation |
| Frontend zaman almaya başlar | Orta-Yüksek | **Gradio'ya geçiş** (tek dosyada tüm demo) — 1 günde hazır |
| Grad-CAM ViT için zor | Yüksek | ViT için attention rollout yerine sadece CNN'lerde Grad-CAM göster, ViT için "attention map" kullan |
| Demo screencast son güne kalır | Orta | Hafta 3 sonunda MVP screencast çek, hafta 4'te güncelle |
| Web demo eğitim metriklerine uygun çıkmaz | Düşük | Inference transform pipeline'ının eğitimle **aynı** olduğunu test et (özellikle normalize) |

---

## 9. Kalite Kontrol Checklist (Teslim Öncesi)

**Akademik:**
- [ ] 3 model eğitildi, `results/models/` altında ağırlıklar var
- [ ] `metrics.csv` 3 modelin Accuracy, Macro-F1, Weighted-F1, Top-3 Acc, eğitim süresi, model boyutu içeriyor
- [ ] Confusion matrix figürleri `results/figures/` altında
- [ ] IMRAD sunumu 11 slayt, ≤12 dk
- [ ] Blog yazısı yayında
- [ ] GitHub README: kurulum + sonuçlar + demo GIF + blog linki

**Web Demo:**
- [ ] `cd webapp/backend && uvicorn main:app` ile API çalışıyor
- [ ] `cd webapp/frontend && npm run dev` ile UI açılıyor
- [ ] 3 ekran akışı: Upload → Model Select → Results sorunsuz
- [ ] "Tümünü Karşılaştır" modu çalışıyor
- [ ] Grad-CAM heatmap görüntüleniyor
- [ ] `webapp/README.md` kurulum talimatları doğru

---

## 10. Sıradaki Adım

İlk pratik aksiyon: **Hafta 1 görevlerine başla.**
1. `notebooks/01_EDA.ipynb` ile veri seti uygunluk kontrolü
2. Stratified split scripti
3. `src/` modüllerinin iskeletini at
4. GitHub repo'yu oluştur (`git init`, ilk commit)

Sonra Hafta 2'ye geçince web demo backend'i paralel başlar.

---

*Bu plan canlı bir dokümandır. Hafta sonlarında güncellenecek; tamamlanan maddeler işaretlenecek.*
