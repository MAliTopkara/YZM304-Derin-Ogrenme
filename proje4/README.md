# Oyun Ekran Görüntülerinden Oyun Tespiti

**Transfer Learning Tabanlı CNN ve Vision Transformer Mimarilerinin Karşılaştırmalı Analizi**

> *Video Game Identification from Gameplay Screenshots: A Comparative Analysis of Transfer Learning Based CNN and Vision Transformer Architectures*

YZM304 Derin Öğrenme — 4. Proje · Ankara Üniversitesi

---

## 📌 Özet

Bu proje, 10 popüler video oyununa ait ekran görüntülerini sınıflandırmak için **3 farklı derin öğrenme mimarisini** karşılaştırır: klasik CNN (ResNet50), modern verimli CNN (EfficientNetB0) ve Transformer tabanlı (Vision Transformer — ViT-Base/16). Modeller doğruluk, eğitim süresi ve model boyutu açısından analiz edilir; ek olarak proje, eğitilmiş modelleri **lokal bir web arayüzünde** (FastAPI + React) interaktif olarak test edilebilir hale getirir.

## 🎯 Sınıflar (10 Oyun)

Among Us · Apex Legends · Fortnite · Forza Horizon · Free Fire · Genshin Impact · God of War · Minecraft · Roblox · Terraria

## 📊 Veri Seti

| Özellik | Değer |
|---|---|
| Kaynak | [Kaggle — Gameplay Images](https://www.kaggle.com/datasets/aditmagotra/gameplay-images) |
| Toplam görsel | 10.000 (sınıf başına 1000) |
| Boyut | 640 × 360 PNG |
| Toplam | ~2.5 GB |
| Split | 70% / 15% / 15% (train / val / test) |

EDA çıktıları: [results/eda_report.md](results/eda_report.md) · [results/figures/](results/figures/)

## 🧠 Modeller

| Model | Paradigma | Parametre | Pretrained | Beklenen Acc |
|---|---|---|---|---|
| ResNet50 | Klasik CNN | 25M | ImageNet | %85-92 |
| EfficientNetB0 | Modern CNN | 5M | ImageNet | %87-93 |
| ViT-Base/16 | Transformer | 86M | ImageNet-21k | %88-95 |

## 🚀 Kurulum

```bash
# 1. Bağımlılıklar
pip install -r requirements.txt

# 2. Veri seti split (70/15/15)
python scripts/run_split.py

# 3. EDA (opsiyonel, raporu yeniden üretir)
python scripts/run_eda.py
```

## 🏋️ Eğitim

```bash
# Tek model
python -m src.train --model resnet50
python -m src.train --model efficientnet_b0
python -m src.train --model vit_base
```

Hiperparametreler `src/config.py` içinde; eğitilmiş ağırlıklar `results/models/` altına kaydedilir.

## 📈 Değerlendirme

```bash
python -m src.evaluate --model resnet50 --weights results/models/resnet50.pth
```

Çıktılar: confusion matrix, sınıf bazlı F1, Top-3 accuracy → `results/figures/` ve `results/metrics.csv`.

## 🌐 Web Demo (Lokal)

İnteraktif demo: görsel yükle → model seç (veya tümünü karşılaştır) → tahminleri ve **Grad-CAM heatmap'i** gör.

```bash
# Backend
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (ayrı terminal)
cd webapp/frontend
npm install
npm run dev
```

Tarayıcı: http://localhost:5173

Detaylı kurulum: [webapp/README.md](webapp/README.md)

## 🗂️ Klasör Yapısı

```
derinogrenme_proje4/
├── Dataset/                 # Ham görseller (10 sınıf × 1000)
├── data/processed/          # Stratified split (train/val/test)
├── src/                     # Eğitim/değerlendirme modülleri
├── scripts/                 # EDA, split, yardımcı scriptler
├── notebooks/               # Jupyter notebookları
├── results/                 # Modeller, figürler, metrikler
├── webapp/                  # FastAPI backend + React frontend
├── presentation/            # IMRAD slaytları + demo gif
└── blog/                    # Blog yazısı
```

Tam plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## 📝 Sonuçlar

> _Bu bölüm eğitim tamamlandıktan sonra metriklerle güncellenecek._

| Model | Test Acc | Macro-F1 | Top-3 Acc | Eğitim süresi | Boyut |
|---|---|---|---|---|---|
| ResNet50 | – | – | – | – | – |
| EfficientNetB0 | – | – | – | – | – |
| ViT-Base/16 | – | – | – | – | – |

## 📚 Kaynaklar

- He et al., "Deep Residual Learning for Image Recognition" (2015)
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs" (2019)
- Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (2020)
- "From Pixels to Titles: Video Game Identification by Screenshots using CNNs" — arXiv:2311.15963
- [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

## 📄 Lisans

Akademik kullanım için.

---
*Son güncelleme: 2026 · YZM304 Derin Öğrenme*
