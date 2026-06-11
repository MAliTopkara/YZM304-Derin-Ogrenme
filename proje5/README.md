# Kemik Röntgenlerinde Kırık Tespiti

**Transfer Öğrenme Tabanlı CNN Mimarilerinin Karşılaştırmalı Analizi ve Açıklanabilir Yapay Zeka**

> *Fracture Detection in Bone X-rays: A Comparative Analysis of Transfer-Learning-Based CNN Architectures with Explainable AI*

YZM304 Derin Öğrenme — 5. Proje · Ankara Üniversitesi

---

## 📌 Özet

Bu proje, **FracAtlas** röntgen veri kümesi üzerinde **kırık var/yok** ikili sınıflandırması için **3 farklı CNN mimarisini** —ResNet-50, DenseNet-121 ve EfficientNet-B0— ortak bir transfer öğrenme çatısında karşılaştırır. Her mimari **3 farklı tohum (seed)** ile eğitilir; sonuçlar Accuracy, Precision, Recall, F1, makro-F1 ve ROC-AUC ile raporlanır ve modeller arası farklar istatistiksel testlerle (eşli t-testi, McNemar, DeLong) sınanır. Son olarak **Grad-CAM** ile modelin gerçekten kırık bölgesine odaklanıp odaklanmadığı görselleştirilir.

## 📊 Veri Seti

| Özellik | Değer |
|---|---|
| Kaynak | [FracAtlas (Scientific Data, 2023)](https://doi.org/10.1038/s41597-023-02432-4) |
| Toplam görsel | 4.083 röntgen |
| Sınıflar | Kırıklı (717) · Kırıksız (3.366) |
| Dengesizlik | Kırık oranı ≈ %17.6 |
| Split | %70 / %15 / %15 (katmanlı, train / val / test) |

> Not: Veri seti (~327 MB) ve eğitilmiş model ağırlıkları (`outputs/checkpoints/`, ~488 MB) boyut nedeniyle depoya **dahil edilmemiştir**. Veri seti yukarıdaki kaynaktan indirilip `FracAtlas/images/{Fractured,Non_fractured}/` yapısına yerleştirilmelidir.

## 🧠 Modeller

| Model | Paradigma | Parametre | Pretrained |
|---|---|---|---|
| ResNet-50 | Klasik artık (residual) CNN | 23.5M | ImageNet |
| DenseNet-121 | Yoğun bağlantılı CNN | 7.0M | ImageNet |
| EfficientNet-B0 | Bileşik ölçekli verimli CNN | 4.0M | ImageNet |

## 🏆 Sonuçlar (Test Seti, 3 Tohum Ortalaması)

| Model | Accuracy | F1 | Makro-F1 | ROC-AUC |
|---|---|---|---|---|
| **ResNet-50** | **0.888 ± 0.003** | **0.677 ± 0.015** | **0.805 ± 0.008** | **0.891 ± 0.011** |
| DenseNet-121 | 0.872 ± 0.018 | 0.651 ± 0.030 | 0.786 ± 0.020 | 0.857 ± 0.012 |
| EfficientNet-B0 | 0.869 ± 0.011 | 0.626 ± 0.005 | 0.773 ± 0.005 | 0.868 ± 0.011 |

**Ana bulgular:**
- **ResNet-50** azınlık (kırık) sınıfında en iyi ve en kararlı; EfficientNet-B0'a F1 üstünlüğü istatistiksel olarak anlamlı (eşli t-testi, *p* = 0.032).
- **EfficientNet-B0** ≈ 1/6 parametreyle rekabetçi doğruluk → verimlilik ekseninde öne çıkıyor.
- **Youden** karar eşiği ile kırık geri-çağırması (recall) 0.67 → **0.73**.
- **Grad-CAM**: en iyi model doğru kararlarında kırık/kortikal süreksizlik bölgesine odaklanıyor.

## 📄 Rapor

IEEE/IMRAD formatında tam rapor:
- [rapor.pdf](rapor.pdf) — derlenmiş PDF (IEEEtran)
- [rapor.tex](rapor.tex) — LaTeX kaynağı
- [RAPOR.md](RAPOR.md) — Markdown sürümü

## 📁 Klasör Yapısı

```
proje5/
├── src/                # Kaynak kod (config, data, models, train, evaluate, gradcam, utils)
├── scripts/            # Eğitim, çoklu-tohum, karşılaştırma, istatistik, figür ve Grad-CAM scriptleri
├── configs/            # Deney konfigürasyonları
├── outputs/
│   ├── results/        # Karşılaştırma / agregasyon / verimlilik / istatistiksel test tabloları
│   ├── figures/        # Öğrenme eğrileri, ROC, karışıklık matrisleri, Grad-CAM, kutu grafiği
│   ├── logs/           # Eğitim logları ve JSON özetleri (her model × tohum)
│   └── splits/         # Veri bölme özeti
├── RAPOR.md / rapor.tex / rapor.pdf
├── IMPLEMENTATION_PLAN.md
└── requirements.txt
```

## 🚀 Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt

# 1) Veri bölmelerini hazırla (FracAtlas indirilmiş olmalı)
python scripts/prepare_splits.py

# 2) Tek bir modeli eğit
python scripts/train_model.py --model resnet50 --seed 42

# 3) Tüm modelleri çoklu tohumla eğit
bash scripts/run_multi_seed.sh

# 4) Karşılaştırma, istatistik ve figürler
python scripts/aggregate_seeds.py
python scripts/compare_models.py
python scripts/stat_tests.py
python scripts/final_report_figures.py

# 5) Grad-CAM açıklanabilirlik
python scripts/run_all_gradcam.py
```

## 🔧 Eğitim Ayarları (Ortak)

İki fazlı transfer öğrenme (2 epoch dondurulmuş baş + ince ayar), AdamW (wd 10⁻⁴), LR baş/backbone = 10⁻³/10⁻⁴, batch 32, 224×224, ağırlıklı rastgele örnekleyici (sınıf dengesizliği), doğrulama F1'inde erken durdurma (patience 5), karışık duyarlık. Tohumlar: 42, 123, 2024.
