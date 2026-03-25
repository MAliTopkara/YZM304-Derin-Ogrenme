# YZM304 Derin Öğrenme Dersi — I. Proje Ödevi

**Mehmet Ali Topkara — 23291093**  
Ankara Üniversitesi · Yapay Zeka ve Veri Mühendisliği  
2025–2026 Bahar Dönemi

---

## 1. Giriş (Introduction)

Bu çalışmada, **sıfırdan (from scratch)** yazılmış çok katmanlı algılayıcı (MLP) modeli ile iki farklı sınıflandırma problemi üzerinde deneyler yapılmıştır:

| | BankNote Authentication | Wine Quality |
|---|---|---|
| **Görev** | İkili sınıflandırma | Çoklu sınıflandırma (7 sınıf) |
| **Boyut** | 1 372 × 4 | 6 497 × 11 |
| **Kayıp fonksiyonu** | Binary Cross Entropy | Categorical Cross Entropy |
| **Çıkış katmanı** | Sigmoid | Softmax |

13.03.2026 tarihinde laboratuvar saatinde BankNote veri seti üzerinde kurulan 1 gizli katmanlı baseline model (`notebooks/00_One_Hidden_Layer_MLP.ipynb`) temel alınarak; katman/nöron sayısı artırma, L2 regülarizasyon, veri ön işleme ve hiperparametre araması ile model iyileştirilmiştir. Aynı mimari ve hiperparametrelerle **Scikit-learn MLPClassifier** ve **PyTorch** kütüphaneleri kullanılarak karşılaştırma yapılmıştır.

---

## 2. Yöntemler (Methods)

### 2.1 Ortam ve Tekrarlanabilirlik

| Parametre | Değer |
|---|---|
| Python | ≥ 3.10 |
| Kurulum | `pip install -r requirements.txt` |
| Çalıştırma | `jupyter notebook notebooks/03_WineQuality_MultiClass.ipynb` |
| Global rastgele tohum | 42 |
| Optimizasyon algoritması | SGD (Stochastic Gradient Descent) |

### 2.2 Veri Ön İşleme

| Parametre | BankNote | Wine Quality |
|---|---|---|
| Veri bölme | %70 train / %15 val / %15 test | %70 train / %15 val / %15 test |
| Ölçeklendirme | StandardScaler | StandardScaler |
| Karıştırma | stratified, seed=42 | stratified, seed=42 |
| Sınıf etiketleri | 0 / 1 | 0–6 (orijinal: 3–9) |

### 2.3 Model Mimarileri — Wine Quality

| Model | Mimari | Gizli Katman | Açıklama |
|---|---|---|---|
| Model 1 | [11, 32, 7] | 1 | Lab çalışması yapısı (softmax uyarlaması) |
| Model 2 | [11, 64, 32, 7] | 2 | Genişletilmiş derin model |
| Model 3 | [11, 64, 32, 7] + L2 | 2 | L2 regülarizasyonlu (λ=0.01) |
| Model 4 | [11, 64, 32, 7] | 2 | Scikit-learn MLPClassifier |
| Model 5 | [11, 64, 32, 7] | 2 | PyTorch nn.Module |

### 2.4 Model Mimarileri — BankNote

| Model | Mimari | Gizli Katman | Açıklama |
|---|---|---|---|
| Model 1 | [4, 5, 1] | 1 | Lab baseline modeli |
| Model 2 | [4, 10, 1] | 1 | Geniş model |
| Model 3 | [4, 8, 4, 1] | 2 | Derin model |
| Model 4 | [4, 8, 4, 1] + L2 | 2 | L2 regülarizasyonlu |
| Model 5 | [4, 16, 8, 1] | 2 | Tuned model |
| Sklearn 1-2 | — | — | Scikit-learn MLPClassifier |
| PyTorch 1-2 | — | — | PyTorch nn.Module |

### 2.5 Ortak Hiperparametreler

| Hiperparametre | Değer |
|---|---|
| Öğrenme oranı (lr) | 0.01 (temel), hiperparametre aramasında değişken |
| Epoch sayısı | 2000 |
| Gizli katman aktivasyonu | tanh |
| Çıkış aktivasyonu | softmax (çoklu) / sigmoid (ikili) |
| Ağırlık başlatma | Xavier (tanh) / He (relu) |
| Bias başlatma | 0 |
| L2 regülarizasyon katsayısı | 0.01 (sadece regülarizasyonlu modellerde) |

### 2.6 Değerlendirme Metrikleri

- **Accuracy** (doğruluk)
- **Precision** (kesinlik) — macro average
- **Recall** (duyarlılık) — macro average
- **F1 Score** — macro average
- **Confusion Matrix** (karmaşıklık matrisi)

### 2.7 Sınıf Yapısı ve Modüler Tasarım

Proje gereksiz fonksiyon tekrarından kaçınmak için modüler bir sınıf yapısı kullanmaktadır:

- **`DataLoader`** — Veri yükleme, EDA ve ön işleme (constructor, public ve private metotlar)
- **`MLPScratch`** — Sıfırdan MLP sınıfı (forward/backward pass, eğitim döngüsü)
- **`metrics.py`** — Confusion matrix, cost/accuracy eğrileri, model karşılaştırma fonksiyonları
- **`utils.py`** — Aktivasyon fonksiyonları (sigmoid, tanh, relu, softmax) ve one-hot encoding

---

## 3. Sonuçlar (Results)

### 3.1 Wine Quality — Model Karşılaştırma

Tüm modeller aynı eğitim/test seti, aynı `random_state=42`, aynı ölçeklendirme (StandardScaler) ve SGD optimizasyonu ile eğitilmiştir.

| Model | Accuracy | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| M1: Scratch [11,32,7] | 52.82% | 0.2109 | 0.2078 | 0.2010 |
| M2: Scratch [11,64,32,7] | 54.15% | 0.2166 | 0.2169 | 0.2110 |
| M3: Scratch L2 Reg. | 54.15% | 0.2166 | 0.2169 | 0.2110 |
| **M4: Sklearn MLP** | **57.85%** | **0.2371** | **0.2432** | **0.2381** |
| M5: PyTorch MLP | 53.54% | 0.2174 | 0.2146 | 0.2093 |

### 3.2 Hiperparametre Araması

| Yapılandırma | Train Acc | Val Acc | Test Acc |
|---|:---:|:---:|:---:|
| [11,16,7] lr=0.01 | 0.5516 | 0.5108 | 0.5344 |
| [11,32,7] lr=0.01 | 0.5448 | 0.5292 | 0.5282 |
| [11,64,7] lr=0.01 | 0.5582 | 0.5241 | 0.5405 |
| [11,32,7] lr=0.05 | 0.5643 | 0.5467 | 0.5364 |
| [11,64,32,7] lr=0.01 | 0.5582 | 0.5262 | 0.5415 |
| **[11,64,32,7] lr=0.05** | **0.5832** | **0.5590** | **0.5754** |
| [11,128,64,7] lr=0.01 | 0.5652 | 0.5426 | 0.5477 |

### 3.3 Overfitting / Underfitting Analizi

| Model | Train Cost | Val Cost | Train Acc | Val Acc | Fark |
|---|:---:|:---:|:---:|:---:|:---:|
| Model 1 | 1.0897 | 1.1028 | 0.5448 | 0.5292 | 0.0155 |
| Model 2 | 1.0626 | 1.0895 | 0.5582 | 0.5262 | 0.0320 |

Train-Val accuracy farkları düşük olduğundan ciddi bir overfitting gözlenmemiştir. L2 regülarizasyon bu farkı daha da daraltmıştır.

Detaylı grafikler ve confusion matrix'ler:
- BankNote: `results/banknote/`
- Wine Quality: `results/wine_quality/`

---

## 4. Tartışma (Discussion)

### Temel Bulgular

1. **Veri Ön İşleme Etkisi:** StandardScaler ile yapılan ölçeklendirme, tüm modellerin yakınsamasını önemli ölçüde hızlandırmış ve performansı artırmıştır.

2. **Model Karmaşıklığı vs. Performans:** Katman ve nöron sayısının artırılması model kapasitesini yükseltmiş ancak Wine Quality gibi dengesiz veri setlerinde her zaman daha iyi sonuç vermemiştir. BankNote veri setinde ise derin modeller daha belirgin iyileşme sağlamıştır.

3. **Regülarizasyon Etkisi:** L2 regülarizasyon, overfitting riskini azaltarak train-val performans farkını daraltmıştır. Ancak bu veri setlerinde dramatik bir accuracy artışı sağlamamıştır.

4. **Kütüphane Karşılaştırması:** Scikit-learn MLPClassifier ve PyTorch ile aynı mimari ve SGD optimizasyonu kullanılarak eğitilen modeller, sıfırdan yazılan modelle karşılaştırılmıştır. Scikit-learn en iyi performansı göstermiş olup bunun sebebi kütüphanenin internal optimizasyon mekanizmalarıdır.

5. **Sınıf Dengesizliği:** Wine Quality veri setindeki dengesiz sınıf dağılımı (kalite 5 ve 6 ağırlıklı, kalite 3 ve 9 çok az), düşük örnekli sınıflarda performansı düşürmüş ve macro average metriklerini etkilemiştir.

### Gelecek Çalışmalar

- Mini-batch SGD ve Adam optimizasyonu
- Batch normalizasyonu ve Dropout regülarizasyonu
- Sınıf ağırlıklandırması (class weighting) — dengesiz sınıf dağılımı için
- Learning rate scheduling
- Daha fazla gizli katman ve nöron denemeleri

---

## Proje Dosya Yapısı

```
derinogrenme_proje1/
├── data/
│   ├── BankNote_Authentication.csv    # Lab çalışması ikili sınıflandırma veri seti
│   └── wine_quality.csv              # Çoklu sınıflandırma veri seti (UCI)
├── notebooks/
│   ├── 00_One_Hidden_Layer_MLP.ipynb  # Lab çalışması orijinal notebook (13.03.2026)
│   ├── 01_EDA_Preprocessing.ipynb     # BankNote EDA ve ön işleme
│   ├── 02_BankNote_Models.ipynb       # BankNote modelleri (5 scratch + sklearn + pytorch)
│   └── 03_WineQuality_MultiClass.ipynb  # Wine Quality ana proje notebook'u
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # DataLoader sınıfı (veri yükleme, EDA, ön işleme)
│   ├── mlp_scratch.py                 # MLPScratch sınıfı (from-scratch MLP)
│   ├── metrics.py                     # Metrik hesaplama ve görselleştirme
│   └── utils.py                       # Aktivasyon fonksiyonları ve yardımcı araçlar
├── results/
│   ├── banknote/                      # BankNote sonuç grafikleri
│   └── wine_quality/                  # Wine Quality sonuç grafikleri
├── README.md                          # Bu dosya (IMRAD formatı)
└── requirements.txt                   # Bağımlılıklar
```

## Kurulum ve Çalıştırma

```bash
# Bağımlılıkları kur
pip install -r requirements.txt

# Notebook'ları sırayla çalıştır
cd notebooks
jupyter notebook
# 01 → 02 → 03 sırasıyla Run All
```

## Gereksinimler

- Python ≥ 3.10
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch (CPU)
