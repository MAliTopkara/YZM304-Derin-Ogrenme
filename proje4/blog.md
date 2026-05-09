# Oyun Ekran Görüntülerinden Oyun Tespiti — 5 Modelin Yarışı

**Mehmet Ali Topkara** · YZM304 Derin Öğrenme · Proje 4 · 2026

> Klasik MLP'den Vision Transformer'a kadar 5 farklı derin öğrenme modelini
> aynı görsel sınıflandırma görevinde karşılaştırdığım deneyimi anlatıyorum.
> Hikâyenin sonunda bir sürpriz var: sıfırdan eğitilen küçücük bir CNN, 86
> milyon parametreli ViT ile yarışıyor.

---

## Neden Oyun Ekran Görüntüleri?

YZM304 dersinin 4. projesi için bir görsel sınıflandırma problemi seçmem
gerekiyordu. Sınıf arkadaşlarımın büyük çoğunluğu **sağlık ve insan tabanlı**
problemleri seçti — tıbbi görüntü sınıflandırması, yüz tanıma, duygu analizi
gibi konular. Ben özellikle farklı bir bağlam denemek istedim ve **oyun ekran
görüntülerini** seçtim.

Bu seçimin iki gerekçesi vardı. Birincisi pratikti: oyun pazarı 2024 itibarıyla
yaklaşık **200 milyar dolara** ulaşmış durumda. Twitch ve YouTube Gaming gibi
platformlara her gün milyonlarca saatlik gameplay videosu yükleniyor; bu
hacimde manuel etiketleme imkansız. Otomatik sınıflandırma sistemleri arama,
içerik moderasyonu ve telif kontrolü için kritik bir altyapı haline gelmiş
durumda.

İkinci gerekçe akademikti: oyun görselleri klasik veri setlerinden farklı
zorluklar barındırıyor. FPS oyunları benzer arayüzlere sahip; MMORPG'ler aynı
sahnede çok farklı görseller üretiyor; aynı oyun farklı modlarında oldukça
farklı görünebiliyor. Yani hem güncel bir problem alanı, hem de modelleri
gerçekten zorlayacak bir görsel zorluk seviyesi.

## Problem Tanımı

Görev basit bir cümleyle ifade edilebilir: **tek bir oyun ekran görüntüsü
verildiğinde, modelin görseli 10 önceden tanımlı sınıftan birine atfetmesi.**
Bu, makine öğrenmesinde **closed-set sınıflandırma** olarak bilinen bir
problem türü. Model her zaman 10 sınıftan birini seçmek zorunda; "bilmiyorum"
diyemez. Bu sınırlama hem matematiksel olarak ilginç hem de demoda göstermek
için iyi bir tartışma noktası.

Üç temel araştırma sorum vardı:
1. Görüntü yapısını yok sayan **MLP**, sıfırdan eğitilen **klasik CNN** ve
   ImageNet üzerinde eğitilmiş **transfer learning** modelleri aynı veri
   setinde nasıl performans gösterir?
2. Mimari değişiminin ve pretrained ağırlıkların etkisi sayısal olarak ne
   kadardır?
3. Modeller "emin oldukları" şey ile "haklı oldukları" şey arasında tutarlı mı?

## Birincil Metrik: Macro-F1

Dersin başından beri öğretildiği gibi, bir sınıflandırma probleminde sadece
accuracy bakmak yetersiz. Veri dengeli olduğu için (her sınıf 1000 örnek)
accuracy ve weighted-F1 yakın çıkıyor — ama **macro-F1** her sınıfa eşit
ağırlık veriyor ve precision–recall'u harmonik ortalama ile birleştiriyor.
Bir sınıfta iyi olup başkasında berbat olmak macro-F1'i aşağı çekiyor; bu
durumlar accuracy'de saklı kalabiliyor.

Bu nedenle tüm yorumları **macro-F1 üzerinden** yapıyorum. Bu seçim multi-class
balanced sınıflandırma için akademik standart aynı zamanda.

## Veri Seti ve Hazırlık

**Kaggle Gameplay Images** veri setini kullandım: 10 popüler oyundan **toplam
10.000 ekran görüntüsü**, her sınıf için 1000 örnek. Sınıflar şunlar: Among
Us, Apex Legends, Fortnite, Forza Horizon, Free Fire, Genshin Impact, God of
War, Minecraft, Roblox, Terraria. FPS, MMORPG, sandbox, racing — bilinçli bir
çeşitlilik var.

Görseller 640×360 piksel PNG formatında. İlk işlem **format dönüşümü**:
görsellerin yaklaşık %20'si RGBA (alfa kanallı) olduğundan tümünü RGB'ye
çevirdim. Sonra hepsini **224×224 piksele** resize ettim — ImageNet pretrained
modellerin beklediği girdi boyutu.

Veri setini **stratified split** ile %70 eğitim, %15 doğrulama, %15 test
oranlarında böldüm; seed=42 ile sabitledim. Eğitim setinde augmentation
uyguladım: rastgele kırpma, yatay çevirme, renk titreşimi ve ±10° rotation.
Validation ve test setlerinde sadece deterministik resize + center crop —
böylece test zamanı ölçümü tek bir görsel için sabit oluyor.

## Yarışan 5 Model

### 1. MLP — Yapay Sinir Ağı Baseline

İlk model, görüntü yapısını **tamamen yok sayan** klasik bir tam bağlı sinir
ağı. Görseli düz bir vektöre çeviriyor (224×224×3 = 150.528 piksel) ve üç tam
bağlı katmandan geçiriyor: 256 nöron → 128 nöron → 10 sınıf. Toplam **38,57
milyon parametre** — hepsi tam bağlı katmanlarda. Bu mimari komşuluk, kenar,
doku gibi uzamsal bilgileri hiç kullanmıyor.

Bu modeli pedagojik baseline olarak ekledim — derste öğretildiği gibi MLP
görüntülerde neden iyi çalışmaz, somut göstereyim diye.

### 2. CNN Scratch — Sıfırdan Eğitilen Klasik CNN

Klasik VGG-mini tarzında dört evrişim bloğundan oluşan bir CNN. Her blokta
3×3 convolution, batch normalization, ReLU ve 2×2 maxpooling var. Filter
sayıları her blokta ikiye katlanıyor (32 → 64 → 128 → 256), feature map
çözünürlüğü her blokta yarıya düşüyor (224 → 112 → 56 → 28 → 14). Sonunda
global average pooling + 128 birimlik tam bağlı katman + 10 sınıflık çıktı.

Toplam parametre sayısı **yalnızca 0,42 milyon** — yani MLP'den 92 kat daha
az. Önemli bir not: bu model **sıfırdan eğitildi**, hiçbir pretrained ağırlık
kullanılmadı.

### 3-4-5. Transfer Learning Üçlüsü

Üç modern mimariyi ImageNet pretrained ağırlıklarla yükledim ve fine-tune
ettim:

- **ResNet50** (2015, klasik CNN, 23,5M parametre) — residual bağlantılarla
  derin ağ eğitimini mümkün kılan sembolik mimari.
- **EfficientNetB0** (2019, modern CNN, 4M parametre) — MBConv blokları ve
  compound scaling fikri ile az parametreyle yüksek doğruluk.
- **ViT-Base/16** (2020, transformer, 86M parametre) — görseli 16×16
  patch'lere bölüp NLP transformer'ını uyguluyor.

Üçü de [timm](https://github.com/huggingface/pytorch-image-models) kütüphanesi
üzerinden tutarlı bir API ile yüklendi.

## Eğitim Protokolü ve Bir Crash Hikayesi

Adil karşılaştırma için **5 modeli aynı protokolde** eğittim. AdamW optimizer,
CosineAnnealingLR scheduler, 20 epoch maksimum, early stopping patience=5.
Baseline modeller için lr=1e-3, transfer learning için lr=1e-4 (pretrained
ağırlıkları bozmamak için daha düşük). AMP (Automatic Mixed Precision)
sayesinde transfer ve CNN scratch eğitimleri yaklaşık **%40 hızlandı**,
doğruluğa zarar vermedi.

Ama bu deneyim sancısız geçmedi. EfficientNetB0 eğitiminin ortasında **CUDA
illegal memory access** hatası aldım — saat boyu süren eğitim çöktü ve
o ana kadarki ilerleme kayboldu. Bu yüzden `train.py` script'ime
**`--resume`** desteği ekledim: her epoch sonu checkpoint kaydeder
(model + optimizer + scheduler + history), sonra `--resume` flag ile kaldığı
yerden devam eder. AMP + incremental history.csv yazımı + `torch.cuda.empty_cache()`
çağrıları crash riskini azalttı. İkinci denemede sorunsuz tamamlandı.

## Sonuçlar — Sürprizler ve Beklenenler

İşte 5 modelin test seti üzerindeki performansı:

| Model | Macro-F1 | Test Acc | Params | Boyut | Eğitim |
|---|---|---|---|---|---|
| **MLP** | **0,4794** | 0,5027 | 38,57 M | 147,1 MB | 11,8 dk |
| **CNN Scratch** | **0,9733** | 0,9733 | **0,42 M** | **1,6 MB** | 11,1 dk |
| **ResNet50** | **0,9913** | 0,9913 | 23,53 M | 89,8 MB | 10,6 dk |
| **EfficientNetB0** | **0,9907** | 0,9907 | 4,02 M | 15,3 MB | **9,5 dk** |
| **ViT-Base/16** | **0,9907** | 0,9907 | 85,81 M | 327,3 MB | 19,5 dk |

### Sürpriz 1: MLP gerçekten kötü, ama düşündüğüm kadar değil

MLP %47,94 macro-F1 ile bitirdi. Rastgele tahminin (%10) çok üzerinde, yani
model bir şeyler öğrenmiş — muhtemelen renk paleti, parlaklık dağılımı gibi
düşük seviyeli sinyaller. Ama görüntüdeki uzamsal yapıyı kullanmadığı için
bu noktanın üzerine çıkamıyor. Sınıf-bazlı bakıldığında özellikle **Apex
Legends ve Free Fire'da F1 değerleri 0,2 civarında**; model bu iki FPS'yi
birbirinden ayırt edemiyor çünkü ikisi de benzer renk paletine sahip.

### Sürpriz 2: CNN Scratch transfer learning ile yarışıyor

Asıl şaşırtıcı olan CNN Scratch'in performansı. **Yalnızca 0,42 milyon
parametre ile 0,9733 macro-F1**'e ulaştı. ResNet50'nin 0,9913'üyle arasında
sadece 0,018 fark var — yani transfer learning'in marjinal katkısı bu görevde
**küçük**, esas sıçrama MLP→CNN geçişinde yaşanıyor.

Bu, dersin temel öğretilerinden birinin canlı kanıtı: **bir görüntü probleminin
%80'i convolution ile çözülür, gerisi mimari ve veri ile rafine edilir**.

### Sürpriz 3: Transfer learning grubunda fark yok

ResNet50, EfficientNetB0 ve ViT-Base arasında macro-F1 farkı **%0,06** —
1500 örnekli test seti için **istatistiksel olarak anlamlı değil**. Wilson
güven aralığı ile %95 güvenle ±%0,5; bu farkın çok altında. Bu da projenin
ana bulgusunu doğruluyor: **mimari değişimi maliyet kararıdır, doğruluk
kararı değil.**

## Pareto: Hangi Model Akıllı Tercih?

Sonuçları doğruluk vs maliyet düzleminde çizdiğimde **EfficientNetB0** Pareto
frontiyerinde yer alıyor: 4 milyon parametre (ResNet'ten 6 kat az), 15,3 MB
dosya, en hızlı eğitim (9,5 dk), neredeyse aynı doğruluk. Production deployment
için optimal seçim.

ViT en pahalı: 86 milyon parametre, 327 MB dosya, 19,5 dakika eğitim. Bu
dataset'in kapasitesini fazlasıyla aşıyor — daha karmaşık görevlerde (ImageNet,
COCO) avantaj sağlar ama burada gereksiz.

## Web Demosu — Lokal İnteraktif Test

Sadece sayılar göstermenin sıkıcı olacağını düşündüğüm için bir **lokal web
uygulaması** yazdım: backend FastAPI, frontend Vite + React + TypeScript +
TailwindCSS. Üç adımdan oluşuyor:

1. **Görsel yükle** — drag-drop veya 10 hazır örnekten birini seç
2. **Model seç** — 5 modelden tek bir tane ya da "tümünü karşılaştır"
3. **Sonuçları gör** — Top-3 tahminler + Grad-CAM/EigenCAM heatmap +
   belirsizlik rozeti

### Belirsizlik Göstergesi — "Bilmiyorum"u Görselleştirmek

Closed-set softmax modellerinin temel sorunu: tahmin etmek **zorundadır**,
"bilmiyorum" diyemez. Bunu kullanıcıya görselleştirmek için her tahmin için
**Shannon entropy** ve **top-1/top-2 margin** hesaplıyorum, sonra 3 seviyeli
bir rozet üretiyorum:
- 🟢 **Kesin** — yüksek max prob (≥0,85), geniş margin, düşük entropy
- 🟡 **Şüpheli** — orta seviye güven; top-1 ve top-2 yarışıyor olabilir
- 🔴 **Belirsiz** — düşük max prob veya yüksek entropy

Comparison modunda **3+ model "belirsiz" işaretlerse**, görsel muhtemelen 10
sınıf dışı (out-of-distribution) — banner kırmızı OOD uyarısı veriyor.

### Grad-CAM ve ViT Sürprizi

CNN'ler için klasik **GradCAM** kullandım: son convolution katmandaki gradient
× activation çarpımı, sınıf-özel bir heatmap üretiyor. Modelin "nereye
baktığını" görselleştiriyor.

Ama ViT'te beklenmedik bir sorunla karşılaştım: ViT softmax doygunluğu yapıyor,
top-1 confidence 1,0'a ulaşıyor ve vanilla GradCAM'in gradient'i tam **sıfır**
oluyor → heatmap tamamen siyah. Çözüm olarak **EigenCAM** kullandım: gradient
gerektirmiyor, son block aktivasyonlarının SVD'sini alıyor.

Demoda eğlenceli bir anekdot: bir Roblox görselinde ViT yanlış cevap verdi
("Genshin Impact"). EigenCAM modelin avatarlar yerine **gökyüzüne baktığını**
gösterdi — hata sebebi anında anlaşıldı. Modelin nereye baktığını görmek
debug süreçlerinde inanılmaz değerli.

## Sınırlamalar ve Gelecek Çalışma

Her projenin sınırları var, bunlar benimkiler:

- **Closed-set sınıflandırma** sınır dışı sınıflar için zorunlu tahmin verir.
  Belirsizlik göstergesi bunun üstesinden gelmiyor, sadece görselleştiriyor.
  Gerçek OOD detection için Mahalanobis distance ya da energy-based
  yöntemler eklenebilir.
- **10K görsel** sınırı; daha fazla sınıf ve sahne çeşitliliği (menü, loading
  ekranı, farklı modlar) eklemek modelin sağlamlığını artırırdı.
- **Confidence calibration** yapılmadı; model %87 emin olabiliyor ve
  yanılabiliyor. Temperature scaling veya focal loss bu durumu iyileştirir.
- **Multi-label** genişletme: bir görselde birden fazla oyun ögesi bulunması
  durumu (overlay, picture-in-picture vb.).
- **Mobil deploy**: EfficientNet → ONNX → mobil cihaz dönüşümü ile gerçek
  zamanlı edge inference yapılabilir.

## Modeli Kendin Dene

Tüm kod ve modeller GitHub'da:

```
github.com/MAliTopkara/YZM304-Derin-Ogrenme/tree/main/proje4
```

Lokal demoyu çalıştırmak için:

```bash
# 1. Bağımlılıklar
pip install -r requirements.txt

# 2. Modelleri eğit (her biri ~10-20 dk, bir kerelik)
python -m src.train --model mlp
python -m src.train --model cnn_scratch --amp
python -m src.train --model resnet50 --amp
python -m src.train --model efficientnet_b0 --amp
python -m src.train --model vit_base --amp

# 3. Backend (terminal 1)
cd webapp/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload

# 4. Frontend (terminal 2)
cd webapp/frontend
npm install
npm run dev
```

Tarayıcıda `http://localhost:5173` aç, oyun ekran görüntüsü yükle, 5 modelin
yargısını yan yana gör.

Detaylı dokümantasyon `proje4/README.md` ve `webapp/README.md` dosyalarında.

## Kapanış

Bu proje bana şunları öğretti:

1. **Mimari fark çoğu zaman maliyet farkıdır.** Bu görevde 0,42M parametreli
   CNN Scratch ile 86M parametreli ViT arasında macro-F1 farkı sadece 0,018.
   Doğruluk değil, verimlilik kararı önemli.
2. **Transfer learning'in en büyük katkısı erken epoch'larda görülür** —
   eğitim hızı ve veri ihtiyacı açısından. Yeterli veri ve zaman varsa
   sıfırdan eğitilen CNN bile yüksek doğruluğa ulaşır.
3. **Confidence ≠ Correctness** — model %87 emin olabilir ve yine de
   yanılabilir. Belirsizlik göstergesi bunu kullanıcıya anlık olarak
   göstermenin pratik bir yolu.
4. **Görselleştirme her zaman değerli.** Grad-CAM olmasaydı ViT'in neden
   yanıldığını asla anlayamazdım.

Bu projede en çok keyif aldığım kısım, sonuçların **beklediğimden farklı**
çıkmasıydı. CNN Scratch'in transfer learning ile bu kadar yarışacağını
öngörmüyordum. Bu, derin öğrenmenin bana sürekli hatırlattığı bir ders:
**hipotez kurmak için sayılar gerekli, sayılar için deney gerekli, deney
için kod gerekli**. Sonuçlar geldikten sonra hikâye yeniden yazılabilir.

Soru, geri bildirim ve iletişim için GitHub Issues açıktır.

---

**Mehmet Ali Topkara** · 23291093 · Ankara Üniversitesi · Yapay Zeka ve Veri Mühendisliği · 2025–2026 Bahar
