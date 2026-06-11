# Kemik Röntgen Görüntülerinde Kırık Tespitinde Evrişimsel Sinir Ağı Mimarilerinin Karşılaştırmalı Analizi: Transfer Öğrenme ve Açıklanabilir Yapay Zeka Yaklaşımı

**Mehmet Ali Topkara**
Yapay Zeka ve Veri Mühendisliği Bölümü
Ankara Üniversitesi
Ankara, Türkiye
mehmetalitopkara080@gmail.com

---

## Özet (Abstract)

*İskelet-kas sistemi röntgenlerinde kırık tespiti, hem sınıf dengesizliği hem de kırık izlerinin ince ve yerel doğası nedeniyle zorlu bir bilgisayarlı görü problemidir. Bu çalışmada, FracAtlas veri kümesi üzerinde üç farklı evrişimsel sinir ağı (CNN) mimarisini —ResNet-50, DenseNet-121 ve EfficientNet-B0— ortak bir transfer öğrenme çatısında, ortak veri bölmesinde ve ortak metriklerde karşılaştıran kontrollü bir çalışma sunuyoruz. Her mimari üç farklı rastgele tohum (seed) ile eğitilmiş; performans, ikili sınıflandırmada Accuracy, Precision, Recall, F1, makro-F1 ve ROC-AUC ile raporlanmıştır. Bulgularımız, ImageNet ön-eğitimli üç mimarinin doğrulukta birbirine yakın olduğunu, ancak azınlık sınıfı (kırık) başarımının modeller arasındaki asıl ayrımı belirlediğini göstermektedir. ResNet-50 en yüksek F1 (0.677 ± 0.015) ve en yüksek ROC-AUC (0.891 ± 0.011) değerini vermiş; EfficientNet-B0'a karşı F1 üstünlüğü eşli t-testi ile istatistiksel olarak anlamlı bulunmuştur (p = 0.032). EfficientNet-B0 ise ResNet-50'nin yaklaşık 1/6'sı parametre ile rekabetçi doğruluk ve AUC sağlayarak verimlilik ekseninde öne çıkmıştır. Sınıf dengesizliği nedeniyle çalışma noktası (karar eşiği) seçiminin başarımı ciddi biçimde değiştirdiğini; Youden temelli eşik ile kırık geri-çağırma oranının 0.66'dan 0.73'e yükseltilebildiğini gösterdik. Son olarak Grad-CAM tabanlı açıklanabilirlik analizi, en iyi modelin doğru sınıflandırdığı örneklerde kırık bölgesine odaklandığını; hataların ise ince/atipik kırıklarda yoğunlaştığını ortaya koymaktadır.*

**Anahtar Kelimeler—** kırık tespiti, transfer öğrenme, evrişimsel sinir ağları, sınıf dengesizliği, Grad-CAM, açıklanabilir yapay zeka, FracAtlas

---

## I. GİRİŞ

Röntgen (X-ışını) görüntülerinden otomatik kırık tespiti; acil servis iş akışlarının hızlandırılması, radyolog iş yükünün azaltılması ve özellikle yoğun ya da uzman erişiminin kısıtlı olduğu ortamlarda ikinci bir okuyucu (second reader) olarak karar desteği sağlanması açısından aktif bir araştırma alanıdır [1], [8], [9]. Kaçırılan bir kırık (yanlış negatif) klinik olarak yüksek maliyetlidir; bu nedenle problem yalnızca genel doğruluk değil, azınlık sınıfının geri-çağırması (recall) ekseninde de değerlendirilmelidir.

Problemin iki temel zorluğu vardır: **(i) sınıf dengesizliği** — açık veri kümelerinde sağlam (kırıksız) görüntüler kırıklı görüntülere kıyasla çok daha fazladır; FracAtlas'ta kırıklı görüntüler toplamın yalnızca ≈%17.6'sını oluşturur. Bu durum, modelin çoğunluk sınıfına yanlı (biased) hale gelmesine ve yalnızca accuracy ile yapılan değerlendirmenin yanıltıcı olmasına yol açar. **(ii) İz belirsizliği** — kırık izleri çoğu zaman ince çatlaklar, kortikal süreksizlikler veya yer değiştirmeler biçiminde küçük ve yerel yapılardır; bütünsel doku özelliklerinden ayırt edilmeleri güçtür.

Sınırlı boyuttaki medikal veri kümelerinde derin ağların sıfırdan eğitilmesi aşırı öğrenmeye (overfitting) yol açtığından, ImageNet üzerinde ön-eğitilmiş ağların ince ayarı (transfer learning) medikal görüntülemede baskın yaklaşımdır [11], [17]. Ancak "hangi CNN mimarisi kırık tespitinde daha iyidir?" sorusu, aynı veri bölmesi, aynı ön-işleme ve aynı eğitim protokolü altında adil biçimde nadiren karşılaştırılır; literatürdeki sonuçlar çoğu zaman farklı protokollerden geldiği için doğrudan kıyaslanamaz [15].

Bu çalışmada, **tek ve sabit bir transfer öğrenme çatısında** üç temsilci CNN mimarisini — güçlü ve klasik bir temel olan **ResNet-50** [2], medikal görüntülemede sık tercih edilen yoğun-bağlantılı **DenseNet-121** [3] ve parametre/hız verimliliğiyle öne çıkan **EfficientNet-B0** [4] — karşılaştırıyoruz. Her mimari üç rastgele tohum ile eğitilerek sonuçların kararlılığı (ortalama ± standart sapma) raporlanmış ve modeller arası farklar istatistiksel testlerle (eşli t-testi, McNemar, DeLong) sınanmıştır. Katkılarımız: **(a)** üç mimarinin tamamen kontrollü ve çok-tohumlu adil bir karşılaştırması; **(b)** sınıf dengesizliğine yönelik çalışma noktası (karar eşiği) analizi ile precision–recall ödünleşiminin klinik açıdan yorumlanması; **(c)** Grad-CAM tabanlı açıklanabilirlik analizi ile modelin gerçekten kırık bölgesine odaklanıp odaklanmadığının nitel değerlendirmesi.

---

## II. İLGİLİ ÇALIŞMALAR

**Derin öğrenme ile kırık tespiti.** Lindsey ve ark. [8], el bileği röntgenlerinde derin öğrenmenin klinisyenlerin kırık tespit doğruluğunu artırdığını geniş bir çalışmada göstermiştir. Kim ve MacKinnon [9], transfer öğrenmenin sınırlı röntgen verisinde sıfırdan eğitime kıyasla daha iyi kırık sınıflandırması sağladığını bildirmiştir. Rajpurkar ve ark. [7] MURA veri kümesi ile büyük ölçekli iskelet-kas anormalliği tespitini standardize etmiştir. Bu çalışmalar, ön-eğitimli CNN'lerin kırık tespitinde güçlü temeller sunduğunu vurgular.

**Mimari temeller.** ResNet [2], artık (residual) bağlantılarla çok derin ağların eğitimini olanaklı kılarak görüntü sınıflandırmada standart bir temel haline gelmiştir. DenseNet [3], her katmanı önceki tüm katmanlara bağlayarak öznitelik yeniden kullanımını artırır ve daha az parametreyle güçlü gradyan akışı sağlar; bu özellik tıbbi görüntülerde sık tercih edilmesinin nedenidir. EfficientNet [4], derinlik–genişlik–çözünürlük boyutlarını birlikte (compound scaling) ölçekleyerek çok daha az parametre ve hesapla rekabetçi doğruluk elde eder.

**Sınıf dengesizliği.** Buda ve ark. [15], CNN'lerde sınıf dengesizliğinin başarımı ciddi biçimde düşürdüğünü ve yeniden örnekleme (resampling) ile maliyet-duyarlı öğrenmenin etkili karşı-önlemler olduğunu göstermiştir. Bizim çalışmamızda ağırlıklı örnekleyici (weighted sampler) ve azınlık sınıfına duyarlı metrikler (F1, makro-F1, recall) bu literatürle uyumludur.

**Açıklanabilirlik.** Selvaraju ve ark. [6] Grad-CAM yöntemini önererek bir CNN'in kararını verirken görüntünün hangi bölgelerine odaklandığını gradyan-ağırlıklı sınıf etkinleştirme haritalarıyla görselleştirmeyi sağlamıştır. Medikal görüntülemede bu, modelin klinik olarak anlamlı bölgelere bakıp bakmadığını doğrulamak için kritik bir araçtır [17].

**Değerlendirme protokolü.** Paplham ve Franc'ın yaş tahmini bağlamında vurguladığı gibi [15], ön-işleme ve protokol farkları sonuçları büyük ölçüde değiştirir; bu nedenle mimari karşılaştırması ancak ortak bir protokol altında anlamlıdır. Bu gözlem, çalışmamızın tüm mimarileri özdeş bölme ve hiperparametrelerle eğitme tasarımının gerekçesidir.

---

## III. YÖNTEM

### A. Veri Kümesi ve Bölünmesi

Çalışmada **FracAtlas** [1] veri kümesi kullanılmıştır: iskelet-kas sistemine ait toplam **4.083** hizalanmış röntgen görüntüsü içerir; bunların **717**'si kırıklı (fractured), **3.366**'sı kırıksız (non-fractured) olarak etiketlidir. Problem, kırık tipini değil **kırık var/yok** (ikili) sınıflandırmasını hedeflemektedir. Veri, sınıf dağılımı korunacak biçimde **katmanlı (stratified)** olarak %70 eğitim / %15 doğrulama / %15 test oranında bölünmüştür (bölme tohumu = 42). Sonuçtaki dağılım Tablo I'de verilmiştir. Veri kümesi belirgin biçimde dengesizdir (kırık oranı ≈ %17.6); bu nedenle değerlendirmede yalnızca accuracy'ye güvenilmemiş, azınlık sınıfına duyarlı metrikler vurgulanmıştır.

**TABLO I**
**VERİ KÜMESİ BÖLÜNMESİ (FRACATLAS, KATMANLI %70/%15/%15)**

| Bölme | Toplam | Kırıksız | Kırıklı | Kırık % |
|---|---|---|---|---|
| Eğitim | 2857 | 2356 | 501 | %17.5 |
| Doğrulama | 613 | 505 | 108 | %17.6 |
| Test | 613 | 505 | 108 | %17.6 |
| **Toplam** | **4083** | **3366** | **717** | **%17.6** |

### B. Mimariler

ImageNet üzerinde ön-eğitilmiş üç CNN, yalnızca sınıflandırma başları (son tam-bağlantılı katman) iki sınıfa göre değiştirilerek kullanılmıştır:

- **ResNet-50** [2] — 23.51M parametre; artık bağlantılı klasik temel.
- **DenseNet-121** [3] — 6.96M parametre; yoğun bağlantılı, öznitelik yeniden kullanımı yüksek.
- **EfficientNet-B0** [4] — 4.01M parametre; bileşik ölçeklemeli, verimlilik odaklı.

Tüm modeller 3 × 224 × 224 girdi alır ve iki logit üretir.

### C. Ön İşleme ve Veri Artırma

Görüntüler 224 × 224 boyutuna ölçeklenmiş ve ImageNet ortalama/standart sapması ile normalize edilmiştir. Medikal görüntülerde aşırı/agresif artırmadan kaçınmak amacıyla yalnızca hafif artırma uygulanmıştır: kenardan boşluklu yeniden boyutlandırma (256 → rastgele 224 kırpma), %50 olasılıkla yatay çevirme, ±10° rastgele döndürme ve hafif parlaklık/kontrast titremesi (±0.1). Sınıf dengesizliğine karşı eğitim yükleyicisinde **ağırlıklı rastgele örnekleyici** (ters-frekans ağırlıklı) kullanılmıştır.

### D. Eğitim Protokolü

İki fazlı bir transfer öğrenme stratejisi uygulanmıştır: **Faz 1** (ilk 2 epoch) yalnızca sınıflandırma başı eğitilirken backbone dondurulur (frozen); **Faz 2**'de backbone açılarak tüm ağ ince ayarlanır (fine-tuning). Backbone ve baş için ayrı öğrenme oranları kullanılır. Tüm modeller özdeş hiperparametrelerle (Tablo II) ve aynı veri bölmesiyle eğitilmiştir. Doğrulama F1'i üzerinde erken durdurma (early stopping, patience = 5) uygulanmış; en iyi doğrulama F1'ini veren ağırlıklar test için saklanmıştır. Donanım: GPU; karışık duyarlık (mixed precision) etkin.

**TABLO II**
**ORTAK HİPERPARAMETRELER**

| Hiperparametre | Değer |
|---|---|
| Optimizatör | AdamW (ağırlık sönümü 10⁻⁴) [10] |
| Öğrenme oranı — baş / backbone | 10⁻³ / 10⁻⁴ |
| Faz yapısı | 2 epoch dondurulmuş + ≤18 epoch ince ayar |
| Maksimum epoch | 20 |
| Erken durdurma | patience = 5 (doğrulama F1) |
| Batch / Görüntü boyutu | 32 / 224 × 224 |
| Kayıp fonksiyonu | Çapraz entropi |
| Dengeleme | Ağırlıklı rastgele örnekleyici |
| Veri artırma | çevirme, ±10°, parlaklık/kontrast ±0.1 |
| Tohumlar (seed) | 42, 123, 2024 |

### E. Değerlendirme Metrikleri ve Eşik Seçimi

İkili sınıflandırmada pozitif sınıf "kırık" alınmıştır. Karışıklık matrisinin TP, TN, FP, FN bileşenlerinden temel metrikler şöyle tanımlanır:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Sınıf dengesizliği nedeniyle her iki sınıfın F1'inin ortalaması olan **makro-F1** ve eşikten bağımsız ayırt-edicilik ölçüsü olan **ROC-AUC** de raporlanmıştır. Ağırlıklı örnekleyiciyle dengeleme, eğitimde sınıf $c$ için ağırlığı

$$w_c = \frac{N}{K \cdot n_c}$$

biçiminde ters-frekansla tanımlar ($N$: toplam örnek, $K$: sınıf sayısı, $n_c$: sınıf $c$ örnek sayısı).

Klinik bağlamda yanlış negatifin (kaçırılan kırık) maliyeti yüksek olduğundan, varsayılan 0.5 eşiğine ek olarak iki çalışma noktası incelenmiştir: doğrulama setinde **F1'i enbüyükleyen eşik** ($t^\*_{F1}$) ve **Youden indeksini** ($J = \text{Recall} + \text{Specificity} - 1$) enbüyükleyen eşik ($t^\*_{\text{Youden}}$). Tüm eşikler yalnızca doğrulama setinde seçilip test setinde uygulanmıştır.

---

## IV. BULGULAR

### A. Ana Karşılaştırma

Tablo III, üç mimarinin üç tohum üzerinden ortalama ± standart sapma test başarımını (varsayılan eşik $t = 0.5$) verir. Üç mimari **accuracy** açısından birbirine yakındır (≈ %87–89); ancak azınlık sınıfını da dengeli ölçen **F1**, **makro-F1** ve **ROC-AUC**'ta **ResNet-50** açık biçimde öndedir (F1 0.677, makro-F1 0.805, AUC 0.891). EfficientNet-B0 ve DenseNet-121 bu metriklerde geride kalır; aralarındaki fark ise küçüktür.

**TABLO III**
**ANA KARŞILAŞTIRMA — TEST SETİ, ORTALAMA ± STD (3 TOHUM, $t = 0.5$)**

| Mimari | Accuracy | Precision | Recall | F1 | Makro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **ResNet-50** | **0.888 ± 0.003** | **0.688 ± 0.013** | 0.667 ± 0.033 | **0.677 ± 0.015** | **0.805 ± 0.008** | **0.891 ± 0.011** |
| DenseNet-121 | 0.872 ± 0.018 | 0.636 ± 0.076 | **0.676 ± 0.065** | 0.651 ± 0.030 | 0.786 ± 0.020 | 0.857 ± 0.012 |
| EfficientNet-B0 | 0.869 ± 0.011 | 0.635 ± 0.052 | 0.623 ± 0.051 | 0.626 ± 0.005 | 0.773 ± 0.005 | 0.868 ± 0.011 |

En iyi tekil çalışma (ResNet-50, tohum 42) test setinde Accuracy 0.891, F1 0.679 ve ROC-AUC 0.880 vermiştir. ResNet-50'nin düşük standart sapması (Accuracy ± 0.003), bu mimarinin tohumlar arası kararlılık açısından da en güvenilir model olduğunu göstermektedir; DenseNet-121 ise en yüksek değişkenliğe sahiptir (Accuracy ± 0.018, Precision ± 0.076).

### B. İstatistiksel Anlamlılık

Modeller arası farkların gürültü düzeyinde olup olmadığını üç testle sınadık. **Eşli t-testi** (her tohum eşlenik gözlem, F1 @ t=0.5): ResNet-50'nin EfficientNet-B0'a F1 üstünlüğü **anlamlıdır** (t = 5.48, **p = 0.032**); buna karşın ResNet-50 vs DenseNet-121 (p = 0.32) ve DenseNet-121 vs EfficientNet-B0 (p = 0.35) farkları anlamlı değildir. **DeLong testi** (ROC-AUC) ResNet-50'nin DenseNet-121'e AUC üstünlüğünü iki tohumda anlamlı bulmuştur (tohum 42 p = 0.016; tohum 123 p = 0.015). **McNemar testi** ise çoğu çiftte anlamlı fark vermemiştir. Bu sonuç ana gözlemimizi destekler: **ResNet-50 azınlık sınıfı başarımında (F1) en güçlü ve en tutarlı modeldir; EfficientNet-B0 ile arasındaki fark istatistiksel olarak anlamlıdır.**

### C. Çalışma Noktası (Karar Eşiği) Analizi

Sınıf dengesizliği nedeniyle karar eşiği seçimi precision–recall dengesini doğrudan belirler (Tablo IV). Varsayılan $t = 0.5$ precision'ı yükseltirken recall'u sınırlar. **Youden** noktası (ResNet için $t \approx 0.30$) eşiği düşürerek kırık geri-çağırma oranını **0.667 → 0.731**'e yükseltir; bu, kaçırılan kırığın maliyetli olduğu klinik senaryoda tercih edilebilir bir ödünleşimdir (precision bir miktar düşse de). **F1-optimal** nokta ise dengeli bir orta yol sunar. Bu analiz, modelin ham doğruluğundan bağımsız olarak, dağıtım hedefine göre çalışma noktasının ayarlanmasının kritik olduğunu göstermektedir.

**TABLO IV**
**ÇALIŞMA NOKTASI KARŞILAŞTIRMASI — RESNET-50 (3 TOHUM ORTALAMASI)**

| Çalışma noktası | Eşik | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Varsayılan | 0.50 | 0.888 | 0.688 | 0.667 | 0.677 |
| F1-optimal | 0.69 | 0.889 | 0.732 | 0.590 | 0.652 |
| Youden | 0.30 | 0.874 | 0.625 | **0.731** | 0.673 |

### D. Verimlilik

Mimariler doğrulukta birbirine yakınken, asıl fark **verimlilikte** ortaya çıkar (Tablo V). **EfficientNet-B0**, ResNet-50'nin yaklaşık **1/6'sı** parametresiyle (4.01M vs 23.51M) rekabetçi accuracy ve AUC sağlar. DenseNet-121 ise 6.96M parametre ile orta konumdadır. Epoch başına süreler benzerdir (≈ 36–40 s); toplam eğitim süresi erken durdurmanın tetiklediği epoch sayısına bağlı olarak değişir. Bu bulgu, sınırlı kaynak veya gömülü/mobil dağıtım hedefi olan senaryolarda EfficientNet-B0'ın güçlü bir aday olduğunu göstermektedir; en yüksek azınlık-sınıfı başarımı öncelikliyse ResNet-50 tercih edilmelidir.

**TABLO V**
**VERİMLİLİK VE MODEL KARMAŞIKLIĞI (3 ÇALIŞMA ORTALAMASI)**

| Mimari | Param (M) | Epoch süresi (s) | Ort. epoch | Toplam eğitim (s) | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| ResNet-50 | 23.51 | 37.6 | 16.3 | 614 | **0.677** | **0.891** |
| DenseNet-121 | 6.96 | 40.0 | 13.0 | 520 | 0.651 | 0.857 |
| EfficientNet-B0 | **4.01** | 35.9 | 15.7 | 563 | 0.626 | 0.868 |

### E. Hata Analizi (Karışıklık Matrisleri)

Tablo VI, tohum 42 için her mimarinin test karışıklık matrisini özetler. Üç modelde de hataların büyük kısmı **kırığı kaçırma** (yanlış negatif, FN) biçimindedir; bu, dengesiz veride beklenen davranıştır ve azınlık sınıfının recall'unun neden darboğaz olduğunu açıklar. ResNet-50 en düşük FN (37) ile en yüksek kırık geri-çağırmasını sağlarken, DenseNet-121 en yüksek yanlış pozitifi (FP = 49) üretmiştir. Sınıf bazlı raporda kırıksız sınıf yüksek başarımla ayrılırken (F1 ≈ 0.94), kırık sınıfı tüm modeller için zor sınıftır (ResNet için F1 ≈ 0.66) — bu, ince kırık izlerinin görsel belirsizliğiyle uyumludur.

**TABLO VI**
**KARIŞIKLIK MATRİSLERİ — TEST SETİ (TOHUM 42). [TN, FP; FN, TP]**

| Mimari | TN | FP | FN | TP | F1 | AUC |
|---|---|---|---|---|---|---|
| ResNet-50 | 475 | 30 | **37** | **71** | **0.679** | **0.880** |
| DenseNet-121 | 456 | 49 | 38 | 70 | 0.617 | 0.846 |
| EfficientNet-B0 | 468 | 37 | 41 | 67 | 0.632 | 0.869 |

### F. Açıklanabilirlik (Grad-CAM)

En iyi modelin (ResNet-50) kararlarını yorumlamak için son evrişim bloğu (`layer4`) üzerinden Grad-CAM ısı haritaları üretilmiştir. **Doğru** sınıflandırılan kırık örneklerinde ısı haritası tutarlı biçimde **kırık/kortikal süreksizlik bölgesinde** yoğunlaşmakta; modelin klinik olarak anlamlı bölgelere odaklandığını doğrulamaktadır. **Hatalı** örneklerde ise ısı haritası ya dağınıktır ya da kırık dışı yapılara (örn. eklem kenarları, implant/donanım gölgeleri) kaymaktadır. Modelin sistematik kör noktaları ince/çizgisel kırıklar ve atipik görüntülerdir. Bu nitel bulgu, niceliksel hata analiziyle (yüksek FN oranı) tutarlıdır.

### Şekiller

- **Şekil 1.** Üç mimarinin tüm tohumlarda birleşik öğrenme eğrileri (doğrulama F1/AUC) — `outputs/figures/learning_curves_combined.png`
- **Şekil 2.** ROC eğrisi karşılaştırması (tüm tohumlar) — `outputs/figures/roc_comparison_all_seeds.png`
- **Şekil 3.** Metrik kutu grafiği (modeller × tohumlar) — `outputs/figures/metric_boxplot.png`
- **Şekil 4.** Karışıklık matrisleri (üç mimari, tohum 42) — `outputs/figures/confusion_*_seed42.png`
- **Şekil 5.** Grad-CAM: ResNet-50 doğru ve hatalı örnekler — `outputs/figures/gradcam_resnet50_correct.png`, `gradcam_resnet50_errors.png`, `gradcam_all_models_seed42.png`

---

## V. TARTIŞMA

**Mimariler doğrulukta yakın, azınlık başarımında ayrışır.** ImageNet ön-eğitimli üç mimari, orta ölçekli ve dengesiz bu veride benzer bir doğruluk tavanına ulaşmaktadır. Asıl ayrım, dengesizlik altında azınlık sınıfı başarımında (F1/AUC) ortaya çıkar; burada ResNet-50 hem en yüksek hem en kararlı (en düşük standart sapma) modeldir ve EfficientNet-B0'a üstünlüğü istatistiksel olarak anlamlıdır.

**Doğruluk ≠ verimlilik.** En yüksek azınlık başarımı (ResNet-50) ile en yüksek parametre verimliliği (EfficientNet-B0) farklı modellerde toplanmıştır. Dağıtım hedefi doğruluk öncelikliyse ResNet-50, kaynak/hız öncelikliyse EfficientNet-B0 tercih edilmelidir. DenseNet-121 her iki eksende de orta konumdadır ve en yüksek tohumlar-arası değişkenliği gösterir.

**Çalışma noktası tasarımı kritiktir.** Sınıf dengesizliği nedeniyle tek bir 0.5 eşiğiyle raporlama yanıltıcıdır; eşik seçimi, aynı modelden çok farklı precision–recall dengeleri çıkarır. Klinik olarak kaçırılan kırığın maliyeti yüksek olduğundan, recall'u yükselten Youden noktası savunulabilir bir tercihtir.

**Açıklanabilirlik güveni artırır.** Grad-CAM, en iyi modelin doğru kararlarında kırık bölgesine odaklandığını göstererek modelin "doğru nedenle doğru" karar verdiğine dair kanıt sunar; bu, medikal karar desteğinde benimsenme için kritiktir.

**Sınırlılıklar.** (i) Tek veri kümesi (FracAtlas) ve tek dahili bölme protokolü — çapraz-veri genellemesi sınanmamıştır. (ii) Problem kırık var/yok ile sınırlıdır; kırık tipi/konumu ele alınmamıştır. (iii) K-katlamalı çapraz doğrulama yerine sabit bölme + çoklu tohum kullanılmıştır. (iv) Sınıf dengesizliği uçlarda (ince kırıklar) hatayı artırmaktadır; mimari karşılaştırması bu rejimle sınırlıdır.

---

## VI. SONUÇ

FracAtlas üzerinde üç CNN mimarisini (ResNet-50, DenseNet-121, EfficientNet-B0) ortak transfer öğrenme çatısında, çok-tohumlu ve istatistiksel olarak sınanmış biçimde karşılaştıran kontrollü bir çalışma sunduk. Mimariler genel doğrulukta yakındır; ancak dengesiz veride asıl belirleyici olan azınlık (kırık) sınıfı başarımında **ResNet-50 en yüksek F1 (0.677) ve ROC-AUC (0.891) değerini vermiş** ve EfficientNet-B0'a F1 üstünlüğü istatistiksel olarak anlamlı bulunmuştur (p = 0.032). **EfficientNet-B0 ise ≈1/6 parametreyle verimlilik ekseninde öne çıkmıştır.** Karar eşiği analizi, Youden noktasıyla kırık geri-çağırmasının 0.73'e yükseltilebildiğini; Grad-CAM analizi ise en iyi modelin kırık bölgesine odaklandığını göstermiştir. Gelecek çalışmalar: çapraz-veri değerlendirmesi (örn. MURA), kırık tipi/konumu için çok-sınıflı veya tespit tabanlı genişletme, odak (focal) kayıp gibi dengesizliğe özgü kayıplar ve daha güçlü mimarilerin (ConvNeXt, Vision Transformer) eklenmesini içermektedir.

---

## KAYNAKLAR

[1] I. Abedeen, M. A. Rahman, F. Z. Prottyasha, T. Ahmed, T. M. Chowdhury, and S. Shatabda, "FracAtlas: A dataset for fracture classification, localization and segmentation of musculoskeletal radiographs," *Scientific Data*, vol. 10, art. 521, 2023.

[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.

[3] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE CVPR*, 2017, pp. 4700–4708.

[4] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, 2019, pp. 6105–6114.

[5] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE CVPR*, 2009, pp. 248–255.

[6] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626.

[7] P. Rajpurkar et al., "MURA: Large dataset for abnormality detection in musculoskeletal radiographs," in *Proc. Medical Imaging with Deep Learning (MIDL)*, 2018.

[8] R. Lindsey et al., "Deep neural network improves fracture detection by clinicians," *Proc. Natl. Acad. Sci. USA*, vol. 115, no. 45, pp. 11591–11596, 2018.

[9] D. H. Kim and T. MacKinnon, "Artificial intelligence in fracture detection: transfer learning from deep convolutional neural networks," *Clinical Radiology*, vol. 73, no. 5, pp. 439–445, 2018.

[10] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.

[11] N. Tajbakhsh et al., "Convolutional neural networks for medical image analysis: Full training or fine tuning?," *IEEE Trans. Med. Imaging*, vol. 35, no. 5, pp. 1299–1312, 2016.

[12] E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach," *Biometrics*, vol. 44, no. 3, pp. 837–845, 1988.

[13] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.

[14] M. Buda, A. Maki, and M. A. Mazurowski, "A systematic study of the class imbalance problem in convolutional neural networks," *Neural Networks*, vol. 106, pp. 249–259, 2018.

[15] J. Paplham and V. Franc, "A call to reflect on evaluation practices for age estimation: Comparative analysis of the state-of-the-art and a unified benchmark," *arXiv:2307.04570*, 2023.

[16] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," *Journal of Big Data*, vol. 6, art. 60, 2019.

[17] G. Litjens et al., "A survey on deep learning in medical image analysis," *Medical Image Analysis*, vol. 42, pp. 60–88, 2017.
