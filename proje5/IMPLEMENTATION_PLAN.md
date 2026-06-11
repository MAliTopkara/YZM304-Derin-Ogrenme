# Implementasyon Plani

## Proje Bilgisi

- Baslik: Kemik Rontgen Goruntulerinde Kirik Tespitinde CNN Mimarilerinin Karsilastirmali Analizi: Transfer Ogrenme ve Aciklanabilir Yapay Zeka Yaklasimi
- Veri kumesi: FracAtlas
- Problem tanimi: `fractured` / `non-fractured` ikili siniflandirma
- Calisma tipi: Prototip duzeyinde, kisa literatur ozeti + orta olcekli deneysel calisma
- Sure kisiti: 1 hafta, gunde 1-2 saat
- Donanim: GPU mevcut

## Ana Hedef

FracAtlas veri kumesi kullanilarak transfer ogrenme tabanli bir siniflandirma sistemi kurulacak ve en az 3 farkli CNN mimarisi ayni deney kosullarinda karsilastirilacaktir. Performans sonuclari temel siniflandirma metrikleri ile raporlanacak, ardindan model kararlarini yorumlamak icin XAI tabanli gorsellestirme eklenecektir.

## Proje Kapsami

### Dahil

- FracAtlas veri kumesinin ikili siniflandirma icin kullanilmasi
- 3 farkli CNN mimarisinin karsilastirilmasi
- Transfer ogrenme kullanimi
- Temel veri on isleme ve veri artirma
- Performans degerlendirmesi: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix analizi
- Grad-CAM veya benzeri XAI analizi
- IEEE IMRAD yapisina uygun yazili rapor

### Dahil Degil

- Cok sinifli kirik tipi siniflandirmasi
- Segmentasyon veya nesne tespiti calismasi
- K-fold cross validation
- Cok sayida hiperparametre optimizasyonu
- Kapsamli benchmark veya buyuk olcekli review makalesi

## Onerilen Modeller

Karsilastirma icin asagidaki mimariler onerilmektedir:

1. ResNet50
2. DenseNet121
3. EfficientNet-B0

Bu secim su nedenle uygundur:

- ResNet50 guclu ve klasik bir baseline verir.
- DenseNet121 medikal goruntulerde sik tercih edilir.
- EfficientNet-B0 hiz ve parametre verimliligi acisindan dengeli bir modeldir.

## Veri Kumesi Notlari

- Toplam goruntu sayisi: 4083
- Kirikli goruntu sayisi: 717
- Kirik olmayan goruntu sayisi: 3366
- Veri dengesizdir, bu nedenle sadece accuracy kullanilmayacaktir.
- Calisma problemi kirik tipi degil, kirik var/yok tespiti olarak tanimlanacaktir.

## Teknik Uygulama Asamalari

### 1. Problem ve Deney Tasariminin Netlestirilmesi

- Calisma problemi ikili siniflandirma olarak sabitlenecek.
- Karsilastirilacak 3 model kesinlestirilecek.
- Deneylerde kullanilacak ortak egitim ayarlari tanimlanacak.

### 2. Veri Hazirlama

- FracAtlas klasor yapisi kullanilarak etiketler hazirlanacak.
- Egitim, dogrulama ve test bolunmesi yapilacak.
- Sinif dengesizligi icin class weight veya benzeri strateji uygulanacak.
- Goruntuler secilen modellere uygun boyuta donusturulecek.

### 3. Veri On Isleme ve Artirma

- Resize
- Normalization
- Yatay/ufak donme tabanli augmentation
- Gerekirse hafif contrast veya brightness degisikligi

Not: Medikal goruntulerde agresif augmentation kullanilmamali.

### 4. Model Kurulumu

- Pretrained agirliklar kullanilacak.
- Son katman ikili siniflandirmaya gore duzenlenecek.
- Ortak optimizer, loss function ve epoch yapisi korunacak.

### 5. Egitim Asamasi

- Her model ayni veri bolmesi ile egitilecek.
- Ayni metrikler kaydedilecek.
- Egitim ve dogrulama kayiplari izlenecek.

### 6. Degerlendirme

- Test seti uzerinde performans raporlanacak.
- Accuracy, Precision, Recall, F1-score ve ROC-AUC hesaplanacak.
- Confusion matrix uretilecek.

### 7. XAI Analizi

- En azindan en iyi model icin Grad-CAM uygulanacak.
- Dogru ve yanlis siniflandirilan orneklerden gorseller alinacak.
- Modelin kirik bolgesine odaklanip odaklanmadigi yorumlanacak.

### 8. Raporlama

- Sonuclar tablo ve sekil olarak duzenlenecek.
- Kisa literatur ozeti eklenecek.
- IEEE IMRAD yapisina uygun metin yazilacak.

## 1 Haftalik Calisma Plani

### Gun 1

- Basligi ve problem tanimini kesinlestir
- Literatur taramasi icin 5-8 temel kaynak belirle
- Kullanilacak 3 modeli sabitle
- Veri kumesini nihai olarak bu problem icin dogrula

Teslim ciktilari:

- Nihai baslik
- Problem tanimi
- Kaynak listesi taslagi

### Gun 2

- Veri hazirlama kodunu kur
- Train/validation/test ayrimi yap
- Dataloader ve augmentation yapisini hazirla
- Class imbalance cozumunu ekle

Teslim ciktilari:

- Calisan veri hazirlama akisi
- Veri dagilimi ozeti

### Gun 3

- 1. modeli egit
- Sonuclari kaydet
- Egitim loglarini duzenli sakla

Teslim ciktilari:

- Ilk model performans tablosu

### Gun 4

- 2. ve 3. modeli egit
- Sonuclari ayni formatta kaydet

Teslim ciktilari:

- 3 model icin temel sonuc tablosu

### Gun 5

- Tum modelleri test setinde karsilastir
- Confusion matrix ve ROC degerlerini olustur
- En iyi modeli belirle

Teslim ciktilari:

- Karsilastirma tablosu
- Grafikler

### Gun 6

- Grad-CAM gorsellerini uret
- 3-5 ornek uzerinden yorum yaz
- Bulgular ve kisitlar bolumunu taslakla

Teslim ciktilari:

- XAI gorselleri
- Sonuc yorumu taslagi

### Gun 7

- IEEE formatinda raporu yaz
- Giris, yontem, sonuclar ve tartisma bolumlerini tamamla
- Son duzenlemeleri yap

Teslim ciktilari:

- Nihai yazili teslim

## Kullanilacak Degerlendirme Metrikleri

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Not: Sinif dengesizligi nedeniyle recall ve F1-score ozellikle vurgulanacaktir.

## Onerilen Dosya/Calisma Ciktilari

- Veri hazirlama notlari
- Egitim kodu
- Model karsilastirma tablosu
- Confusion matrix gorselleri
- ROC egri gorselleri
- Grad-CAM ciktilari
- IEEE formatinda rapor

## IEEE IMRAD Yapisina Gore Yazim Plani

### Introduction

- Problem ve motivasyon
- Literatur ozeti
- Calismanin katkisi

### Methods

- Veri kumesi
- On isleme
- CNN mimarileri
- Transfer ogrenme yontemi
- Degerlendirme metrikleri

### Results

- Performans tablolari
- Confusion matrix
- ROC-AUC sonuclari
- XAI ornekleri

### Discussion

- Hangi model neden daha iyi
- Veri kumesi sinirliliklari
- Yanlis siniflandirma nedenleri

### Conclusion

- Kisa genel ozet
- Gelecek calisma onerileri

## Riskler ve Kontrol Noktalari

### Riskler

- Veri dengesizligi nedeniyle model yanliligi
- Egitim suresinin beklenenden uzun olmasi
- Tum modellerin benzer performans vermesi
- XAI ciktilarinin anlamsiz olmasi

### Kontrol Noktalari

- Gun 2 sonunda veri akisi calisiyor olmali
- Gun 4 sonunda uc modelin egitimi tamamlanmis olmali
- Gun 5 sonunda karsilastirma tablosu hazir olmali
- Gun 7 sonunda yazili teslim tamamlanmis olmali

## Minimum Basari Kriteri

Asagidaki maddeler tamamlanirsa proje basarili ve teslim edilebilir duzeyde kabul edilir:

1. FracAtlas ile ikili siniflandirma problemi kurulmus olmali
2. En az 3 CNN modeli egitilmis olmali
3. Tum modeller icin ayni metriklerle sonuc alinmis olmali
4. En az bir XAI ornegi uretilmis olmali
5. Sonuclar kisa bir IEEE formatli rapora aktarilmis olmali

## Not

Bu plan, tam kapsamli arastirma calismasindan cok, 1 haftalik sureye uygun bir prototip proje icin hazirlanmistir. Gerekirse once minimum basari kriterleri tamamlanacak, ek iyilestirmeler daha sonra eklenecektir.