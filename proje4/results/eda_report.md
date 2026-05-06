# EDA Raporu — Gameplay Images Dataset

**Toplam görsel:** 10000

**Sınıf sayısı:** 10

**Bozuk dosya:** 0


## Sınıf Dağılımı

| Sınıf | Görsel sayısı |
|---|---|
| Among Us | 1000 |
| Apex Legends | 1000 |
| Fortnite | 1000 |
| Forza Horizon | 1000 |
| Free Fire | 1000 |
| Genshin Impact | 1000 |
| God of War | 1000 |
| Minecraft | 1000 |
| Roblox | 1000 |
| Terraria | 1000 |

## Görsel Boyutları

| Boyut | Adet |
|---|---|
| 640×360 | 10000 |

## Renk Modları

| Mode | Adet |
|---|---|
| RGBA | 2000 |
| RGB | 8000 |

## Uygunluk Kararı

- Toplam ≥ 8000: **True** (10000)
- Tüm sınıflar ≥ 700: **True**
- Sınıflar dengeli (≤50 fark): **True**
- Bozuk dosya oranı: **0.0000%**

### ✅ PROJE UYGUN


> ⚠️ **Not:** Bazı görseller RGBA modunda. Preprocessing pipeline'ında RGB'ye dönüştürülecek (`src/transforms.py` içinde `_to_rgb`).
