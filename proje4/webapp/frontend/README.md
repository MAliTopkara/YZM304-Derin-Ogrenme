# Gameplay Classifier — Frontend (Vite + React + TypeScript + Tailwind v4)

Bu dizin, FastAPI backend'in (`webapp/backend/`) önyüzü. 3 modeli (ResNet50,
EfficientNetB0, ViT-Base/16) yan yana karşılaştıran demo uygulama.

## Çalıştırma

```bash
# 1) Backend ayrı bir terminalde
cd ../backend
python -m uvicorn main:app --reload   # http://127.0.0.1:8000

# 2) Frontend (bu dizinde)
npm install                            # ilk kurulum
npm run dev                            # http://localhost:5173
```

Vite dev server `/api/*` isteklerini `127.0.0.1:8000`'e proxy'ler
(`vite.config.ts`), CORS sorunu yok.

## Yapı

```
src/
├── App.tsx                    # router (3 sayfa + 404 redirect)
├── main.tsx                   # BrowserRouter + AppStateProvider
├── index.css                  # tailwindcss + tema
├── components/
│   └── AppShell.tsx           # üst-bar + step indicator
├── pages/
│   ├── UploadPage.tsx         # 1. adım: görsel yükleme    (Görev 5)
│   ├── ModelSelectPage.tsx    # 2. adım: model seçimi      (Görev 6)
│   └── ResultsPage.tsx        # 3. adım: tahmin + Grad-CAM (Görev 7)
└── lib/
    ├── api.ts                 # tipli FastAPI istemcisi
    ├── types.ts               # paylaşılan TS tipleri (backend ile aynı)
    ├── state.tsx              # AppStateProvider (uploaded image, model)
    └── utils.ts               # cn(), formatPercent(), formatMs()
```

## Stack

- **Vite 8** + **React 19** + **TypeScript 5.x**
- **Tailwind CSS v4** (`@tailwindcss/vite` plugin, config-less)
- **react-router-dom v7** — 3 sayfa client-side routing
- **lucide-react** — ikonlar
- **clsx** + **tailwind-merge** — `cn()` helper

## Production Build

```bash
npm run build           # tsc -b && vite build → dist/
npm run preview         # dist'i serve et
```

## Notlar

- Tüm sayfa state'i `useAppState()` üzerinden paylaşılır (Context API).
  Yeni sayfa eklerken `state.tsx`'a alan ekle.
- API base path `/api` (Vite proxy üzerinden). Production için
  `VITE_API_BASE` env değişkenini ayarla.
- `ApiError` sınıfı: `status` + `detail` taşır, sayfa katmanında
  user-friendly mesaja çevirilir.
