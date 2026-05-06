/** 1. adım — kullanıcı görsel yükler ya da örneklerden birini seçer. */
import { AlertCircle } from "lucide-react";
import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";

import { DropZone } from "../components/DropZone";
import { ImagePreview } from "../components/ImagePreview";
import { SamplesGrid } from "../components/SamplesGrid";
import { useAppState } from "../lib/state";

export default function UploadPage() {
  const { uploaded, setUploaded } = useAppState();
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const accept = useCallback(
    (file: File) => {
      setError(null);
      setUploaded({ file, previewUrl: URL.createObjectURL(file) });
    },
    [setUploaded],
  );

  const reset = useCallback(() => {
    setUploaded(null);
    setError(null);
  }, [setUploaded]);

  return (
    <div className="space-y-8 animate-fade-in">
      <div className="text-center space-y-2">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-800 tracking-tight">
          Bir oyun ekran görüntüsü yükle
        </h1>
        <p className="text-slate-600 max-w-2xl mx-auto">
          Eğittiğimiz 3 modelden hangisinin hangi oyunu daha iyi tanıdığını yan
          yana karşılaştır. <span className="text-slate-500">10 sınıf:</span>{" "}
          <span className="text-slate-700">
            Among Us, Apex Legends, Fortnite, Forza Horizon, Free Fire, Genshin
            Impact, God of War, Minecraft, Roblox, Terraria.
          </span>
        </p>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 flex items-start gap-3">
          <AlertCircle className="size-5 text-rose-500 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-rose-800">{error}</div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="ml-auto text-rose-500 hover:text-rose-700 text-sm"
          >
            Kapat
          </button>
        </div>
      )}

      {uploaded ? (
        <ImagePreview
          image={uploaded}
          onReset={reset}
          onContinue={() => navigate("/model")}
        />
      ) : (
        <DropZone onFile={accept} onError={setError} />
      )}

      {!uploaded && (
        <SamplesGrid onSelect={(f) => accept(f)} onError={setError} />
      )}
    </div>
  );
}
