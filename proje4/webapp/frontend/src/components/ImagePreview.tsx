/** Yüklenmiş görselin önizlemesi + "değiştir" + "devam et" eylemleri. */
import { ArrowRight, RotateCcw } from "lucide-react";

import type { UploadedImage } from "../lib/types";
import { cn } from "../lib/utils";

interface ImagePreviewProps {
  image: UploadedImage;
  onReset: () => void;
  onContinue: () => void;
}

export function ImagePreview({ image, onReset, onContinue }: ImagePreviewProps) {
  const sizeKb = (image.file.size / 1024).toFixed(0);
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="grid md:grid-cols-[2fr_1fr] gap-5 items-start">
        <div className="rounded-xl overflow-hidden border border-slate-100 bg-slate-50">
          <img
            src={image.previewUrl}
            alt="Yüklenen görsel"
            className="w-full aspect-[16/9] object-contain bg-slate-900"
          />
        </div>

        <div className="space-y-3">
          <div>
            <h3 className="font-semibold text-slate-800">Hazır.</h3>
            <p className="text-sm text-slate-500 mt-1">
              Dosya{" "}
              <span className="font-mono text-xs text-slate-700">
                {image.file.name}
              </span>{" "}
              ({sizeKb} KB) seçildi.
            </p>
          </div>

          <div className="flex flex-col gap-2 pt-2">
            <button
              type="button"
              onClick={onContinue}
              className={cn(
                "w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl",
                "bg-brand-600 text-white font-medium shadow-md",
                "hover:bg-brand-700 hover:shadow-lg active:scale-[0.99] transition-all",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2",
              )}
            >
              Modeli Seç
              <ArrowRight className="size-4" />
            </button>
            <button
              type="button"
              onClick={onReset}
              className={cn(
                "w-full inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl",
                "bg-white text-slate-700 border border-slate-200",
                "hover:bg-slate-50 active:scale-[0.99] transition-all",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-300",
              )}
            >
              <RotateCcw className="size-4" />
              Başka bir görsel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
