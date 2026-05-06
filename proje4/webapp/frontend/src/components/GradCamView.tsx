/** Orijinal görsel + Grad-CAM overlay. Toggle ile yan yana ya da blend modu. */
import { Eye, EyeOff, Image as ImageIcon, Info } from "lucide-react";
import { useState } from "react";

import { cn } from "../lib/utils";

interface GradCamViewProps {
  /** Orijinal görselin URL'si (object URL ya da public path). */
  originalUrl: string;
  /** Heatmap overlay PNG'si (server'dan base64 — data URI verilebilir). */
  overlayDataUrl: string | null;
  /** İlk render'da ne göstersin: overlay (default) ya da original. */
  initialMode?: "overlay" | "original";
  /** Compact mod — comparison view'da daha küçük başlık. */
  compact?: boolean;
}

export function GradCamView({
  originalUrl,
  overlayDataUrl,
  initialMode = "overlay",
  compact,
}: GradCamViewProps) {
  const [mode, setMode] = useState<"overlay" | "original">(initialMode);

  const showOverlay = mode === "overlay" && !!overlayDataUrl;

  return (
    <div className="space-y-2">
      <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-900 aspect-[16/9]">
        {/* base layer always */}
        <img
          src={originalUrl}
          alt="Görsel"
          className="absolute inset-0 w-full h-full object-contain"
        />
        {/* overlay layer — opacity geçişi ile cross-fade */}
        {overlayDataUrl && (
          <img
            src={overlayDataUrl}
            alt="Grad-CAM heatmap"
            className={cn(
              "absolute inset-0 w-full h-full object-contain transition-opacity duration-300",
              showOverlay ? "opacity-100" : "opacity-0",
            )}
          />
        )}

        {/* mode toggle — overlay'i göster/gizle */}
        {overlayDataUrl && (
          <div className="absolute bottom-2 right-2 flex gap-1 bg-black/60 backdrop-blur rounded-lg p-1">
            <button
              type="button"
              onClick={() => setMode("original")}
              className={cn(
                "px-2 py-1 text-xs rounded inline-flex items-center gap-1.5 transition-colors",
                mode === "original" ? "bg-white text-slate-800" : "text-white/80 hover:text-white",
              )}
              title="Orijinal görsel"
            >
              <ImageIcon className="size-3.5" />
              {!compact && "Orijinal"}
            </button>
            <button
              type="button"
              onClick={() => setMode("overlay")}
              className={cn(
                "px-2 py-1 text-xs rounded inline-flex items-center gap-1.5 transition-colors",
                mode === "overlay" ? "bg-white text-slate-800" : "text-white/80 hover:text-white",
              )}
              title="Grad-CAM heatmap"
            >
              {mode === "overlay" ? (
                <Eye className="size-3.5" />
              ) : (
                <EyeOff className="size-3.5" />
              )}
              {!compact && "Heatmap"}
            </button>
          </div>
        )}

        {/* heatmap rengi legend'i — sol-alt */}
        {overlayDataUrl && showOverlay && !compact && (
          <div
            className="absolute bottom-2 left-2 flex items-center gap-1.5 bg-black/60 backdrop-blur rounded-lg px-2 py-1 text-[11px] text-white/90"
            title="Heatmap: kırmızı bölgeler modelin kararına en çok katkıda bulundu, mavi bölgeler az."
          >
            <Info className="size-3" />
            <span>az</span>
            <div
              className="h-2 w-12 rounded-full"
              style={{
                background:
                  "linear-gradient(to right, #2563eb, #06b6d4, #84cc16, #facc15, #ef4444)",
              }}
            />
            <span>çok</span>
          </div>
        )}
      </div>
    </div>
  );
}

/** Backend'in base64 PNG'sini browser-yenebilir data URI'ye çevirir. */
export function pngB64ToDataUrl(b64: string | null | undefined): string | null {
  if (!b64) return null;
  return `data:image/png;base64,${b64}`;
}
