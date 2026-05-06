/** Tek modelin sonuç paneli — Grad-CAM solda, Top-3 + meta sağda. */
import { Loader2, Timer } from "lucide-react";

import { ConfidenceBadge } from "./ConfidenceBadge";
import { GradCamView, pngB64ToDataUrl } from "./GradCamView";
import { PredictionBars } from "./PredictionBars";
import { accentClasses, MODEL_META } from "../lib/modelMeta";
import type { GradCamResponse, ModelName, PredictResponse } from "../lib/types";
import { cn, formatPercent, formatMs } from "../lib/utils";

interface SingleResultViewProps {
  model: ModelName;
  originalUrl: string;
  /** /predict response (predictions + inference + size). */
  predict: PredictResponse | null;
  /** /gradcam response (overlay + predicted class). */
  gradcam: GradCamResponse | null;
  loading: boolean;
  error: string | null;
}

export function SingleResultView({
  model,
  originalUrl,
  predict,
  gradcam,
  loading,
  error,
}: SingleResultViewProps) {
  const meta = MODEL_META[model];
  const accent = accentClasses(meta.accent);
  const overlayDataUrl = pngB64ToDataUrl(gradcam?.overlay_png_b64);
  const top1 = predict?.predictions[0];

  return (
    <div className="grid md:grid-cols-[1.4fr_1fr] gap-5">
      {/* Sol — görsel + heatmap */}
      <div className="space-y-3">
        <GradCamView originalUrl={originalUrl} overlayDataUrl={overlayDataUrl} />
        {gradcam && (
          <p className="text-xs text-slate-500 px-1">
            {model === "vit_base"
              ? "ViT için EigenCAM kullanılır (gradient-free, attention'a dayalı)."
              : "GradCAM — modelin son conv katmanındaki gradient'lerden üretilir."}
          </p>
        )}
      </div>

      {/* Sağ — top-3 + meta */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4 shadow-sm">
        <header className="flex items-start gap-3">
          <div
            className={cn(
              "size-10 rounded-xl flex items-center justify-center flex-shrink-0",
              accent.iconBg,
            )}
          >
            <span className="font-bold">
              {meta.display.charAt(0)}
            </span>
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="font-semibold text-slate-800 truncate">{meta.display}</h2>
            <p className="text-xs text-slate-500">{meta.paradigm}</p>
          </div>
          {top1 && (
            <span className={cn("text-xs px-2 py-0.5 rounded-full border", accent.badge)}>
              {formatPercent(top1.confidence, 1)}
            </span>
          )}
        </header>

        {error && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 px-3 py-2 text-sm text-rose-800">
            {error}
          </div>
        )}

        {loading && !predict && (
          <div className="py-6 flex items-center justify-center text-slate-500 text-sm gap-2">
            <Loader2 className="size-4 animate-spin" />
            Model çalışıyor…
          </div>
        )}

        {predict && top1 && (
          <>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="text-xs uppercase tracking-wide text-slate-400">Tahmin</div>
                {predict.uncertainty && (
                  <ConfidenceBadge uncertainty={predict.uncertainty} showDetails />
                )}
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-slate-900">{top1.class}</span>
                <span className="text-sm text-slate-500">
                  {formatPercent(top1.confidence, 2)} güven
                </span>
              </div>
            </div>

            <div>
              <div className="text-xs uppercase tracking-wide text-slate-400 mb-2">Top-3</div>
              <PredictionBars
                predictions={predict.predictions}
                accentBar={accent.bar}
              />
            </div>

            <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-100">
              <Stat
                icon={<Timer className="size-3.5" />}
                label="Inference"
                value={formatMs(predict.inference_ms)}
              />
              <Stat label="Boyut" value={`${predict.size_mb.toFixed(0)} MB`} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon?: React.ReactNode;
}) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-slate-400 inline-flex items-center gap-1">
        {icon}
        {label}
      </div>
      <div className="text-sm font-semibold text-slate-800">{value}</div>
    </div>
  );
}
