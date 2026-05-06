/** Karşılaştırma modu — 3 modelin sonucu yan yana. */
import { Loader2, Timer } from "lucide-react";

import { ConfidenceBadge } from "./ConfidenceBadge";
import { GradCamView, pngB64ToDataUrl } from "./GradCamView";
import { PredictionBars } from "./PredictionBars";
import { accentClasses, MODEL_META, MODEL_ORDER } from "../lib/modelMeta";
import type {
  ConfidenceLevel,
  GradCamAllItem,
  ModelName,
  PredictAllItem,
} from "../lib/types";
import { cn, formatMs, formatPercent } from "../lib/utils";

interface ComparisonResultViewProps {
  originalUrl: string;
  predictAll: PredictAllItem[] | null;
  gradcamAll: GradCamAllItem[] | null;
  loading: boolean;
  error: string | null;
}

export function ComparisonResultView({
  originalUrl,
  predictAll,
  gradcamAll,
  loading,
  error,
}: ComparisonResultViewProps) {
  // model adına göre dict'e indekle
  const predictMap: Partial<Record<ModelName, PredictAllItem>> = Object.fromEntries(
    (predictAll ?? []).map((p) => [p.model, p]),
  );
  const gradMap: Partial<Record<ModelName, GradCamAllItem>> = Object.fromEntries(
    (gradcamAll ?? []).map((g) => [g.model, g]),
  );

  // herkes kabul ettiyse tek tahmin, çelişki varsa farklılığı vurgula
  const top1Set = new Set(
    MODEL_ORDER.map((n) => predictMap[n]?.predictions?.[0]?.class).filter(Boolean),
  );
  const allAgree = top1Set.size === 1 && predictAll && predictAll.length === 3;

  // Toplu belirsizlik analizi.
  // Alarm (kırmızı OOD) sadece gerçek "low" sinyaller varken tetiklenir;
  // tek bir "medium" + çelişki, OOD demek için yeterli değil.
  const levels: ConfidenceLevel[] = MODEL_ORDER.map(
    (n) => predictMap[n]?.uncertainty?.level,
  ).filter(Boolean) as ConfidenceLevel[];
  const lowCount = levels.filter((l) => l === "low").length;
  const highCount = levels.filter((l) => l === "high").length;

  let banner: { tone: "ok" | "warn" | "alert"; line1: string; line2: string };

  if (lowCount >= 2) {
    banner = {
      tone: "alert",
      line1: "Görsel listedeki 10 sınıfa benzemiyor olabilir (OOD).",
      line2: `${lowCount}/3 model düşük güven veriyor — gerçek sınıf burada bulunmuyor olabilir.`,
    };
  } else if (allAgree && highCount === 3) {
    banner = {
      tone: "ok",
      line1: `3 model de "${[...top1Set][0]}" diyor.`,
      line2: "Yüksek güven, geniş margin — kesin konsensüs.",
    };
  } else if (allAgree) {
    banner = {
      tone: "warn",
      line1: `3 model de "${[...top1Set][0]}" diyor.`,
      line2: "Konsensüs var ama en az bir model şüpheli — yine de tahmin makul.",
    };
  } else {
    const certaintyDesc =
      highCount === 3
        ? "üçü de kendinden emin"
        : highCount === 0
          ? "hepsi şüpheli"
          : "bazıları şüpheli";
    banner = {
      tone: "warn",
      line1: `Modeller anlaşamadı: ${[...top1Set].join(", ")}.`,
      line2: `${certaintyDesc} — biri yanılıyor olmalı (gerçek sınıf bunlardan biri).`,
    };
  }

  return (
    <div className="space-y-5">
      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
          {error}
        </div>
      )}

      {/* Konsensüs / OOD banner'ı */}
      {predictAll && (
        <div
          className={cn(
            "rounded-xl px-4 py-3 text-sm",
            banner.tone === "ok" &&
              "bg-emerald-50 border border-emerald-200 text-emerald-800",
            banner.tone === "warn" &&
              "bg-amber-50 border border-amber-200 text-amber-800",
            banner.tone === "alert" &&
              "bg-rose-50 border border-rose-200 text-rose-800",
          )}
        >
          <div className="font-medium">{banner.line1}</div>
          <div className="text-xs opacity-80 mt-0.5">{banner.line2}</div>
        </div>
      )}

      <div className="grid md:grid-cols-3 gap-4">
        {MODEL_ORDER.map((name) => {
          const meta = MODEL_META[name];
          const accent = accentClasses(meta.accent);
          const pred = predictMap[name];
          const grad = gradMap[name];
          const top1 = pred?.predictions?.[0];
          const overlayUrl = pngB64ToDataUrl(grad?.overlay_png_b64 ?? null);
          const errMsg = pred?.error ?? grad?.error;

          return (
            <div
              key={name}
              className="rounded-2xl border border-slate-200 bg-white p-4 space-y-3 shadow-sm flex flex-col"
            >
              <header className="flex items-center gap-2.5">
                <div
                  className={cn(
                    "size-9 rounded-lg flex items-center justify-center flex-shrink-0",
                    accent.iconBg,
                  )}
                >
                  <span className="font-bold text-sm">{meta.display.charAt(0)}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-slate-800 truncate text-sm">
                    {meta.display}
                  </h3>
                  <p className="text-[11px] text-slate-500">{meta.paradigm}</p>
                </div>
                {pred?.uncertainty && (
                  <ConfidenceBadge uncertainty={pred.uncertainty} compact />
                )}
              </header>

              {errMsg && (
                <div className="rounded bg-rose-50 border border-rose-200 px-2 py-1.5 text-xs text-rose-800">
                  {errMsg}
                </div>
              )}

              {loading && !pred && (
                <div className="py-8 flex items-center justify-center text-slate-500 text-xs gap-2">
                  <Loader2 className="size-3.5 animate-spin" />
                  Çalışıyor…
                </div>
              )}

              {pred && top1 && (
                <>
                  <GradCamView originalUrl={originalUrl} overlayDataUrl={overlayUrl} compact />
                  <div className="space-y-0.5">
                    <div className="text-[11px] uppercase tracking-wide text-slate-400">Tahmin</div>
                    <div className="flex items-baseline gap-1.5">
                      <span className="text-lg font-bold text-slate-900 truncate">{top1.class}</span>
                      <span className="text-xs text-slate-500">
                        {formatPercent(top1.confidence, 1)}
                      </span>
                    </div>
                  </div>

                  <PredictionBars
                    predictions={pred.predictions}
                    accentBar={accent.bar}
                  />

                  <div className="text-[11px] text-slate-500 inline-flex items-center gap-1 pt-1 border-t border-slate-100">
                    <Timer className="size-3" />
                    {pred.inference_ms != null && formatMs(pred.inference_ms)}
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
