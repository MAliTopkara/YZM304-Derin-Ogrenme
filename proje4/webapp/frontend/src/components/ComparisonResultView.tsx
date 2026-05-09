/** Karşılaştırma modu — 5 modelin sonucu yan yana.
 *
 * Düzen: 2 satır
 *   - Üst: Baseline (MLP, CNN scratch) — 2 sütun
 *   - Alt:  Transfer Learning (ResNet, EffNet, ViT) — 3 sütun
 *
 * Bu yapı pedagojik akışı yansıtır: "zayıf'tan güçlü'ye, transfer learning'in
 * değerini somutlaştır".
 */
import { Loader2, Timer } from "lucide-react";

import { ConfidenceBadge } from "./ConfidenceBadge";
import { GradCamView, pngB64ToDataUrl } from "./GradCamView";
import { PredictionBars } from "./PredictionBars";
import {
  accentClasses,
  BASELINE_MODELS,
  MODEL_META,
  MODEL_ORDER,
  TRANSFER_MODELS,
} from "../lib/modelMeta";
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
  const predictMap: Partial<Record<ModelName, PredictAllItem>> = Object.fromEntries(
    (predictAll ?? []).map((p) => [p.model, p]),
  );
  const gradMap: Partial<Record<ModelName, GradCamAllItem>> = Object.fromEntries(
    (gradcamAll ?? []).map((g) => [g.model, g]),
  );

  // Tahmin uyumu — yalnızca trained ve hata almamış modelleri say
  const validPredictions = MODEL_ORDER.map((n) => predictMap[n]?.predictions?.[0]?.class).filter(
    Boolean,
  ) as string[];
  const top1Set = new Set(validPredictions);
  const totalModels = MODEL_ORDER.length;
  const allAgree =
    top1Set.size === 1 && validPredictions.length === totalModels;

  const levels: ConfidenceLevel[] = MODEL_ORDER.map(
    (n) => predictMap[n]?.uncertainty?.level,
  ).filter(Boolean) as ConfidenceLevel[];
  const lowCount = levels.filter((l) => l === "low").length;
  const highCount = levels.filter((l) => l === "high").length;

  // Banner mantığı 5 modele genelleştirildi:
  //   - 3+ model "low"  → alert (gerçek OOD sinyali)
  //   - tümü hemfikir + tüm transfer modeller "high" → ok
  //   - hemfikir ama yumuşak → warn
  //   - çelişki → warn
  let banner: { tone: "ok" | "warn" | "alert"; line1: string; line2: string };

  if (lowCount >= 3) {
    banner = {
      tone: "alert",
      line1: "Görsel listedeki 10 sınıfa benzemiyor olabilir (OOD).",
      line2: `${lowCount}/${totalModels} model düşük güven veriyor.`,
    };
  } else if (allAgree && highCount >= TRANSFER_MODELS.length) {
    // tüm transfer modelleri "high" güvende ve hepsi aynı sınıfta birleşiyor
    banner = {
      tone: "ok",
      line1: `Tüm modeller "${[...top1Set][0]}" diyor.`,
      line2: "Yüksek güven, geniş margin — kesin konsensüs (baseline + transfer).",
    };
  } else if (allAgree) {
    banner = {
      tone: "warn",
      line1: `Tüm modeller "${[...top1Set][0]}" diyor.`,
      line2: "Konsensüs var ama bazıları şüpheli.",
    };
  } else {
    banner = {
      tone: "warn",
      line1: `Modeller anlaşamadı: ${[...top1Set].slice(0, 5).join(", ")}.`,
      line2:
        "Baseline'lar ile transfer learning farklı yorumlamış olabilir — alttaki kartlara bak.",
    };
  }

  return (
    <div className="space-y-5">
      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
          {error}
        </div>
      )}

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

      {/* Baseline grubu */}
      <ModelGroup
        title="Baseline · Sıfırdan Eğitim"
        names={BASELINE_MODELS}
        cols={2}
        predictMap={predictMap}
        gradMap={gradMap}
        loading={loading}
        originalUrl={originalUrl}
      />

      {/* Transfer grubu */}
      <ModelGroup
        title="Transfer Learning · ImageNet Pretrained"
        names={TRANSFER_MODELS}
        cols={3}
        predictMap={predictMap}
        gradMap={gradMap}
        loading={loading}
        originalUrl={originalUrl}
      />
    </div>
  );
}

interface ModelGroupProps {
  title: string;
  names: ModelName[];
  cols: 2 | 3;
  predictMap: Partial<Record<ModelName, PredictAllItem>>;
  gradMap: Partial<Record<ModelName, GradCamAllItem>>;
  loading: boolean;
  originalUrl: string;
}

function ModelGroup({
  title,
  names,
  cols,
  predictMap,
  gradMap,
  loading,
  originalUrl,
}: ModelGroupProps) {
  return (
    <section className="space-y-2">
      <div className="flex items-center gap-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          {title}
        </h2>
        <div className="h-px flex-1 bg-slate-200" />
      </div>
      <div
        className={cn(
          "grid gap-4",
          cols === 2 ? "md:grid-cols-2" : "md:grid-cols-3",
        )}
      >
        {names.map((name) => (
          <ResultCard
            key={name}
            name={name}
            pred={predictMap[name]}
            grad={gradMap[name]}
            loading={loading}
            originalUrl={originalUrl}
          />
        ))}
      </div>
    </section>
  );
}

interface ResultCardProps {
  name: ModelName;
  pred?: PredictAllItem;
  grad?: GradCamAllItem;
  loading: boolean;
  originalUrl: string;
}

function ResultCard({ name, pred, grad, loading, originalUrl }: ResultCardProps) {
  const meta = MODEL_META[name];
  const accent = accentClasses(meta.accent);
  const top1 = pred?.predictions?.[0];
  const overlayUrl = meta.hasHeatmap ? pngB64ToDataUrl(grad?.overlay_png_b64 ?? null) : null;
  const errMsg = pred?.error ?? (grad?.error && meta.hasHeatmap ? grad.error : undefined);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-3 shadow-sm flex flex-col">
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
          <h3 className="font-semibold text-slate-800 truncate text-sm">{meta.display}</h3>
          <p className="text-[11px] text-slate-500">{meta.paradigm}</p>
        </div>
        {pred?.uncertainty && <ConfidenceBadge uncertainty={pred.uncertainty} compact />}
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
          {meta.hasHeatmap ? (
            <GradCamView originalUrl={originalUrl} overlayDataUrl={overlayUrl} compact />
          ) : (
            <NoHeatmapPlaceholder originalUrl={originalUrl} />
          )}

          <div className="space-y-0.5">
            <div className="text-[11px] uppercase tracking-wide text-slate-400">Tahmin</div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-lg font-bold text-slate-900 truncate">{top1.class}</span>
              <span className="text-xs text-slate-500">
                {formatPercent(top1.confidence, 1)}
              </span>
            </div>
          </div>

          <PredictionBars predictions={pred.predictions} accentBar={accent.bar} />

          <div className="text-[11px] text-slate-500 inline-flex items-center gap-1 pt-1 border-t border-slate-100">
            <Timer className="size-3" />
            {pred.inference_ms != null && formatMs(pred.inference_ms)}
          </div>
        </>
      )}
    </div>
  );
}

function NoHeatmapPlaceholder({ originalUrl }: { originalUrl: string }) {
  return (
    <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-900 aspect-[16/9]">
      <img
        src={originalUrl}
        alt="Görsel"
        className="absolute inset-0 w-full h-full object-contain opacity-60"
      />
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="bg-black/70 backdrop-blur px-3 py-1.5 rounded-lg text-white text-[11px]">
          Bu mimari için spatial heatmap üretilemez
        </div>
      </div>
    </div>
  );
}
