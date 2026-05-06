/** Results sayfası bekleme state'i — düz spinner yerine yapı önizlemesi.
 *
 * Tek modelle aynı, comparison'da 3 sütun olacak.
 */
import { Loader2 } from "lucide-react";

import { MODEL_ORDER } from "../lib/modelMeta";
import { cn } from "../lib/utils";

interface ResultSkeletonProps {
  variant: "single" | "all";
  originalUrl: string;
}

export function ResultSkeleton({ variant, originalUrl }: ResultSkeletonProps) {
  if (variant === "single") {
    return (
      <div className="grid md:grid-cols-[1.4fr_1fr] gap-5">
        <SkeletonImage url={originalUrl} />
        <SkeletonStatsCard />
      </div>
    );
  }
  return (
    <div className="space-y-5">
      <div className="rounded-xl px-4 py-3 bg-slate-100 border border-slate-200 animate-pulse">
        <div className="h-4 w-2/3 bg-slate-300/70 rounded" />
        <div className="h-3 w-1/3 bg-slate-300/50 rounded mt-2" />
      </div>
      <div className="grid md:grid-cols-3 gap-4">
        {MODEL_ORDER.map((name) => (
          <div
            key={name}
            className="rounded-2xl border border-slate-200 bg-white p-4 space-y-3 shadow-sm"
          >
            <div className="flex items-center gap-2.5">
              <div className="size-9 rounded-lg bg-slate-200 animate-pulse" />
              <div className="flex-1 space-y-1">
                <div className="h-3.5 w-2/3 bg-slate-200 rounded animate-pulse" />
                <div className="h-2.5 w-1/3 bg-slate-200/70 rounded animate-pulse" />
              </div>
            </div>
            <SkeletonImage url={originalUrl} compact />
            <div className="space-y-1.5">
              <div className="h-3 w-1/2 bg-slate-200 rounded animate-pulse" />
              {[0, 1, 2].map((i) => (
                <div key={i} className="space-y-1">
                  <div className="flex justify-between">
                    <div className="h-2.5 w-1/3 bg-slate-200 rounded animate-pulse" />
                    <div className="h-2.5 w-12 bg-slate-200/70 rounded animate-pulse" />
                  </div>
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-slate-200 animate-pulse"
                      style={{ width: `${[60, 30, 10][i]}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SkeletonImage({ url, compact }: { url: string; compact?: boolean }) {
  return (
    <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-900 aspect-[16/9]">
      <img
        src={url}
        alt="Görsel"
        className="absolute inset-0 w-full h-full object-contain opacity-70"
      />
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
      <div
        className={cn(
          "absolute inset-0 flex flex-col items-center justify-center text-white text-xs gap-2",
          "bg-black/40 backdrop-blur-[2px]",
        )}
      >
        <Loader2 className={cn("animate-spin", compact ? "size-5" : "size-7")} />
        <span className="opacity-80">Heatmap üretiliyor…</span>
      </div>
    </div>
  );
}

function SkeletonStatsCard() {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4 shadow-sm">
      <div className="flex items-start gap-3">
        <div className="size-10 rounded-xl bg-slate-200 animate-pulse" />
        <div className="flex-1 space-y-2">
          <div className="h-4 w-2/3 bg-slate-200 rounded animate-pulse" />
          <div className="h-3 w-1/3 bg-slate-200/70 rounded animate-pulse" />
        </div>
      </div>
      <div className="space-y-1.5">
        <div className="h-3 w-1/4 bg-slate-200 rounded animate-pulse" />
        <div className="h-7 w-2/3 bg-slate-200 rounded animate-pulse" />
      </div>
      <div className="space-y-2.5">
        {[0, 1, 2].map((i) => (
          <div key={i} className="space-y-1">
            <div className="flex justify-between">
              <div className="h-3 w-1/3 bg-slate-200 rounded animate-pulse" />
              <div className="h-3 w-12 bg-slate-200/70 rounded animate-pulse" />
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-slate-200 animate-pulse"
                style={{ width: `${[80, 50, 20][i]}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
