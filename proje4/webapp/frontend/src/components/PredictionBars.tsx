/** Top-K tahmini animasyonlu güven barları olarak göster. */
import { useEffect, useState } from "react";

import type { Prediction } from "../lib/types";
import { cn, formatPercent } from "../lib/utils";

interface PredictionBarsProps {
  predictions: Prediction[];
  /** En güçlü tahminin barı vurgu rengi. Örn 'bg-blue-500'. Default: brand. */
  accentBar?: string;
  /** Doğru sınıf bilinmiyorsa null. Eşleşen tahmin yeşil yüzük alır. */
  groundTruth?: string | null;
}

export function PredictionBars({
  predictions,
  accentBar = "bg-brand-600",
  groundTruth,
}: PredictionBarsProps) {
  const [animated, setAnimated] = useState(false);

  // mount'ta bir kez animate et
  useEffect(() => {
    const id = requestAnimationFrame(() => setAnimated(true));
    return () => cancelAnimationFrame(id);
  }, [predictions]);

  return (
    <ol className="space-y-2.5">
      {predictions.map((p, i) => {
        const isMatch = groundTruth && p.class === groundTruth;
        return (
          <li key={p.class} className="space-y-1">
            <div className="flex items-baseline justify-between text-sm">
              <span
                className={cn(
                  "font-medium",
                  isMatch ? "text-emerald-700" : "text-slate-800",
                )}
              >
                <span className="text-slate-400 mr-1.5">{i + 1}.</span>
                {p.class}
                {isMatch && (
                  <span className="ml-1.5 text-xs px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                    doğru
                  </span>
                )}
              </span>
              <span className="font-mono text-xs text-slate-700">
                {formatPercent(p.confidence, 2)}
              </span>
            </div>
            <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-[width] duration-700 ease-out",
                  i === 0 ? accentBar : "bg-slate-300",
                )}
                style={{ width: animated ? `${p.confidence * 100}%` : "0%" }}
              />
            </div>
          </li>
        );
      })}
    </ol>
  );
}
