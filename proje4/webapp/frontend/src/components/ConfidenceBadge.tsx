/** Modelin tahmin belirsizliğini görsel olarak özetler.
 *
 * Üç seviye:
 *   high   — model emin (yeşil, "Kesin")
 *   medium — kararsız (sarı, "Şüpheli")
 *   low    — muhtemelen OOD görsel (kırmızı, "Belirsiz · listede yok olabilir")
 *
 * Hover/focus durumunda detaylı tooltip: max_prob, margin, entropy.
 */
import { CheckCircle2, HelpCircle, ShieldAlert } from "lucide-react";
import type { ReactElement } from "react";

import type { Uncertainty } from "../lib/types";
import { cn, formatPercent } from "../lib/utils";

interface ConfidenceBadgeProps {
  uncertainty: Uncertainty;
  /** Compact = sadece ikon + kelime, padding küçülür. Default: false. */
  compact?: boolean;
  /** Tooltip detaylarını her zaman alt satırda göster (compact=false default). */
  showDetails?: boolean;
}

const STYLES: Record<
  Uncertainty["level"],
  { label: string; cls: string; icon: ReactElement; hint: string }
> = {
  high: {
    label: "Kesin",
    cls: "bg-emerald-50 text-emerald-700 border-emerald-200",
    icon: <CheckCircle2 className="size-3.5" />,
    hint: "Model çok emin — yüksek olasılık ve geniş margin.",
  },
  medium: {
    label: "Şüpheli",
    cls: "bg-amber-50 text-amber-800 border-amber-200",
    icon: <HelpCircle className="size-3.5" />,
    hint: "Top-1 ile top-2 yakın ya da entropy yüksek — ikinci tahmin de mantıklı olabilir.",
  },
  low: {
    label: "Belirsiz",
    cls: "bg-rose-50 text-rose-700 border-rose-200",
    icon: <ShieldAlert className="size-3.5" />,
    hint: "Düşük max güven veya yüksek entropy — görsel listedeki 10 oyun dışı olabilir (OOD).",
  },
};

export function ConfidenceBadge({
  uncertainty,
  compact = false,
  showDetails = false,
}: ConfidenceBadgeProps) {
  const s = STYLES[uncertainty.level];
  const tooltip =
    `${s.hint}\n` +
    `max prob: ${formatPercent(uncertainty.max_prob, 1)}\n` +
    `margin: ${formatPercent(uncertainty.margin, 1)}\n` +
    `entropy: ${formatPercent(uncertainty.entropy_normalized, 1)}`;

  return (
    <div className="inline-flex flex-col gap-0.5">
      <span
        title={tooltip}
        className={cn(
          "inline-flex items-center gap-1 rounded-full border font-medium",
          s.cls,
          compact ? "px-1.5 py-0 text-[10px]" : "px-2 py-0.5 text-xs",
        )}
      >
        {s.icon}
        {s.label}
      </span>
      {showDetails && (
        <div className="text-[10px] text-slate-500 leading-snug pl-0.5">
          margin {formatPercent(uncertainty.margin, 1)} · entropy{" "}
          {formatPercent(uncertainty.entropy_normalized, 1)}
        </div>
      )}
    </div>
  );
}
