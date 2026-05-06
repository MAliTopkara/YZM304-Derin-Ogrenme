/** Tek model kartı — paradigma, parametre, accuracy, hız bilgileri. */
import { AlertTriangle, Check, Cpu, Layers, Timer, Zap } from "lucide-react";

import { accentClasses, type ModelMeta } from "../lib/modelMeta";
import { formatPercent } from "../lib/utils";
import { cn } from "../lib/utils";

interface ModelCardProps {
  meta: ModelMeta;
  trained: boolean;
  selected?: boolean;
  onSelect: () => void;
}

const ICONS: Record<ModelMeta["accent"], typeof Cpu> = {
  blue: Layers,
  emerald: Zap,
  rose: Cpu,
};

export function ModelCard({ meta, trained, selected, onSelect }: ModelCardProps) {
  const Icon = ICONS[meta.accent];
  const classes = accentClasses(meta.accent);

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={!trained}
      className={cn(
        "group relative text-left rounded-2xl border bg-white p-5 shadow-sm transition-all",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2",
        trained && classes.hoverBorder,
        trained && "hover:shadow-lg hover:-translate-y-0.5",
        !trained && "opacity-70 cursor-not-allowed",
        selected ? classes.selectedBorder : "border-slate-200",
      )}
    >
      {selected && (
        <div className="absolute -top-2 -right-2 size-7 rounded-full bg-emerald-500 text-white flex items-center justify-center shadow-md">
          <Check className="size-4" />
        </div>
      )}

      <div className="flex items-start gap-3">
        <div className={cn("size-12 rounded-xl flex items-center justify-center", classes.iconBg)}>
          <Icon className="size-6" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 flex-wrap">
            <h3 className="font-semibold text-slate-800 truncate">{meta.display}</h3>
            <span className={cn("text-xs px-2 py-0.5 rounded-full border", classes.badge)}>
              {meta.paradigm}
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-1 leading-snug">{meta.blurb}</p>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <Metric label="Test Acc" value={formatPercent(meta.testAccuracy, 2)} primary />
        <Metric label="Parametre" value={`${meta.paramsM.toFixed(1)}M`} />
        <Metric label="Boyut" value={`${meta.sizeMB.toFixed(0)} MB`} />
        <Metric
          label="Inference"
          value={`${meta.inferenceMs.toFixed(1)} ms`}
          icon={<Timer className="size-3.5 text-slate-400" />}
        />
      </div>

      {!trained && (
        <div className="mt-3 flex items-center gap-2 rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-xs text-amber-800">
          <AlertTriangle className="size-4 flex-shrink-0" />
          Bu model henüz eğitilmedi (results/models/{meta.name}.pth bulunamadı).
        </div>
      )}
    </button>
  );
}

function Metric({
  label,
  value,
  primary,
  icon,
}: {
  label: string;
  value: string;
  primary?: boolean;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col">
      <div className="text-[11px] uppercase tracking-wide text-slate-400 flex items-center gap-1">
        {icon}
        {label}
      </div>
      <div className={cn("font-semibold text-slate-800", primary && "text-base")}>{value}</div>
    </div>
  );
}
