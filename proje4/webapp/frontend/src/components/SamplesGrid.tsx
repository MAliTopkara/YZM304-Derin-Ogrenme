/** Her sınıftan birer örnek görselin grid'i. Tıklayınca seçilir. */
import { useState } from "react";

import { fetchSampleAsFile, SAMPLE_IMAGES, type SampleImage } from "../lib/samples";
import { cn } from "../lib/utils";

interface SamplesGridProps {
  onSelect: (file: File, sample: SampleImage) => void;
  onError: (message: string) => void;
  disabled?: boolean;
}

export function SamplesGrid({ onSelect, onError, disabled }: SamplesGridProps) {
  const [loadingSlug, setLoadingSlug] = useState<string | null>(null);

  const handleClick = async (sample: SampleImage) => {
    if (disabled || loadingSlug) return;
    setLoadingSlug(sample.slug);
    try {
      const file = await fetchSampleAsFile(sample);
      onSelect(file, sample);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingSlug(null);
    }
  };

  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <div className="h-px flex-1 bg-slate-200" />
        <span className="text-sm text-slate-500">veya örneklerden birini dene</span>
        <div className="h-px flex-1 bg-slate-200" />
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {SAMPLE_IMAGES.map((s) => {
          const loading = loadingSlug === s.slug;
          return (
            <button
              key={s.slug}
              type="button"
              onClick={() => handleClick(s)}
              disabled={disabled || !!loadingSlug}
              className={cn(
                "group relative rounded-xl overflow-hidden border border-slate-200 bg-white shadow-sm",
                "hover:shadow-lg hover:border-brand-500 hover:-translate-y-0.5 transition-all",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500",
                (disabled || loadingSlug) && "opacity-60 cursor-not-allowed hover:translate-y-0",
              )}
            >
              <img
                src={s.file}
                alt={s.class}
                className="aspect-[16/9] w-full object-cover"
                loading="lazy"
              />
              <div
                className={cn(
                  "absolute inset-x-0 bottom-0 px-2 py-1.5 text-xs font-medium text-white",
                  "bg-gradient-to-t from-black/70 to-transparent",
                )}
              >
                {s.class}
              </div>
              {loading && (
                <div className="absolute inset-0 bg-white/70 flex items-center justify-center text-xs text-slate-600">
                  Yükleniyor…
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
