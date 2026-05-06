/** Drag-drop + click-to-select görsel yükleme alanı. */
import { ImagePlus, UploadCloud } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { cn } from "../lib/utils";

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
const ACCEPTED_LABEL = "PNG, JPG veya WebP";
const MAX_BYTES = 5 * 1024 * 1024; // 5 MB

interface DropZoneProps {
  onFile: (file: File) => void;
  onError: (message: string) => void;
  disabled?: boolean;
}

export function DropZone({ onFile, onError, disabled }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File | null | undefined) => {
      if (!file) return;
      if (!ACCEPTED_TYPES.includes(file.type)) {
        onError(`Geçersiz dosya tipi (${file.type || "?"}). ${ACCEPTED_LABEL} bekleniyor.`);
        return;
      }
      if (file.size > MAX_BYTES) {
        const mb = (file.size / 1024 / 1024).toFixed(1);
        onError(`Dosya çok büyük (${mb} MB). Maksimum 5 MB.`);
        return;
      }
      onFile(file);
    },
    [onFile, onError],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      handleFile(e.dataTransfer.files?.[0]);
    },
    [handleFile, disabled],
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLDivElement>) => {
      if (disabled) return;
      const item = Array.from(e.clipboardData.items).find((it) =>
        it.type.startsWith("image/"),
      );
      if (item) {
        e.preventDefault();
        handleFile(item.getAsFile());
      }
    },
    [handleFile, disabled],
  );

  return (
    <div
      onDragEnter={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onPaste={handlePaste}
      onClick={() => !disabled && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !disabled) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      className={cn(
        "relative rounded-2xl border-2 border-dashed p-12 transition-all cursor-pointer",
        "bg-white/60 hover:bg-white/80 hover:border-brand-500 hover:shadow-md",
        "flex flex-col items-center justify-center gap-3 text-center",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500",
        isDragging && "border-brand-500 bg-brand-50/80 shadow-lg scale-[1.01]",
        !isDragging && "border-slate-300",
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="sr-only"
        onChange={(e) => handleFile(e.target.files?.[0])}
        disabled={disabled}
      />

      <div
        className={cn(
          "size-16 rounded-2xl flex items-center justify-center transition-colors",
          isDragging
            ? "bg-brand-500 text-white"
            : "bg-brand-100 text-brand-600",
        )}
      >
        {isDragging ? (
          <ImagePlus className="size-8" />
        ) : (
          <UploadCloud className="size-8" />
        )}
      </div>

      <div className="space-y-1">
        <p className="text-base font-medium text-slate-800">
          {isDragging
            ? "Bırakmak için serbest bırak"
            : "Görseli buraya sürükle ya da tıklayıp seç"}
        </p>
        <p className="text-sm text-slate-500">
          {ACCEPTED_LABEL} · maksimum 5 MB
        </p>
        <p className="text-xs text-slate-400">İpucu: Ctrl/Cmd+V ile de yapıştırabilirsin</p>
      </div>
    </div>
  );
}
