/**
 * Sayfalar arası state — UploadedImage ve seçilen model.
 *
 * Küçük uygulama olduğu için React Context yeterli; Redux/Zustand overkill.
 */
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import type { ModelName, UploadedImage } from "./types";

interface AppStateContextValue {
  uploaded: UploadedImage | null;
  setUploaded: (u: UploadedImage | null) => void;
  selectedModel: ModelName | "all" | null;
  setSelectedModel: (m: ModelName | "all" | null) => void;
  reset: () => void;
}

const AppStateContext = createContext<AppStateContextValue | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [uploaded, setUploadedRaw] = useState<UploadedImage | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelName | "all" | null>(null);

  const setUploaded = useCallback(
    (u: UploadedImage | null) => {
      setUploadedRaw((prev) => {
        // önceki preview URL'sini revoke et, memory leak önlemek için
        if (prev && prev.previewUrl.startsWith("blob:")) {
          URL.revokeObjectURL(prev.previewUrl);
        }
        return u;
      });
    },
    [],
  );

  const reset = useCallback(() => {
    setUploaded(null);
    setSelectedModel(null);
  }, [setUploaded]);

  const value = useMemo<AppStateContextValue>(
    () => ({ uploaded, setUploaded, selectedModel, setSelectedModel, reset }),
    [uploaded, setUploaded, selectedModel, reset],
  );

  return (
    <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>
  );
}

export function useAppState(): AppStateContextValue {
  const ctx = useContext(AppStateContext);
  if (!ctx) {
    throw new Error("useAppState must be used within AppStateProvider");
  }
  return ctx;
}
