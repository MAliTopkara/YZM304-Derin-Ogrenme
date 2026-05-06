/**
 * FastAPI istemcisi — webapp/backend/main.py endpoint'lerinin tipli wrapper'ı.
 *
 * Vite dev server'da `/api/*` istekleri 127.0.0.1:8000'e proxy'leniyor (vite.config.ts).
 * Production build için VITE_API_BASE env değişkeni kullanılır.
 */
import type {
  GradCamAllItem,
  GradCamResponse,
  ModelInfo,
  ModelName,
  PredictAllItem,
  PredictResponse,
} from "./types";

const API_BASE: string = import.meta.env.VITE_API_BASE ?? "/api";

class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* body wasn't JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

function makeFormData(file: File): FormData {
  const fd = new FormData();
  fd.append("file", file, file.name);
  return fd;
}

export const api = {
  health: async (): Promise<{ status: string }> => {
    const res = await fetch(`${API_BASE}/health`);
    return handle(res);
  },

  listModels: async (): Promise<ModelInfo[]> => {
    const res = await fetch(`${API_BASE}/models`);
    return handle(res);
  },

  predict: async (
    file: File,
    model: ModelName,
    topK = 3,
  ): Promise<PredictResponse> => {
    const res = await fetch(
      `${API_BASE}/predict?model=${encodeURIComponent(model)}&top_k=${topK}`,
      { method: "POST", body: makeFormData(file) },
    );
    return handle(res);
  },

  predictAll: async (file: File, topK = 3): Promise<PredictAllItem[]> => {
    const res = await fetch(`${API_BASE}/predict/all?top_k=${topK}`, {
      method: "POST",
      body: makeFormData(file),
    });
    return handle(res);
  },

  gradcam: async (
    file: File,
    model: ModelName,
    alpha = 0.45,
  ): Promise<GradCamResponse> => {
    const res = await fetch(
      `${API_BASE}/gradcam?model=${encodeURIComponent(model)}&alpha=${alpha}`,
      { method: "POST", body: makeFormData(file) },
    );
    return handle(res);
  },

  gradcamAll: async (file: File, alpha = 0.45): Promise<GradCamAllItem[]> => {
    const res = await fetch(`${API_BASE}/gradcam/all?alpha=${alpha}`, {
      method: "POST",
      body: makeFormData(file),
    });
    return handle(res);
  },
};

export { ApiError };
