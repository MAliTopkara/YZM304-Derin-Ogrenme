/** 3. adım — modelin (veya 3 modelin) sonucu. */
import { ArrowLeft, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ComparisonResultView } from "../components/ComparisonResultView";
import { ResultSkeleton } from "../components/ResultSkeleton";
import { SingleResultView } from "../components/SingleResultView";
import { api, ApiError } from "../lib/api";
import { useAppState } from "../lib/state";
import type {
  GradCamAllItem,
  GradCamResponse,
  ModelName,
  PredictAllItem,
  PredictResponse,
} from "../lib/types";
import { cn } from "../lib/utils";

export default function ResultsPage() {
  const { uploaded, selectedModel, reset } = useAppState();
  const navigate = useNavigate();

  // single-mode state
  const [predict, setPredict] = useState<PredictResponse | null>(null);
  const [gradcam, setGradcam] = useState<GradCamResponse | null>(null);

  // all-mode state
  const [predictAll, setPredictAll] = useState<PredictAllItem[] | null>(null);
  const [gradcamAll, setGradcamAll] = useState<GradCamAllItem[] | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // upload veya seçim yoksa geri yönlendir
  useEffect(() => {
    if (!uploaded) {
      navigate("/", { replace: true });
      return;
    }
    if (!selectedModel) {
      navigate("/model", { replace: true });
    }
  }, [uploaded, selectedModel, navigate]);

  // model + upload değişince istekleri tetikle
  useEffect(() => {
    if (!uploaded || !selectedModel) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPredict(null);
    setGradcam(null);
    setPredictAll(null);
    setGradcamAll(null);

    const handleErr = (e: unknown) => {
      if (cancelled) return;
      if (e instanceof ApiError) {
        setError(`API ${e.status}: ${e.detail}`);
      } else {
        setError(
          "Backend'e bağlanılamadı — webapp/backend'de uvicorn çalışıyor mu?",
        );
      }
    };

    if (selectedModel === "all") {
      Promise.all([
        api.predictAll(uploaded.file, 3),
        api.gradcamAll(uploaded.file, 0.45),
      ])
        .then(([pa, ga]) => {
          if (cancelled) return;
          setPredictAll(pa);
          setGradcamAll(ga);
        })
        .catch(handleErr)
        .finally(() => !cancelled && setLoading(false));
    } else {
      const name = selectedModel as ModelName;
      Promise.all([
        api.predict(uploaded.file, name, 3),
        api.gradcam(uploaded.file, name, 0.45),
      ])
        .then(([p, g]) => {
          if (cancelled) return;
          setPredict(p);
          setGradcam(g);
        })
        .catch(handleErr)
        .finally(() => !cancelled && setLoading(false));
    }

    return () => {
      cancelled = true;
    };
  }, [uploaded, selectedModel]);

  if (!uploaded || !selectedModel) return null;

  const isAll = selectedModel === "all";
  const showSkeleton =
    loading &&
    ((!isAll && !predict && !gradcam) ||
      (isAll && !predictAll && !gradcamAll));

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => navigate("/model")}
          className="text-sm text-slate-500 hover:text-slate-800 inline-flex items-center gap-1"
        >
          <ArrowLeft className="size-4" />
          Model
        </button>
        <button
          type="button"
          onClick={() => {
            reset();
            navigate("/");
          }}
          className={cn(
            "inline-flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg",
            "bg-white border border-slate-200 text-slate-700",
            "hover:bg-slate-50 hover:border-slate-300 transition-colors",
          )}
        >
          <RefreshCw className="size-3.5" />
          Yeni görsel
        </button>
      </div>

      <div className="text-center space-y-1">
        <h1 className="text-2xl md:text-3xl font-bold text-slate-800 tracking-tight">
          {isAll ? "3 modelin yargısı" : "Sonuç"}
        </h1>
        <p className="text-sm text-slate-500">
          {isAll
            ? "Aynı görsel, üç farklı mimari — tahminler ve Grad-CAM'ler yan yana."
            : "Top-3 tahminler ve modelin nereye baktığı (Grad-CAM)."}
        </p>
      </div>

      {showSkeleton ? (
        <ResultSkeleton
          variant={isAll ? "all" : "single"}
          originalUrl={uploaded.previewUrl}
        />
      ) : isAll ? (
        <ComparisonResultView
          originalUrl={uploaded.previewUrl}
          predictAll={predictAll}
          gradcamAll={gradcamAll}
          loading={loading}
          error={error}
        />
      ) : (
        <SingleResultView
          model={selectedModel as ModelName}
          originalUrl={uploaded.previewUrl}
          predict={predict}
          gradcam={gradcam}
          loading={loading}
          error={error}
        />
      )}
    </div>
  );
}
