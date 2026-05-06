/** 2. adım — kullanıcı tek model ya da "tümünü karşılaştır" seçer. */
import { AlertCircle, ArrowLeft, ArrowRight, Loader2, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ModelCard } from "../components/ModelCard";
import { api, ApiError } from "../lib/api";
import { MODEL_META, MODEL_ORDER } from "../lib/modelMeta";
import { useAppState } from "../lib/state";
import type { ModelInfo, ModelName } from "../lib/types";
import { cn } from "../lib/utils";

export default function ModelSelectPage() {
  const { uploaded, selectedModel, setSelectedModel } = useAppState();
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // upload yoksa Görsel'e geri dön
  useEffect(() => {
    if (!uploaded) {
      navigate("/", { replace: true });
    }
  }, [uploaded, navigate]);

  // /models çağır
  useEffect(() => {
    let cancelled = false;
    api
      .listModels()
      .then((m) => {
        if (!cancelled) setModels(m);
      })
      .catch((e) => {
        if (cancelled) return;
        if (e instanceof ApiError) {
          setError(`API ${e.status}: ${e.detail}`);
        } else {
          setError(
            "Backend'e bağlanılamadı. webapp/backend dizininde " +
              "`python -m uvicorn main:app --reload` çalışıyor mu?",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const trainedMap: Record<ModelName, boolean> = (() => {
    const m: Record<ModelName, boolean> = {
      resnet50: false,
      efficientnet_b0: false,
      vit_base: false,
    };
    models?.forEach((info) => {
      m[info.name as ModelName] = info.trained;
    });
    return m;
  })();

  const allTrained = MODEL_ORDER.every((n) => trainedMap[n]);
  const trainedCount = MODEL_ORDER.filter((n) => trainedMap[n]).length;

  const pickOne = (name: ModelName) => {
    setSelectedModel(name);
    navigate("/results");
  };

  const pickAll = () => {
    setSelectedModel("all");
    navigate("/results");
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => navigate("/")}
          className="text-sm text-slate-500 hover:text-slate-800 inline-flex items-center gap-1"
        >
          <ArrowLeft className="size-4" />
          Görsel
        </button>
        {models && (
          <span className="text-xs text-slate-500">
            {trainedCount}/{MODEL_ORDER.length} model hazır
          </span>
        )}
      </div>

      <div className="text-center space-y-2">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-800 tracking-tight">
          Modeli seç
        </h1>
        <p className="text-slate-600 max-w-2xl mx-auto">
          Tek bir mimariyi denemek için kart'a tıkla; ya da{" "}
          <span className="font-medium text-slate-800">tümünü yan yana karşılaştır</span>.
        </p>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 flex items-start gap-3">
          <AlertCircle className="size-5 text-rose-500 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-rose-800 flex-1">{error}</div>
        </div>
      )}

      {!models && !error && (
        <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center text-slate-500">
          <Loader2 className="mx-auto size-8 animate-spin opacity-50" />
          <p className="text-sm mt-3">Modeller yükleniyor…</p>
        </div>
      )}

      {models && (
        <>
          {/* Tümünü karşılaştır CTA — kart üstünde, projenin asıl tezi */}
          <button
            type="button"
            onClick={pickAll}
            disabled={!allTrained}
            className={cn(
              "group w-full rounded-2xl border-2 p-5 transition-all flex items-center gap-4",
              "focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2",
              allTrained
                ? "border-brand-500 bg-gradient-to-br from-brand-50 to-white hover:shadow-lg hover:from-brand-100"
                : "border-slate-200 bg-slate-50 opacity-70 cursor-not-allowed",
              selectedModel === "all" && "ring-2 ring-brand-300",
            )}
          >
            <div className="size-14 rounded-2xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center text-white shadow-md">
              <Sparkles className="size-7" />
            </div>
            <div className="flex-1 text-left">
              <div className="font-semibold text-slate-800 text-lg">Tümünü Karşılaştır</div>
              <div className="text-sm text-slate-600 mt-0.5">
                3 model aynı görseli yargılar — tahminler ve Grad-CAM'ler yan yana.
                Projenin asıl tezini sergiler.
              </div>
            </div>
            <ArrowRight className="size-5 text-brand-600 group-hover:translate-x-0.5 transition-transform" />
          </button>

          <div className="grid md:grid-cols-3 gap-4">
            {MODEL_ORDER.map((name) => (
              <ModelCard
                key={name}
                meta={MODEL_META[name]}
                trained={trainedMap[name]}
                selected={selectedModel === name}
                onSelect={() => pickOne(name)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
