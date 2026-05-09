/** UI'da gösterilecek zengin model meta bilgisi. metrics.csv'den senkronize tutulur.
 *
 * 5 model — 2 baseline + 3 transfer learning. Sıralama pedagojik akışı yansıtır:
 * en zayıftan en güçlüye.
 *
 * Kaynak: results/metrics.csv (test set sonuçları, AMP eğitimi).
 */
import type { ModelName } from "./types";

export type ModelCategory = "baseline" | "transfer";

export interface ModelMeta {
  name: ModelName;
  display: string;
  paradigm: string;
  category: ModelCategory;
  /** 1-2 cümlelik açıklama (kart altında). */
  blurb: string;
  /** test set accuracy (0..1). */
  testAccuracy: number;
  /** test set macro-F1 (0..1) — birincil metriğimiz. */
  macroF1: number;
  /** parametre sayısı, milyon cinsinden. */
  paramsM: number;
  /** weights .pth dosyasının boyutu, MB. */
  sizeMB: number;
  /** batched inference süresi (ms/örnek), evaluate.py'den. */
  inferenceMs: number;
  /** AMP ile eğitim süresi, dk. */
  trainingTimeMin: number;
  /** Heatmap (Grad-CAM/EigenCAM) üretilebiliyor mu? MLP için false. */
  hasHeatmap: boolean;
  /** Tailwind accent rengi (kart vurgusu). */
  accent: "slate" | "amber" | "blue" | "emerald" | "rose";
}

export const MODEL_META: Record<ModelName, ModelMeta> = {
  mlp: {
    name: "mlp",
    display: "MLP",
    paradigm: "Tam Bağlı Baseline",
    category: "baseline",
    blurb:
      "Tam bağlı (fully-connected) sinir ağı. Görüntüyü düz vektöre çevirir, uzamsal yapıyı yok sayar.",
    testAccuracy: 0.5027,
    macroF1: 0.4794,
    paramsM: 38.6,
    sizeMB: 147.1,
    inferenceMs: 9.83,
    trainingTimeMin: 11.8,
    hasHeatmap: false,
    accent: "slate",
  },
  cnn_scratch: {
    name: "cnn_scratch",
    display: "CNN (Scratch)",
    paradigm: "CNN — Sıfırdan",
    category: "baseline",
    blurb:
      "Sıfırdan eğitilen klasik CNN (VGG-mini). Convolution öğreniyor — 0.42M parametreyle bile baseline'ı aşıyor.",
    testAccuracy: 0.9733,
    macroF1: 0.9733,
    paramsM: 0.42,
    sizeMB: 1.6,
    inferenceMs: 9.13,
    trainingTimeMin: 11.1,
    hasHeatmap: true,
    accent: "amber",
  },
  resnet50: {
    name: "resnet50",
    display: "ResNet50",
    paradigm: "Klasik CNN — Transfer",
    category: "transfer",
    blurb:
      "Residual bağlantılarla derin CNN. 2015'in sembolü, hâlâ güçlü bir baseline.",
    testAccuracy: 0.9913,
    macroF1: 0.9913,
    paramsM: 23.5,
    sizeMB: 89.8,
    inferenceMs: 9.29,
    trainingTimeMin: 10.6,
    hasHeatmap: true,
    accent: "blue",
  },
  efficientnet_b0: {
    name: "efficientnet_b0",
    display: "EfficientNetB0",
    paradigm: "Modern CNN — Transfer",
    category: "transfer",
    blurb:
      "Compound scaling + MBConv blokları. Boyut/doğruluk dengesinde Pareto-optimal.",
    testAccuracy: 0.9907,
    macroF1: 0.9907,
    paramsM: 4.0,
    sizeMB: 15.3,
    inferenceMs: 9.99,
    trainingTimeMin: 9.5,
    hasHeatmap: true,
    accent: "emerald",
  },
  vit_base: {
    name: "vit_base",
    display: "ViT-Base/16",
    paradigm: "Transformer — Transfer",
    category: "transfer",
    blurb:
      "Vision Transformer: görüntüyü 16×16 patch token'larına ayırır. Daha az inductive bias.",
    testAccuracy: 0.9907,
    macroF1: 0.9907,
    paramsM: 85.8,
    sizeMB: 327.3,
    inferenceMs: 12.01,
    trainingTimeMin: 19.5,
    hasHeatmap: true,
    accent: "rose",
  },
};

/** Pedagojik sıralama: en zayıftan en güçlüye. */
export const MODEL_ORDER: ModelName[] = [
  "mlp",
  "cnn_scratch",
  "resnet50",
  "efficientnet_b0",
  "vit_base",
];

export const BASELINE_MODELS: ModelName[] = ["mlp", "cnn_scratch"];
export const TRANSFER_MODELS: ModelName[] = [
  "resnet50",
  "efficientnet_b0",
  "vit_base",
];

/** Tailwind class isimlerini accent'a göre seç. */
export function accentClasses(accent: ModelMeta["accent"]) {
  return {
    slate: {
      ring: "ring-slate-200",
      iconBg: "bg-slate-100 text-slate-700",
      badge: "bg-slate-50 text-slate-700 border-slate-200",
      hoverBorder: "hover:border-slate-400",
      selectedBorder: "border-slate-500 ring-2 ring-slate-200",
      bar: "bg-slate-500",
    },
    amber: {
      ring: "ring-amber-200",
      iconBg: "bg-amber-100 text-amber-700",
      badge: "bg-amber-50 text-amber-700 border-amber-200",
      hoverBorder: "hover:border-amber-400",
      selectedBorder: "border-amber-500 ring-2 ring-amber-200",
      bar: "bg-amber-500",
    },
    blue: {
      ring: "ring-blue-200",
      iconBg: "bg-blue-100 text-blue-700",
      badge: "bg-blue-50 text-blue-700 border-blue-200",
      hoverBorder: "hover:border-blue-400",
      selectedBorder: "border-blue-500 ring-2 ring-blue-200",
      bar: "bg-blue-500",
    },
    emerald: {
      ring: "ring-emerald-200",
      iconBg: "bg-emerald-100 text-emerald-700",
      badge: "bg-emerald-50 text-emerald-700 border-emerald-200",
      hoverBorder: "hover:border-emerald-400",
      selectedBorder: "border-emerald-500 ring-2 ring-emerald-200",
      bar: "bg-emerald-500",
    },
    rose: {
      ring: "ring-rose-200",
      iconBg: "bg-rose-100 text-rose-700",
      badge: "bg-rose-50 text-rose-700 border-rose-200",
      hoverBorder: "hover:border-rose-400",
      selectedBorder: "border-rose-500 ring-2 ring-rose-200",
      bar: "bg-rose-500",
    },
  }[accent];
}
