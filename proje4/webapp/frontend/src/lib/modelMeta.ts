/** UI'da gösterilecek zengin model meta bilgisi. metrics.csv'den senkronize tutulur.
 *
 * Kaynak: results/metrics.csv (test set sonuçları, AMP eğitimi, Hafta 2 sonu).
 */
import type { ModelName } from "./types";

export interface ModelMeta {
  name: ModelName;
  display: string;
  paradigm: string;
  /** 1-2 cümlelik açıklama (kart altında). */
  blurb: string;
  /** test set accuracy (0..1). */
  testAccuracy: number;
  /** parametre sayısı, milyon cinsinden. */
  paramsM: number;
  /** weights .pth dosyasının boyutu, MB. */
  sizeMB: number;
  /** batched inference süresi (ms/örnek), evaluate.py'den. */
  inferenceMs: number;
  /** AMP ile eğitim süresi, dk. */
  trainingTimeMin: number;
  /** Tailwind accent rengi (kart vurgusu). */
  accent: "blue" | "emerald" | "rose";
}

export const MODEL_META: Record<ModelName, ModelMeta> = {
  resnet50: {
    name: "resnet50",
    display: "ResNet50",
    paradigm: "Klasik CNN",
    blurb:
      "Residual bağlantılarla derin CNN. 2015'in sembolü, hâlâ güçlü bir baseline.",
    testAccuracy: 0.9913,
    paramsM: 23.5,
    sizeMB: 89.8,
    inferenceMs: 9.29,
    trainingTimeMin: 10.6,
    accent: "blue",
  },
  efficientnet_b0: {
    name: "efficientnet_b0",
    display: "EfficientNetB0",
    paradigm: "Modern CNN",
    blurb:
      "Compound scaling + MBConv blokları. Boyut/doğruluk dengesinde Pareto-optimal.",
    testAccuracy: 0.9907,
    paramsM: 4.0,
    sizeMB: 15.3,
    inferenceMs: 9.99,
    trainingTimeMin: 9.5,
    accent: "emerald",
  },
  vit_base: {
    name: "vit_base",
    display: "ViT-Base/16",
    paradigm: "Transformer",
    blurb:
      "Vision Transformer: görüntüyü 16×16 patch token'larına ayırır. Daha az inductive bias.",
    testAccuracy: 0.9907,
    paramsM: 85.8,
    sizeMB: 327.3,
    inferenceMs: 12.01,
    trainingTimeMin: 19.5,
    accent: "rose",
  },
};

export const MODEL_ORDER: ModelName[] = ["resnet50", "efficientnet_b0", "vit_base"];

/** Tailwind class isimlerini accent'a göre seç. */
export function accentClasses(accent: ModelMeta["accent"]) {
  return {
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
