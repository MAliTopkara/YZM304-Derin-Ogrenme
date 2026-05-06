/** Backend ile paylaşılan tipler. webapp/backend/main.py response şemaları ile birebir. */

export type ModelName = "resnet50" | "efficientnet_b0" | "vit_base";

export interface ModelInfo {
  name: ModelName;
  display_name: string;
  paradigm: string;
  params_m: number;
  trained: boolean;
  weights_path: string | null;
}

export interface Prediction {
  class: string;
  confidence: number;
}

export type ConfidenceLevel = "high" | "medium" | "low";

export interface Uncertainty {
  /** Top-1 olasılığı (predictions[0].confidence ile aynı). */
  max_prob: number;
  /** Top-1 ile top-2 arasındaki güven farkı — küçükse model kararsız. */
  margin: number;
  /** Shannon entropy / log(K). 0 = kesin, 1 = uniform. */
  entropy_normalized: number;
  /** Backend'in heuristic sınıflandırması. */
  level: ConfidenceLevel;
}

export interface PredictResponse {
  model: ModelName;
  display_name: string;
  predictions: Prediction[];
  uncertainty: Uncertainty;
  inference_ms: number;
  size_mb: number;
}

export interface PredictAllItem extends Partial<PredictResponse> {
  model: ModelName;
  display_name: string;
  predictions: Prediction[];
  uncertainty?: Uncertainty;
  /** Eğer ağırlık eksikse error alanı dolu gelir, predictions boş. */
  error?: string;
}

export interface GradCamResponse {
  model: ModelName;
  display_name: string;
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  /** base64-encoded PNG (data URI'ye sarılmadan, ham). */
  overlay_png_b64: string;
  inference_ms: number;
  alpha: number;
}

/** /gradcam/all — eğitilmemiş model için error alanı dolu, overlay_png_b64 boş. */
export interface GradCamAllItem extends GradCamResponse {
  error?: string;
}

/** İki sayfa arası state taşımak için (UploadPage → ModelSelectPage → ResultsPage). */
export interface UploadedImage {
  /** File objesi — gerçek upload için. */
  file: File;
  /** ObjectURL ya da data URL — preview için. */
  previewUrl: string;
}
