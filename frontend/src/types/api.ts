/** Types for the typed parts of the API. Large analytic payloads use narrower local types. */

export interface ImagePanel {
  title: string;
  image: string;
  description?: string;
  stats?: { mean: number; std: number; min: number; max: number; entropy_bits: number };
  histogram?: { intensity: number; count: number; cdf: number }[];
  psnr?: number;
  ssim?: number;
  threshold?: number | null;
  foreground_pct?: number;
  bytes?: number;
}

export interface DatasetInfo {
  id: string;
  name: string;
  domain: string;
  used_by: string[];
  source_url: string;
  homepage?: string;
  license: string;
  access: string;
  synthetic: boolean;
  description: string;
  path: string | null;
  download: string | null;
  required: boolean;
  note: string | null;
  available: boolean;
  downloaded_at: string | null;
  size_mb: number | null;
}

export interface ModelInfo {
  id: string;
  name: string;
  task: string;
  tutorials: string[];
  status: "trained" | "not_trained" | "training" | "pretrained";
  evaluated: boolean;
  path: string | null;
  updated_at: string | null;
  metrics: Record<string, unknown> | null;
  how_to_train: string;
}

export interface Tutorial {
  id: string;
  title: string;
  module: string;
  route: string;
  datasets: string[];
  objective: string;
  concepts: string[];
  inputs: string;
  outputs: string;
  endpoints: string[];
  status: "verified" | "not verified" | "verification failed";
  verification: {
    passed: boolean;
    verified_at: string;
    error: string | null;
    checks_passed: number;
    checks_total: number;
    checks: { check: string; passed: boolean; detail: string }[];
  } | null;
  data_ready: boolean;
  runs: number;
  doc: string;
}

export interface HistoryItem {
  id: number;
  module: string;
  tutorial: string | null;
  title: string;
  dataset: string | null;
  summary: Record<string, unknown>;
  created_at: string;
}

export interface SystemStatus {
  database: boolean;
  datasets: {
    total: number;
    available: number;
    required: number;
    required_available: number;
    items: Pick<DatasetInfo, "id" | "name" | "domain" | "available" | "synthetic" | "required">[];
  };
  models: Pick<ModelInfo, "id" | "name" | "status" | "metrics" | "updated_at">[];
  tutorials: {
    total: number;
    verified: number;
    data_ready: number;
    explored: number;
    items: Pick<Tutorial, "id" | "title" | "module" | "route" | "data_ready" | "runs" | "status">[];
  };
  history: { total_runs: number; recent: HistoryItem[]; predictions: number };
}

export interface Prediction {
  id: number;
  filename: string;
  predicted_label: "NORMAL" | "PNEUMONIA";
  probability_pneumonia: number;
  probabilities: Record<string, number>;
  threshold: number;
  model_name: string;
  image: string;
  gradcam: { overlay: string; heatmap: string; target_class: string; method: string } | null;
  warnings: string[];
  disclaimer: string;
}

export interface ScreeningStatus {
  model_available: boolean;
  model_name?: string;
  threshold?: number;
  img_size?: number;
  trained_at?: string;
  data_source?: string;
  message?: string;
}

export interface BinaryMetrics {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  sensitivity: number;
  specificity: number;
  f1: number;
  roc_auc: number | null;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  support: { normal: number; pneumonia: number };
}

export interface Evaluation {
  checkpoint: string;
  data_source: string;
  split: string;
  samples: number;
  test_loss: number;
  metrics: BinaryMetrics;
  metrics_at_0_5: BinaryMetrics;
  curves: { roc: { fpr: number; tpr: number; threshold: number }[]; pr: { recall: number; precision: number }[] };
  probability_histogram: { bin_start: number; bin_end: number; normal: number; pneumonia: number }[];
  evaluated_at: string;
  model_meta: Record<string, unknown>;
}

export interface TrainingHistory {
  history: {
    epoch: number;
    train_loss: number;
    train_accuracy: number;
    val_loss: number;
    val_accuracy: number;
    val_roc_auc: number;
    val_f1: number;
    lr: number;
    seconds: number;
  }[];
  meta: Record<string, unknown> & { val_metrics_at_threshold?: BinaryMetrics; data_notes?: string[] };
}

export interface SourceOption {
  id: string;
  label: string;
  group?: string;
}
