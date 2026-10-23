interface ReliabilityBin {
  confidence: number;
  accuracy: number;
}

interface CalibrationSummary {
  ece: number;
  mce: number;
  brier: number;
  reliability: ReliabilityBin[];
}

interface DecisionCurve {
  thresholds: number[];
  net_benefit_uncalibrated: number[];
  net_benefit_recalibrated: number[];
  treat_all: number[];
}

export interface Calibration {
  model?: string;
  n_windows: number;
  n_bins: number;
  loso: CalibrationSummary;
  loso_matched: CalibrationSummary;
  within_subject: CalibrationSummary;
  recalibrated_isotonic: CalibrationSummary;
  recalibrated_sigmoid: CalibrationSummary;
  calibration_optimism_gap_ece: number;
  recalibration_reduction_ece: number;
  decision_curve: DecisionCurve;
}

export interface ModelResult {
  model: string;
  accuracy_mean: number;
  f1_macro_mean: number;
  balanced_accuracy: number | null;
}

export interface TaskResult {
  n_windows: number;
  n_features: number;
  n_subjects?: number;
  classes: string[];
  models: ModelResult[];
  best_model: string;
  inference_model?: string;
  loso_accuracy: number;
  loso_matched_accuracy?: number | null;
  within_subject_accuracy?: number | null;
  optimism_gap_pts?: number | null;
}

export interface BenchmarkResults {
  benchmark_protocol_version?: number;
  benchmark_source_matches_current?: boolean;
  unverified_sections?: string[];
  binary: TaskResult;
  multiclass: TaskResult;
  shap?: { feature: string; mean_abs_shap: number }[];
  shap_model?: string;
  shap_scope?: string;
  calibration?: Calibration | null;
  wrist?: {
    same_model_rf?: { chest: number | null; wrist: number | null; drop_pts: number | null };
  } | null;
  cross_dataset?: {
    wesad_to_noneeg?: { balanced_accuracy: number | null };
    noneeg_to_wesad?: { balanced_accuracy: number | null };
  } | null;
}
