const SECTION_LABELS = {
  shap: 'SHAP', stats: 'statistics', wrist: 'wrist comparison',
  cross_dataset: 'transfer', calibration: 'calibration',
  personalization: 'personalization', tuning: 'tuning', ablation: 'ablation',
} as const;

interface BenchmarkMetadata extends Partial<Record<keyof typeof SECTION_LABELS, unknown>> {
  benchmark_protocol_version?: number;
  unverified_sections?: string[];
}

export function ancillaryBenchmarkNotice(metadata: BenchmarkMetadata): string | null {
  const version = metadata.benchmark_protocol_version;
  const hasCurrentProtocol = Number.isInteger(version) && (version ?? 0) >= 2;
  const labels: Record<string, string> = SECTION_LABELS;
  const present = Object.entries(SECTION_LABELS)
    .filter(([name]) => metadata[name as keyof typeof SECTION_LABELS] != null)
    .map(([name]) => name);
  const sections = [...new Set(metadata.unverified_sections ?? (hasCurrentProtocol ? present : []))];
  return sections.length > 0
    ? `Additional snapshots lack verified linkage to this benchmark: ${sections.map((name) => labels[name] ?? name).join(', ')}.`
    : null;
}

export function formatPercent(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%` : 'Unavailable';
}

export function matchedGap(
  loso: number | null | undefined, withinSubject: number | null | undefined,
): number | undefined {
  return typeof loso === 'number' && Number.isFinite(loso)
    && typeof withinSubject === 'number' && Number.isFinite(withinSubject)
    ? (withinSubject - loso) * 100 : undefined;
}
