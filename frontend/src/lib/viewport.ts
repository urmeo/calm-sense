export function zoomRange(
  range: [number, number], duration: number, factor: number, minimumWidth = 0,
): [number, number] {
  if (!Number.isFinite(duration) || duration <= 0) return [0, 0];
  if (!range.every(Number.isFinite) || range[1] <= range[0]
    || !Number.isFinite(factor) || factor <= 0
    || !Number.isFinite(minimumWidth) || minimumWidth < 0) return [0, duration];
  const width = Math.min(Math.max((range[1] - range[0]) * factor, minimumWidth, duration * 1e-9), duration);
  const center = (range[0] + range[1]) / 2;
  const start = Math.max(0, Math.min(center - width / 2, duration - width));
  return [start, start + width];
}

export function dataZoomRange(event: unknown, duration: number): [number, number] | null {
  if (!event || typeof event !== 'object' || !Number.isFinite(duration) || duration <= 0) return null;
  const action = event as Record<string, unknown>;
  const change = Array.isArray(action.batch) ? action.batch[0] : action;
  if (!change || typeof change !== 'object') return null;
  const { startValue, endValue, start, end } = change as Record<string, unknown>;
  const range = startValue !== undefined || endValue !== undefined
    ? [startValue, endValue]
    : [typeof start === 'number' ? start * duration / 100 : undefined,
      typeof end === 'number' ? end * duration / 100 : undefined];
  if (!range.every((value) => typeof value === 'number' && Number.isFinite(value))) return null;
  const lower = Math.max(0, range[0] as number);
  const upper = Math.min(duration, range[1] as number);
  return upper > lower ? [lower, upper] : null;
}
