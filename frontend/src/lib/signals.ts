export interface SignalRecording {
  time: number[];
  ecg: number[];
  eda: number[];
  temp: number[];
  accX: number[];
  accY: number[];
  accZ: number[];
  conditions: string[];
}

const channels = ['time', 'ecg', 'eda', 'temp', 'accX', 'accY', 'accZ'] as const;

export function readSignalRecording(value: unknown): SignalRecording | null {
  if (!value || typeof value !== 'object') return null;
  const row = value as Record<string, unknown>;
  const time = row.time;
  if (!Array.isArray(time) || time.length < 2 || time[0] !== 0) return null;
  for (const channel of channels) {
    const samples = row[channel];
    if (!Array.isArray(samples) || samples.length !== time.length) return null;
    for (const sample of samples) {
      if (typeof sample !== 'number' || !Number.isFinite(sample)) return null;
    }
  }
  for (let index = 1; index < time.length; index++) {
    if (time[index] <= time[index - 1]) return null;
  }
  if (!Array.isArray(row.conditions) || row.conditions.length !== time.length) return null;
  for (const condition of row.conditions) {
    if (typeof condition !== 'string') return null;
  }
  return value as SignalRecording;
}

export function recordingDuration({ time }: SignalRecording): number {
  return time[time.length - 1] + (time[1] - time[0]);
}

export function conditionSegments(recording: SignalRecording) {
  const { conditions, time } = recording;
  const segments: { name: string; x0: number; x1: number }[] = [];
  let start = 0;
  for (let i = 1; i <= conditions.length; i++) {
    if (i === conditions.length || conditions[i] !== conditions[start]) {
      segments.push({ name: conditions[start], x0: time[start], x1: i < time.length ? time[i] : recordingDuration(recording) });
      start = i;
    }
  }
  return segments;
}

type SignalChannel = Exclude<keyof SignalRecording, 'time' | 'conditions'>;
type SignalPoint = [time: number, value: number | null];

export function signalPoints(recording: SignalRecording, channel: SignalChannel) {
  const { time, conditions } = recording;
  const values = recording[channel];
  const points: SignalPoint[] = [];
  for (let index = 0; index < time.length; index++) {
    if (index > 0 && conditions[index] !== conditions[index - 1]) {
      const previous = time[index - 1];
      points.push([previous + (time[index] - previous) / 2, null]);
    }
    points.push([time[index], values[index]]);
  }
  return points;
}
