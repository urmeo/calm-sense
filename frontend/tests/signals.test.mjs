import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readSignalRecording, signalPoints, conditionSegments } from '../src/lib/signals.ts';

const recording = () => ({
  time: [0, 0.5, 1],
  ecg: [1, 2, 3], eda: [1, 2, 3], temp: [30, 31, 32],
  accX: [0, 0, 0], accY: [0, 0, 0], accZ: [1, 1, 1],
  conditions: ['Baseline', 'Stress', 'Stress'],
});

test('signal validation rejects holes in every sample channel and condition array', () => {
  for (const channel of ['time', 'ecg', 'eda', 'temp', 'accX', 'accY', 'accZ', 'conditions']) {
    const value = recording();
    delete value[channel][1];
    assert.equal(readSignalRecording(value), null, channel);
  }
});

test('signal validation rejects channels with only a length and no samples', () => {
  for (const channel of ['ecg', 'eda', 'temp', 'accX', 'accY', 'accZ', 'conditions']) {
    assert.equal(readSignalRecording({ ...recording(), [channel]: Array(3) }), null, channel);
  }
});

test('signal validation preserves all dense samples and conditions', () => {
  const value = recording();
  assert.equal(readSignalRecording(value), value);
  assert.deepEqual(value.conditions, ['Baseline', 'Stress', 'Stress']);
});

test('signal points break independent clips without removing measured samples', () => {
  const value = recording();
  assert.deepEqual(signalPoints(value, 'ecg'), [[0, 1], [0.25, null], [0.5, 2], [1, 3]]);
  value.conditions = ['Baseline', 'Baseline', 'Baseline'];
  assert.deepEqual(signalPoints(value, 'ecg'), [[0, 1], [0.5, 2], [1, 3]]);
});

test('all saved channel samples remain intact with one break per clip join', async () => {
  const { default: recordings } = await import('../../outputs/dashboard/signals.ts');
  for (const value of Object.values(recordings)) {
    const recording = readSignalRecording(value);
    assert.ok(recording);
    const joins = conditionSegments(recording).length - 1;
    for (const channel of ['ecg', 'eda', 'temp', 'accX', 'accY', 'accZ']) {
      const points = signalPoints(recording, channel);
      assert.equal(points.filter(([, value]) => value === null).length, joins);
      assert.deepEqual(points.filter(([, value]) => value !== null),
        recording.time.map((time, index) => [time, recording[channel][index]]));
    }
  }
});
