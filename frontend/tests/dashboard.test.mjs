import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readDarkMode, saveDarkMode } from '../src/lib/preferences.ts';
import { zoomRange, dataZoomRange } from '../src/lib/viewport.ts';
import { ancillaryBenchmarkNotice, formatPercent, matchedGap } from '../src/lib/benchmarks.ts';
import { readSignalRecording, recordingDuration, conditionSegments } from '../src/lib/signals.ts';
import { containNavigationFocus } from '../src/lib/navigation.ts';

test('corrupt theme preferences do not crash startup or enable dark mode', () => {
  for (const saved of [null, '', '{invalid', 'null', '1', 'false', '"true"']) {
    globalThis.window = { localStorage: { getItem: () => saved } };
    assert.equal(readDarkMode(), false);
  }
  globalThis.window = { localStorage: { getItem: () => 'true' } };
  assert.equal(readDarkMode(), true);
});

test('blocked storage does not prevent theme changes', () => {
  globalThis.window = {
    get localStorage() { throw new Error('Storage blocked'); },
  };
  assert.equal(readDarkMode(), false);
  assert.doesNotThrow(() => saveDarkMode(true));
  let saved;
  globalThis.window = { localStorage: { setItem: (key, value) => { saved = [key, value]; } } };
  saveDarkMode(true);
  assert.deepEqual(saved, ['darkMode', 'true']);
});

test('zoom out preserves its requested width at recording edges', () => {
  assert.deepEqual(zoomRange([0, 10], 120, 2), [0, 20]);
  assert.deepEqual(zoomRange([110, 120], 120, 2), [100, 120]);
  assert.deepEqual(zoomRange([0, 120], 120, 2), [0, 120]);
  assert.deepEqual(zoomRange([40, 80], 120, 0.5), [50, 70]);
});

test('benchmarks without ancillary snapshots do not show a notice', () => {
  assert.equal(ancillaryBenchmarkNotice({}), null);
  assert.equal(ancillaryBenchmarkNotice({ benchmark_protocol_version: 2 }), null);
});

test('unverified snapshots retain their labels and remove duplicate sections', () => {
  assert.equal(ancillaryBenchmarkNotice({
    benchmark_protocol_version: 2,
    unverified_sections: ['calibration', 'cross_dataset', 'wrist', 'shap', 'calibration', 'custom analysis'],
  }), 'Additional snapshots lack verified linkage to this benchmark: calibration, transfer, wrist comparison, SHAP, custom analysis.');
});

test('current protocols infer missing ancillary verification but respect an explicit empty list', () => {
  for (const version of [2, 3]) {
    const metadata = {
      benchmark_protocol_version: version, calibration: {}, wrist: {}, shap: [], cross_dataset: null,
    };
    assert.equal(ancillaryBenchmarkNotice(metadata),
      'Additional snapshots lack verified linkage to this benchmark: SHAP, wrist comparison, calibration.');
    assert.equal(ancillaryBenchmarkNotice({ ...metadata, unverified_sections: [] }), null);
  }
});

test('missing or historical protocols use only explicitly recorded unverified sections', () => {
  for (const version of [undefined, 1, NaN, Infinity, 2.1, -1]) {
    const metadata = { benchmark_protocol_version: version, calibration: {} };
    assert.equal(ancillaryBenchmarkNotice(metadata), null);
    assert.equal(ancillaryBenchmarkNotice({ ...metadata, unverified_sections: ['calibration'] }),
      'Additional snapshots lack verified linkage to this benchmark: calibration.');
  }
});

test('missing or nonfinite comparison metrics remain unavailable rather than zero', () => {
  for (const value of [undefined, null, NaN, Infinity]) {
    assert.equal(formatPercent(value), 'Unavailable');
    assert.equal(matchedGap(value, 0.9), undefined);
    assert.equal(matchedGap(0.8, value), undefined);
  }
  assert.equal(formatPercent(0), '0.0%');
  assert.equal(formatPercent(0.9132857382783859), '91.3%');
  assert.ok(Math.abs(matchedGap(0.9, 0.85) + 5) < 1e-10);
});

test('chart zoom events reject malformed values and clamp panning to the clip', () => {
  for (const event of [null, 1, {}, { start: '0', end: 10 }, { start: 0 },
    { start: 0, end: Infinity }, { start: 10, end: 0 }, { start: 120, end: 150 },
    { batch: [] }, { batch: [null] }, { startValue: 0, endValue: NaN }]) {
    assert.equal(dataZoomRange(event, 120), null);
  }
  assert.deepEqual(dataZoomRange({ start: -10, end: 25 }, 120), [0, 30]);
  assert.deepEqual(dataZoomRange({ startValue: 100, endValue: 140 }, 120), [100, 120]);
  assert.deepEqual(dataZoomRange({ batch: [{ start: 0, end: 100 }] }, 120), [0, 120]);
  assert.deepEqual(dataZoomRange({ batch: [{ start: 25, end: 50 }] }, 120), [30, 60]);
  assert.equal(dataZoomRange({ start: 0, end: 100 }, 0), null);
  assert.equal(dataZoomRange({ start: 0, end: 100 }, NaN), null);
});

test('repeated zoom keeps at least one sample interval and recovers invalid ranges', () => {
  let range = [0, 120];
  for (let i = 0; i < 1000; i++) range = zoomRange(range, 120, 0.5, 1 / 30);
  assert.ok(range[1] - range[0] >= 1 / 30 - 1e-12);
  assert.deepEqual(zoomRange([NaN, 10], 120, 0.5), [0, 120]);
  assert.deepEqual(zoomRange([5, 5], 120, 0.5), [0, 120]);
  assert.deepEqual(zoomRange([0, 120], 120, 0), [0, 120]);
});

const clip = () => ({
  time: [0, 0.5, 1], ecg: [1, 2, 3], eda: [1, 2, 3], temp: [30, 31, 32],
  accX: [0, 0, 0], accY: [0, 0, 0], accZ: [1, 1, 1],
  conditions: ['Baseline', 'Stress', 'Stress'],
});

test('signal clips require aligned finite channels and increasing display times', () => {
  assert.equal(readSignalRecording(null), null);
  assert.equal(readSignalRecording({}), null);
  for (const changes of [{ time: [] }, { time: [0] }, { time: [0, 0, 1] },
    { time: [0, 1, 0.5] }, { time: [0, 0.5, NaN] }, { eda: [1, 2] },
    { accZ: [0, Infinity, 1] }, { ecg: [true, 2, 3] },
    { conditions: ['Baseline'] }, { conditions: [1, 'Stress', 'Stress'] }]) {
    assert.equal(readSignalRecording({ ...clip(), ...changes }), null);
  }
  const recording = readSignalRecording(clip());
  assert.equal(recordingDuration(recording), 1.5);
  assert.deepEqual(conditionSegments(recording), [
    { name: 'Baseline', x0: 0, x1: 0.5 }, { name: 'Stress', x0: 0.5, x1: 1.5 },
  ]);
});

test('the saved signal exports remain valid and all condition segments cover the clip', async () => {
  const { default: recordings } = await import('../../outputs/dashboard/signals.ts');
  assert.ok(Object.keys(recordings).length > 0);
  for (const row of Object.values(recordings)) {
    const recording = readSignalRecording(row);
    assert.ok(recording);
    const segments = conditionSegments(recording);
    assert.equal(segments[0].x0, 0);
    assert.equal(segments.at(-1).x1, recordingDuration(recording));
    for (let i = 1; i < segments.length; i++) assert.equal(segments[i].x0, segments[i - 1].x1);
  }
});

test('mobile menu contains keyboard focus, closes on Escape and restores its trigger', () => {
  const listeners = new Map();
  const document = {
    activeElement: null,
    addEventListener: (type, callback) => listeners.set(type, callback),
    removeEventListener: (type) => listeners.delete(type),
  };
  class Element {
    isConnected = true;
    ownerDocument = document;
    focus() { document.activeElement = this; listeners.get('focusin')?.(); }
  }
  const originalHTMLElement = globalThis.HTMLElement;
  globalThis.HTMLElement = Element;
  try {
    const trigger = new Element();
    const first = new Element();
    const last = new Element();
    const sidebar = new Element();
    sidebar.querySelectorAll = () => [first, last];
    sidebar.contains = (element) => element === first || element === last;
    const background = { inert: false };
    trigger.focus();
    let closed = 0;
    const release = containNavigationFocus(sidebar, background, () => closed++);
    assert.equal(background.inert, true);
    assert.equal(document.activeElement, first);
    let prevented = 0;
    listeners.get('keydown')({ key: 'Tab', shiftKey: true, preventDefault: () => prevented++ });
    assert.equal(document.activeElement, last);
    listeners.get('keydown')({ key: 'Tab', shiftKey: false, preventDefault: () => prevented++ });
    assert.equal(document.activeElement, first);
    trigger.focus();
    assert.equal(document.activeElement, first);
    listeners.get('keydown')({ key: 'Escape', preventDefault: () => prevented++ });
    assert.equal(closed, 1);
    assert.equal(prevented, 3);
    release();
    assert.equal(background.inert, false);
    assert.equal(document.activeElement, trigger);
    assert.equal(listeners.size, 0);
  } finally {
    globalThis.HTMLElement = originalHTMLElement;
  }
});
