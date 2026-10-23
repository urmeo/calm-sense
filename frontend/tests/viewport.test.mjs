import assert from 'node:assert/strict';
import { test } from 'node:test';
import { zoomRange } from '../src/lib/viewport.ts';

test('invalid minimum zoom widths restore a finite full recording range', () => {
  for (const minimumWidth of [NaN, Infinity, -Infinity, -1]) {
    const range = zoomRange([20, 40], 120, 0.5, minimumWidth);
    assert.deepEqual(range, [0, 120]);
    assert.ok(range.every(Number.isFinite));
  }
});

test('valid minimum widths preserve zooming and clamp to the recording duration', () => {
  assert.deepEqual(zoomRange([0, 120], 120, 0.5, 0), [30, 90]);
  assert.deepEqual(zoomRange([0, 10], 120, 0.5, 30), [0, 30]);
  assert.deepEqual(zoomRange([0, 10], 120, 0.5, 150), [0, 120]);
});
