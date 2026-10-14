import React, { useState, useMemo, useEffect } from 'react';
import Chart, { type ChartOption } from '../components/Chart';
import { ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';
import realSignals from '../../../outputs/dashboard/signals';
import { zoomRange, dataZoomRange } from '../lib/viewport';
import { readSignalRecording, recordingDuration, conditionSegments, type SignalRecording } from '../lib/signals';

const recordings = Object.fromEntries(Object.entries(realSignals).map(([subject, recording]) => [subject, readSignalRecording(recording)]));
const subjects = Object.keys(recordings);

const PANELS = [
  { key: 'ecg', title: 'ECG (mV)', series: [{ y: 'ecg', name: 'ECG', color: '#E53E3E' }] },
  { key: 'eda', title: 'EDA (μS)', series: [{ y: 'eda', name: 'EDA', color: '#3182CE' }] },
  { key: 'temp', title: 'Temp (°C)', series: [{ y: 'temp', name: 'Temperature', color: '#38A169' }] },
  {
    key: 'acc',
    title: 'ACC (g)',
    series: [
      { y: 'accX', name: 'ACC X', color: '#805AD5' },
      { y: 'accY', name: 'ACC Y', color: '#DD6B20' },
      { y: 'accZ', name: 'ACC Z', color: '#319795' },
    ],
  },
] as const;

const CONDITIONS: Record<string, { background: string; color: string; swatch: string; label: string }> = {
  Baseline: { background: 'rgba(56, 161, 105, 0.10)', color: '#38A169', swatch: 'bg-green-200', label: 'Baseline (Rest)' },
  Stress: { background: 'rgba(229, 62, 62, 0.10)', color: '#E53E3E', swatch: 'bg-red-200', label: 'Stress (TSST)' },
  Amusement: { background: 'rgba(214, 158, 46, 0.10)', color: '#D69E2E', swatch: 'bg-yellow-200', label: 'Amusement (Fun Videos)' },
};

const SignalPlots: React.FC<{ signalData: SignalRecording; subject: string }> = ({ signalData, subject }) => {
  const [visibleSignals, setVisibleSignals] = useState({ ecg: true, eda: true, temp: true, acc: false });
  const duration = recordingDuration(signalData);
  const sampleStep = signalData.time[1] - signalData.time[0];
  const [xRange, setXRange] = useState<[number, number]>([0, duration]);
  useEffect(() => setXRange([0, duration]), [signalData, duration]);
  const segments = useMemo(() => conditionSegments(signalData), [signalData]);

  const toggleSignal = (signal: keyof typeof visibleSignals) =>
    setVisibleSignals((prev) => ({ ...prev, [signal]: !prev[signal] }));

  const handleZoomIn = () => setXRange((range) => zoomRange(range, duration, 0.5, sampleStep));
  const handleZoomOut = () => setXRange((range) => zoomRange(range, duration, 2, sampleStep));
  const handleReset = () => setXRange([0, duration]);

  const visiblePanels = PANELS.filter((p) => visibleSignals[p.key]);

  const panelHeight = (420 - (visiblePanels.length - 1) * 20) / Math.max(visiblePanels.length, 1);
  const axisIndices = visiblePanels.map((_, i) => i);
  const option: ChartOption = {
    title: { text: `Signal Explorer: Subject ${subject}`, left: 'center', textStyle: { fontSize: 18 } },
    legend: { type: 'scroll', bottom: 0 },
    tooltip: { trigger: 'axis', confine: true, renderMode: 'richText' },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    grid: visiblePanels.map((_, i) => ({ left: 65, right: 30, top: 65 + i * (panelHeight + 20), height: panelHeight })),
    xAxis: visiblePanels.map((_, i) => ({
      type: 'value', gridIndex: i, min: 0, max: duration,
      name: i === visiblePanels.length - 1 ? 'Displayed time (s)' : '',
      nameLocation: 'middle', nameGap: 30,
      axisLabel: { show: i === visiblePanels.length - 1 },
      axisPointer: { show: true },
    })),
    yAxis: visiblePanels.map((panel, i) => ({
      type: 'value', gridIndex: i, name: panel.title, nameLocation: 'middle', nameGap: 45,
      scale: true, splitLine: { lineStyle: { opacity: 0.2 } },
    })),
    dataZoom: [
      { id: 'signal-gesture', type: 'inside', xAxisIndex: axisIndices, filterMode: 'none',
        startValue: xRange[0], endValue: xRange[1], rangeMode: ['value', 'value'], minValueSpan: sampleStep },
      { id: 'signal-slider', type: 'slider', xAxisIndex: axisIndices, filterMode: 'none', bottom: 35, height: 20,
        startValue: xRange[0], endValue: xRange[1], rangeMode: ['value', 'value'], minValueSpan: sampleStep },
    ],
    series: visiblePanels.flatMap((panel, i) => panel.series.map((series, j) => ({
      name: series.name, type: 'line', xAxisIndex: i, yAxisIndex: i,
      data: signalData.time.map((time, index) => [time, signalData[series.y][index]]),
      showSymbol: false, lineStyle: { color: series.color, width: 1 }, itemStyle: { color: series.color },
      markArea: j === 0 ? {
        silent: true,
        data: segments.map((segment) => [
          { name: i === 0 ? segment.name : '', xAxis: segment.x0,
            itemStyle: { color: CONDITIONS[segment.name]?.background || 'rgba(0,0,0,0.04)' },
            label: { color: CONDITIONS[segment.name]?.color || '#666', position: 'insideTop' } },
          { xAxis: segment.x1 },
        ]),
      } : undefined,
    }))),
  };

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex flex-wrap items-center gap-4">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Signals:</span>
            {PANELS.map(({ key: signal }) => (
              <label key={signal} className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={visibleSignals[signal]}
                  onChange={() => toggleSignal(signal)}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <span className="text-sm text-gray-600 dark:text-gray-400 uppercase">{signal}</span>
              </label>
            ))}
          </div>
          <div className="flex items-center space-x-2 ml-auto">
            <button onClick={handleZoomIn} aria-label="Zoom in" className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600" title="Zoom In">
              <ZoomIn className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            <button onClick={handleZoomOut} aria-label="Zoom out" className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600" title="Zoom Out">
              <ZoomOut className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
            <button onClick={handleReset} aria-label="Reset signal view" className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600" title="Reset View">
              <RefreshCw className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        {visiblePanels.length === 0 ? (
          <p role="status" className="text-gray-600 dark:text-gray-300">Select a signal to display.</p>
        ) : <Chart
          label={`Subject ${subject}: ${visiblePanels.map((panel) => panel.title).join(', ')}. Displayed clip time ${xRange[0].toFixed(1)} to ${xRange[1].toFixed(1)} seconds. Use zoom controls or drag the slider to inspect a range.`}
          height={600} option={option} downloadFilename={`signal-${subject}.svg`}
          onDataZoom={(event) => {
            const range = dataZoomRange(event, duration);
            if (range) setXRange(range);
          }}
        />}
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">Condition Legend</h3>
        <div className="flex flex-wrap gap-4">
          {Object.entries(CONDITIONS).map(([name, { swatch, label }]) => (
            <div key={name} className="flex items-center space-x-2">
              <div className={`w-6 h-4 rounded ${swatch}`}></div>
              <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const SignalExplorer: React.FC = () => {
  const [selectedSubject, setSelectedSubject] = useState(subjects[0] ?? '');
  const recording = recordings[selectedSubject];
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Signal Explorer</h1>
          <p className="text-gray-500 dark:text-gray-400">
            Real WESAD chest signal clips, downsampled and concatenated for display.
            The time axis does not represent a continuous recording.
          </p>
        </div>
        <select
          aria-label="WESAD subject"
          disabled={subjects.length === 0}
          value={selectedSubject}
          onChange={(e) => setSelectedSubject(e.target.value)}
          className="px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
        >
          {subjects.length === 0 && <option value="">No recordings available</option>}
          {subjects.map((subject) => (
            <option key={subject} value={subject}>
              Subject {subject}
            </option>
          ))}
        </select>
      </div>

      {recording ? <SignalPlots signalData={recording} subject={selectedSubject} /> : (
        <p role="status" className="text-gray-600 dark:text-gray-300">Signal samples are unavailable for this subject.</p>
      )}
    </div>
  );
};

export default SignalExplorer;
