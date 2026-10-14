import React from 'react';
import Chart, { type ChartOption } from '../components/Chart';
import { barChartOption } from '../lib/charts';
import { Gauge, Target, AlertTriangle, TrendingDown } from 'lucide-react';
import results from '../data';
import Panel from '../components/Panel';
import SummaryCard from '../components/SummaryCard';

const fmt = (v: number) => v.toFixed(3);
const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;

const COLORS = {
  loso: '#3182CE',
  recal: '#38A169',
  sigmoid: '#805AD5',
  diagonal: '#9CA3AF',
};

const lineSeries = (x: number[], y: number[], name: string, color: string) => ({
  name, type: 'line' as const, data: x.map((value, i) => [value, y[i]]),
  symbolSize: 5, lineStyle: { color }, itemStyle: { color },
});

const calibrationAxes: ChartOption = {
  grid: { left: 55, right: 20, top: 80, bottom: 55 },
  xAxis: { type: 'value', name: 'Confidence', min: 0, max: 1, nameLocation: 'middle', nameGap: 30 },
  yAxis: { type: 'value', name: 'Accuracy', min: 0, max: 1, nameLocation: 'middle', nameGap: 40 },
  legend: { type: 'scroll', top: 5, textStyle: { fontSize: 10 } },
};

const CalibrationPanel: React.FC = () => {
  const cal = results.calibration;
  if (!cal) return <p role="status" className="text-gray-600 dark:text-gray-300">Calibration results are unavailable for this benchmark.</p>;
  const names: Record<string, string> = { lr: 'Logistic Regression', rf: 'Random Forest', xgb: 'XGBoost', lgbm: 'LightGBM' };
  const model = cal.model ? names[cal.model] ?? cal.model : 'Unrecorded model';

  const { loso, loso_matched, within_subject, recalibrated_isotonic, recalibrated_sigmoid, decision_curve: decisionCurve } = cal;

  const evaluations = [
    { name: 'LOSO', summary: loso, color: COLORS.loso },
    { name: 'LOSO + isotonic', summary: recalibrated_isotonic, color: COLORS.recal },
    { name: 'LOSO + sigmoid', summary: recalibrated_sigmoid, color: COLORS.sigmoid },
  ];
  const rows = [
    { key: 'Subject-mixed · matched non-overlapping', summary: within_subject },
    { key: 'LOSO · matched non-overlapping', summary: loso_matched },
    { key: 'LOSO · all windows', summary: loso },
    { key: 'LOSO + isotonic · all windows', summary: recalibrated_isotonic },
    { key: 'LOSO + sigmoid · all windows', summary: recalibrated_sigmoid },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Gauge className="w-6 h-6" /> Calibration
        </h1>
        <p className="text-gray-500 dark:text-gray-400">
          Saved binary {model} probabilities for held-out subjects
          ({cal.n_windows} windows, {cal.n_bins} confidence bins).
          Recalibration uses out-of-fold probabilities from training subjects only.
          The optimism comparison uses a separate subset of non-overlapping windows.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard icon={<Target className="w-6 h-6 text-blue-600" />} label="Full-window LOSO ECE" value={fmt(loso.ece)} />
        <SummaryCard
          icon={<AlertTriangle className="w-6 h-6 text-orange-500" />}
          label="Matched optimism gap (ECE)"
          value={signed(cal.calibration_optimism_gap_ece)}
        />
        <SummaryCard
          icon={<TrendingDown className="w-6 h-6 text-green-600" />}
          label="Full-window isotonic ECE reduction"
          value={signed(cal.recalibration_reduction_ece)}
        />
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl p-4 text-sm text-orange-800 dark:text-orange-200">
        On matched non-overlapping windows, subject-mixed ECE is{' '}
        <strong>{fmt(within_subject.ece)}</strong> and LOSO ECE is{' '}
        <strong>{fmt(loso_matched.ece)}</strong>. LOSO minus subject-mixed ECE is{' '}
        <strong>{signed(cal.calibration_optimism_gap_ece)}</strong>.
        Separately, full-window LOSO ECE changes from <strong>{fmt(loso.ece)}</strong> to{' '}
        <strong>{fmt(recalibrated_isotonic.ece)}</strong> with training-subject isotonic recalibration.
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Reliability · all windows">
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
            Confidence is the model's probability for its predicted class.
            Each point compares mean confidence with accuracy in one bin.
          </p>
          <Chart height={380} label="Reliability curves for LOSO, isotonic and sigmoid calibration, against perfect calibration."
            option={{
              ...calibrationAxes,
              series: [
                { name: 'Perfectly calibrated', type: 'line', data: [[0, 0], [1, 1]],
                  showSymbol: false, lineStyle: { color: COLORS.diagonal, type: 'dotted' } },
                ...evaluations.map(({ name, summary, color }) => lineSeries(
                  summary.reliability.map((r) => r.confidence),
                  summary.reliability.map((r) => r.accuracy),
                  `${name} (ECE ${fmt(summary.ece)})`, color)),
              ],
            }}
          />
        </Panel>

        <Panel title="Expected calibration error · all windows">
          <Chart height={380} label="Expected calibration error for LOSO, isotonic and sigmoid calibration."
            option={barChartOption({ labels: evaluations.map((e) => e.name),
              values: evaluations.map((e) => e.summary.ece), colors: evaluations.map((e) => e.color),
              axisName: 'ECE', showValues: true })}
          />
        </Panel>
      </div>

      <Panel title="Decision-curve analysis">
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Exploratory net benefit on full-window LOSO predictions, versus alerting everyone
          or no one. No clinical deployment has been validated.
        </p>
        <Chart height={360} label="Exploratory decision curves: uncalibrated and recalibrated net benefit versus alerting everyone or no one."
          option={{
            grid: { left: 60, right: 20, top: 20, bottom: 90 },
            xAxis: { type: 'value', name: 'Alert threshold', nameLocation: 'middle', nameGap: 30 },
            yAxis: { type: 'value', name: 'Net benefit', nameLocation: 'middle', nameGap: 45, scale: true },
            legend: { type: 'scroll', bottom: 5, textStyle: { fontSize: 11 } },
            series: [
              lineSeries(decisionCurve.thresholds, decisionCurve.net_benefit_uncalibrated, 'Uncalibrated', COLORS.loso),
              lineSeries(decisionCurve.thresholds, decisionCurve.net_benefit_recalibrated, 'Isotonic recalibration', COLORS.recal),
              { ...lineSeries(decisionCurve.thresholds, decisionCurve.treat_all, 'Alert everyone', COLORS.diagonal),
                showSymbol: false, lineStyle: { color: COLORS.diagonal, type: 'dashed' } },
              { ...lineSeries(decisionCurve.thresholds, decisionCurve.thresholds.map(() => 0), 'Alert no one', '#6B7280'),
                showSymbol: false, lineStyle: { color: '#6B7280', type: 'dotted' } },
            ],
          }}
        />
      </Panel>

      <Panel title="Calibration metrics by evaluation">
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          ECE is the average absolute gap between confidence and accuracy across bins,
          weighted by window count. MCE is the largest bin gap.
          Brier is the mean squared error of the stress probability.
          Matched and full-window rows use different evaluation sets.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700 text-left text-gray-600 dark:text-gray-300">
                <th className="px-4 py-2">Evaluation</th>
                <th className="px-4 py-2">ECE</th>
                <th className="px-4 py-2">MCE</th>
                <th className="px-4 py-2">Brier</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key} className="border-b border-gray-100 dark:border-gray-700">
                  <td className="px-4 py-2 font-medium text-gray-900 dark:text-white">{row.key}</td>
                  <td className="px-4 py-2">{fmt(row.summary.ece)}</td>
                  <td className="px-4 py-2">{fmt(row.summary.mce)}</td>
                  <td className="px-4 py-2">{fmt(row.summary.brier)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
};

export default CalibrationPanel;
