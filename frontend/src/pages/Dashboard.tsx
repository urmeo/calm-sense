import React from 'react';
import { Activity, Brain, Layers, Award } from 'lucide-react';
import results, { shapContext } from '../data';
import Panel from '../components/Panel';
import Chart from '../components/Chart';
import { barChartOption } from '../lib/charts';
import { formatPercent as pct, matchedGap } from '../lib/benchmarks';

const CARD_COLORS: Record<string, string> = {
  blue: 'bg-blue-100 dark:bg-blue-900/30',
  green: 'bg-green-100 dark:bg-green-900/30',
  purple: 'bg-purple-100 dark:bg-purple-900/30',
  orange: 'bg-orange-100 dark:bg-orange-900/30',
};

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color?: string;
}> = ({ title, value, icon, color = 'blue' }) => (
  <Panel>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
        <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">{value}</p>
      </div>
      <div className={`p-3 rounded-lg ${CARD_COLORS[color]}`}>{icon}</div>
    </div>
  </Panel>
);

const FeatureImportanceChart: React.FC = () => {
  const palette = ['#3182CE', '#38A169', '#D69E2E', '#E53E3E', '#805AD5', '#DD6B20', '#319795', '#D53F8C'];
  const data = [...(results.shap ?? [])]
    .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
    .slice(0, 8).map((feature, i) => ({
    name: feature.feature,
    value: feature.mean_abs_shap,
    color: palette[i % palette.length],
  }));

  return (
    <Panel title={`Top features · ${shapContext.title}`}>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">{shapContext.description}</p>
      {data.length === 0 ? <p role="status" className="text-sm text-gray-500 dark:text-gray-400">SHAP values are unavailable for this benchmark.</p> : <Chart
        label="Top eight features by mean absolute SHAP contribution."
        option={barChartOption({ labels: data.map((d) => d.name), values: data.map((d) => d.value),
          colors: data.map((d) => d.color), horizontal: true, axisName: 'mean |SHAP|' })}
      />}
    </Panel>
  );
};

const OptimismGapChart: React.FC = () => {
  const loso = results.binary.loso_matched_accuracy;
  const within = results.binary.within_subject_accuracy;
  const gap = matchedGap(loso, within);
  const hasMatched = gap !== undefined;
  const data = [
    { name: 'LOSO\n(subject-independent)', value: loso, color: '#3182CE' },
    { name: 'Subject-mixed\n(5-fold)', value: within, color: '#E67E22' },
  ];
  return (
    <Panel title="Optimism gap">
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        {hasMatched ? `${results.binary.best_model}: subject-mixed accuracy minus LOSO accuracy is ${gap.toFixed(1)} percentage points. Both pool predictions from the same non-overlapping windows.` : 'The matched-window comparison is unavailable for the selected benchmark model.'}
      </p>
      {hasMatched && <Chart height={250}
        label={`Matched LOSO versus subject-mixed accuracy. Optimism gap: ${gap?.toFixed(1)} percentage points.`}
        option={barChartOption({ labels: data.map((d) => d.name),
          values: data.map((d) => typeof d.value === 'number' ? d.value * 100 : null),
          colors: data.map((d) => d.color), maximum: 100, percent: true, precision: 1, axisName: 'Accuracy' })}
      />}
    </Panel>
  );
};

const ModelComparisonList: React.FC = () => {
  const models = [...results.binary.models].sort(
    (a, b) => b.accuracy_mean - a.accuracy_mean
  );
  return (
    <Panel title="Binary LOSO accuracy by model · subject means">
      <div className="space-y-3">
        {models.map((m) => (
          <div key={m.model} className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400 w-40">{m.model}</span>
            <div className="flex-1 mx-3 bg-gray-100 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full"
                style={{ width: `${m.accuracy_mean * 100}%` }}
              />
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white w-16 text-right">
              {pct(m.accuracy_mean)}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
};

const DatasetSummary: React.FC = () => {
  const { binary, multiclass } = results;
  const rows: [string, string | number][] = [
    ['Dataset', 'WESAD (chest signals)'],
    ['Subjects', binary.n_subjects ?? 'Not recorded'],
    ['Windows (binary)', binary.n_windows],
    ['Features', binary.n_features],
    ['Binary classes', binary.classes.join(', ')],
    ['3-class classes', multiclass.classes.join(', ')],
    ['Validation', 'Leave-One-Subject-Out'],
  ];
  return (
    <Panel title="Dataset & setup">
      <div className="space-y-3">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">{k}</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{v}</span>
          </div>
        ))}
      </div>
    </Panel>
  );
};

const Dashboard: React.FC = () => {
  const { binary, multiclass } = results;

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Saved WESAD experiments. Leave-one-subject-out (LOSO) validation tests each
          person with a model trained on the other subjects. Accuracy below is the
          mean across held-out subjects.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Binary LOSO accuracy" value={pct(binary.loso_accuracy)} icon={<Activity className="w-6 h-6 text-blue-600" />} color="blue" />
        <MetricCard title="3-class LOSO accuracy" value={pct(multiclass.loso_accuracy)} icon={<Brain className="w-6 h-6 text-green-600" />} color="green" />
        <MetricCard title="Saved features" value={binary.n_features} icon={<Layers className="w-6 h-6 text-purple-600" />} color="purple" />
        <MetricCard title="Highest binary accuracy" value={binary.best_model} icon={<Award className="w-6 h-6 text-orange-600" />} color="orange" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FeatureImportanceChart />
        <OptimismGapChart />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ModelComparisonList />
        <DatasetSummary />
      </div>
    </div>
  );
};

export default Dashboard;
