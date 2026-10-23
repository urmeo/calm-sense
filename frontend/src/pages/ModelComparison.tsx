import React, { useState } from 'react';
import { Trophy, AlertTriangle, Activity } from 'lucide-react';
import results, { shapContext } from '../data';
import Panel from '../components/Panel';
import Chart from '../components/Chart';
import { barChartOption } from '../lib/charts';
import SummaryCard from '../components/SummaryCard';
import PageHeading from '../components/PageHeading';
import ResultsTable from '../components/ResultsTable';
import { formatPercent as pct, matchedGap } from '../lib/benchmarks';

type Task = 'binary' | 'multiclass';

const ModelComparison: React.FC = () => {
  const [task, setTask] = useState<Task>('binary');
  const data = results[task];
  const models = [...data.models].sort(
    (a, b) => b.accuracy_mean - a.accuracy_mean
  );
  const best = models[0];
  const losoMatched = data.loso_matched_accuracy;
  const gap = matchedGap(losoMatched, data.within_subject_accuracy);
  const wrist = results.wrist?.same_model_rf;
  const transfer = results.cross_dataset;
  const shap = results.shap ?? [];

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <PageHeading title="Model Comparison">
            WESAD chest signals. {data.n_subjects === undefined ? 'The subject count is not recorded.' : `${data.n_subjects} subjects, each held out once.`}{' '}
            Accuracy and macro-F1 give each subject equal weight.
            Balanced accuracy pools all held-out predictions.
        </PageHeading>
        <select
          aria-label="Classification task"
          value={task}
          onChange={(e) => {
            const value = e.target.value;
            if (value === 'binary' || value === 'multiclass') setTask(value);
          }}
          className="max-w-full shrink-0 px-3 py-1.5 bg-gray-100 dark:bg-gray-700 border-0 rounded-lg text-sm"
        >
          <option value="binary">Binary (baseline vs. stress)</option>
          <option value="multiclass">3-class (baseline/stress/amusement)</option>
        </select>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard icon={<Trophy className="w-6 h-6 text-green-600" />} label="Highest mean accuracy" value={best?.model ?? 'Unavailable'} />
        <SummaryCard
          icon={<Activity className="w-6 h-6 text-blue-600" />}
          label="LOSO accuracy · subject mean"
          value={pct(best?.accuracy_mean)}
        />
        <SummaryCard
          icon={<AlertTriangle className="w-6 h-6 text-orange-500" />}
          label="Matched optimism gap"
          value={gap === undefined ? 'Unavailable' : `${gap >= 0 ? '+' : ''}${gap.toFixed(1)} pts`}
        />
      </div>

      {gap !== undefined && <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl p-4 text-sm text-orange-800 dark:text-orange-200">
        On the same non-overlapping windows, {data.best_model} pooled accuracy is{' '}
        <strong>{pct(data.within_subject_accuracy)}</strong> under subject-mixed 5-fold validation
        versus <strong>{pct(losoMatched)}</strong> under LOSO. The gap is{' '}
        <strong>{gap.toFixed(1)} percentage points</strong>.
      </div>}

      {task === 'binary' && (wrist || transfer) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {wrist && <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 text-sm">
            <p className="font-semibold text-gray-900 dark:text-white mb-1">Binary RF: wrist vs. chest</p>
            <p className="text-gray-600 dark:text-gray-400">
              Mean LOSO accuracy with Random Forest is{' '}
              <strong>{pct(wrist.wrist)}</strong> for Empatica E4 wrist signals and{' '}
              {pct(wrist.chest)} for chest signals. Chest-to-wrist difference:{' '}
              {typeof wrist.drop_pts === 'number' && Number.isFinite(wrist.drop_pts) ? `${wrist.drop_pts.toFixed(1)} pts.` : 'unavailable.'}{' '}
              This laboratory comparison does not establish sensor equivalence or field performance.
            </p>
          </div>}
          {transfer && <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 text-sm">
            <p className="font-semibold text-gray-900 dark:text-white mb-1">Binary cross-dataset transfer</p>
            <p className="text-gray-600 dark:text-gray-400">
              Balanced accuracy: WESAD → PhysioNet Non-EEG{' '}
              <strong>{pct(transfer.wesad_to_noneeg?.balanced_accuracy)}</strong>;
              reverse transfer{' '}
              <strong>{pct(transfer.noneeg_to_wesad?.balanced_accuracy)}</strong>.
              Signal units, devices, stressors, and labels differ; this pair does not isolate dataset shift.
            </p>
          </div>}
        </div>
      )}

      <Panel title={`Subject-independent performance (${data.n_windows} windows)`}>
        {models.length === 0 && <p role="status" className="text-gray-500 dark:text-gray-400 mb-3">No model results are available for this task.</p>}
        <ResultsTable
          columns={['Model', 'Accuracy (subject mean)', 'Macro-F1 (subject mean)', 'Balanced accuracy (pooled)']}
          rows={models.map((model, index) => ({
            key: model.model,
            highlighted: index === 0,
            cells: [
              <span className="flex items-center gap-2">{index === 0 && <Trophy className="w-4 h-4 text-yellow-500" />}{model.model}</span>,
              pct(model.accuracy_mean), pct(model.f1_macro_mean), pct(model.balanced_accuracy),
            ],
          }))}
        />
      </Panel>

      <div className={`grid grid-cols-1 ${task === 'binary' ? 'lg:grid-cols-2' : ''} gap-6`}>
        <Panel title="LOSO accuracy · subject means">
          <Chart label={`${task} mean subject LOSO accuracy by model, from highest to lowest.`}
            option={barChartOption({ labels: models.map((model) => model.model), values: models.map((model) => +(model.accuracy_mean * 100).toFixed(1)),
              colors: models.map((_, i) => i === 0 ? '#38A169' : '#3182CE'),
              axisName: 'Accuracy', maximum: 100, percent: true, precision: 1 })}
          />
        </Panel>

        {task === 'binary' && (
          <Panel title={shapContext.title}>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              {shapContext.description} These values do not measure held-out performance or causal effects.
            </p>
            {shap.length > 0 ? (
              <Chart label={`${shapContext.title}: mean absolute feature contributions.`}
                option={barChartOption({ labels: shap.map((d) => d.feature), values: shap.map((d) => d.mean_abs_shap),
                  colors: ['#805AD5'], horizontal: true, axisName: 'mean |SHAP|' })}
              />
            ) : (
              <p role="status" className="text-sm text-gray-500 dark:text-gray-400">SHAP values are unavailable for this benchmark.</p>
            )}
          </Panel>
        )}
      </div>
    </div>
  );
};

export default ModelComparison;
