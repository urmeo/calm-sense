import React from 'react';
import Chart from '../components/Chart';
import { barChartOption } from '../lib/charts';
import { FileSearch, Info, AlertTriangle } from 'lucide-react';
import results, { shapContext } from '../data';
import Panel from '../components/Panel';

const importance = [...(results.shap ?? [])]
  .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
  .slice(0, 12)
  .map((entry) => ({ feature: entry.feature.replace(/_/g, ' '), value: entry.mean_abs_shap }));
const maxImportance = Math.max(...importance.map((d) => d.value), 0);

const ExplainabilityDashboard: React.FC = () => {
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <FileSearch className="w-6 h-6" /> Explainability
        </h1>
        <p className="text-gray-500 dark:text-gray-400">
          {shapContext.title}. {shapContext.description}
          {' '}These values do not measure held-out performance or causal effects.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Top features (mean |SHAP|)">
          {importance.length === 0 && <p role="status" className="text-sm text-gray-500 dark:text-gray-400">SHAP values are unavailable for this benchmark.</p>}
          <div className="space-y-2">
            {importance.map((item) => (
              <div key={item.feature}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300">{item.feature}</span>
                  <span className="font-mono text-gray-500">{item.value.toFixed(3)}</span>
                </div>
                <div className="w-full bg-gray-100 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-indigo-500 h-2 rounded-full"
                    style={{ width: `${maxImportance > 0 ? (item.value / maxImportance) * 100 : 0}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Importance ranking">
          {importance.length > 0 && <Chart height={360}
            label="Top twelve features ranked by mean absolute SHAP contribution."
            option={barChartOption({ labels: importance.map((d) => d.feature), values: importance.map((d) => d.value),
              colors: ['#6366F1'], horizontal: true, axisName: 'mean |SHAP|' })}
          />}
        </Panel>
      </div>

      <Panel title="What the model relies on">
        <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300">
          {importance.length > 0 ? (
            <p>
              The largest mean absolute contribution in this fit is{' '}
              <strong>{importance[0].feature}</strong>. The ranking summarizes model behavior
              across the training rows; it is not a per-person physiological assessment.
            </p>
          ) : (
            <p>SHAP values are unavailable for this benchmark.</p>
          )}
          <div className="flex items-start gap-2 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5 shrink-0" />
            <p className="text-yellow-800 dark:text-yellow-300">
              <strong>Motion confound:</strong> accelerometer features may encode differences
              between laboratory tasks. The ablation study measures accuracy after removing
              motion features; a high SHAP value alone does not establish a stress biomarker.
            </p>
          </div>
          <div className="flex items-start gap-2 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <Info className="w-5 h-5 text-blue-500 mt-0.5 shrink-0" />
            <p className="text-blue-800 dark:text-blue-300">
              <strong>Research scope:</strong> aggregate feature importance only. This dashboard
              does not diagnose stress or provide individual clinical interpretation.
            </p>
          </div>
        </div>
      </Panel>
    </div>
  );
};

export default ExplainabilityDashboard;
