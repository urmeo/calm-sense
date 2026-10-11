import savedResults from '../../outputs/dashboard/results';
import type { BenchmarkResults } from './types';

const results: BenchmarkResults = savedResults;

export const shapContext = {
  title: `SHAP · ${results.shap_model ?? 'model not recorded'}`,
  description: results.shap_scope === 'full_data_binary_fit'
    ? 'Mean absolute SHAP values from a binary model fitted and explained on all binary-task windows.'
    : 'The saved results do not record which data was used to fit and explain the model.',
};

export default results;
