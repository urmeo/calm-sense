import { binary, multiclass } from '../../../outputs/results/metrics.json';
import { best_model, best_ci95, omnibus_friedman, pairwise_holm } from '../../../outputs/results/stats.json';
import { models as thresholdModels } from '../../../outputs/results/threshold_metrics.json';
import { generated_at } from '../../../outputs/results/provenance.json';
import { formatPercent } from '../lib/benchmarks';
import stageArrow from '../assets/pipeline/stage-arrow.svg';
import trainingArrow from '../assets/pipeline/training-arrow.svg';
import testArrow from '../assets/pipeline/test-arrow.svg';
import aggregationArrow from '../assets/pipeline/aggregation-arrow.svg';
import './ResearchPipeline.css';

const people = binary.n_subjects;
const stages = [
  { title: 'Record', lines: ['ECG · EDA · temp.', 'Respiration · motion'] },
  { title: 'Filter + window', lines: ['60 s · 30 s stride', 'At least 90% label purity'] },
  { title: 'Split by person', lines: [`${people - 1} train / validate`, '1 untouched test person'], training: true },
  { title: 'Fit two paths', lines: ['Features or 1D-CNN', 'Train inside each fold'], training: true },
  { title: 'Predict + save', lines: ['Held-out predictions', 'Plots + dashboard'] },
];

const modelPaths = [
  {
    name: 'Feature model path',
    input: `${binary.n_features} features`,
    inputDetail: 'Signal features',
    preparation: 'Impute + standardize',
    preparationLines: ['Fit medians and scaling', 'on training people only'],
    model: 'Four classifiers',
    modelDetail: 'LR · RF · XGBoost · LightGBM',
  },
  {
    name: 'CNN model path',
    input: '5 channels',
    inputDetail: '1,024 samples',
    preparation: 'Split train / validation',
    preparationLines: ['Different people; normalize', 'with inner training windows'],
    model: '1D-CNN',
    modelDetail: 'Early stop on validation',
  },
];

const randomForest = binary.models.find(({ model }) => model === 'Random Forest');
const lightGBM = multiclass.models.find(({ model }) => model === 'LightGBM');
const randomForestAUROC = thresholdModels.find(({ model }) => model === 'Random Forest')?.auroc;
const confidenceInterval = best_model === 'Random Forest'
  ? `${formatPercent(best_ci95[0]).replace('%', '')}–${formatPercent(best_ci95[1])}`
  : 'Unavailable';
const evidence = [
  {
    model: 'Random Forest · binary',
    value: formatPercent(randomForest?.accuracy_mean),
    measure: 'Mean accuracy across people',
    detail: `95% subject-bootstrap CI: ${confidenceInterval}`,
  },
  {
    model: 'Random Forest · binary',
    value: randomForestAUROC?.toFixed(3) ?? 'Unavailable',
    measure: 'Pooled held-out AUROC',
    detail: 'Binary curves: four feature models only',
  },
  {
    model: 'LightGBM · three-class',
    value: formatPercent(lightGBM?.accuracy_mean),
    measure: 'Mean accuracy across people',
    detail: 'Baseline · stress · amusement',
  },
];
const benchmarkDate = new Date(generated_at).toLocaleDateString('en-GB', {
  day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC',
});
const pairwiseSummary = Object.values(pairwise_holm).every(({ p_holm }) => p_holm >= 0.05)
  ? 'no significant Holm-adjusted pairwise accuracy differences.'
  : 'see the saved statistics for Holm-adjusted pairwise accuracy differences.';

/** Primary WESAD benchmark with training boundaries and saved results. */
export default function ResearchPipeline() {
  return (
    <section className="research-pipeline" aria-labelledby="pipeline-title" data-node-id="3:2">
      <header className="pipeline-intro">
        <p className="pipeline-kicker">CalmSense / research pipeline</p>
        <h2 id="pipeline-title">Classify stress in people unseen during training.</h2>
        <p>WESAD chest · {people} people · Baseline / stress; amusement added for three classes</p>
      </header>

      <ol className="pipeline-stages" aria-label="Research pipeline in five steps">
        {stages.map(({ title, lines, training }, index) => (
          <li key={title} className={`pipeline-stage${training ? ' pipeline-stage-training' : ''}`}>
            <span className="pipeline-stage-number">{String(index + 1).padStart(2, '0')}</span>
            <h3>{title}</h3>
            {lines.map((line) => <p key={line}>{line}</p>)}
            {index < stages.length - 1 && (
              <img className="pipeline-arrow pipeline-stage-arrow" src={stageArrow} width="16.5" height="9.50006" alt="" />
            )}
          </li>
        ))}
      </ol>

      <section className="pipeline-fold" aria-labelledby="pipeline-fold-title">
        <header className="pipeline-section-heading">
          <h3 id="pipeline-fold-title">What happens inside each fold</h3>
          <p>LOSO = leave one person out</p>
        </header>
        <div className="pipeline-boundaries">
          <section className="pipeline-training" aria-labelledby="pipeline-training-title">
            <h4 id="pipeline-training-title">{people - 1} people · training and validation</h4>
            <div className="pipeline-paths">
              {modelPaths.map((path) => (
                <ol key={path.name} className="pipeline-path" aria-label={path.name}>
                  <li className="pipeline-model-card">
                    <h5>{path.input}</h5>
                    <p>{path.inputDetail}</p>
                    <img className="pipeline-arrow pipeline-training-arrow" src={trainingArrow} width="21.5" height="9.50006" alt="" />
                  </li>
                  <li className="pipeline-model-card pipeline-transform">
                    <h5>{path.preparation}</h5>
                    {path.preparationLines.map((line) => <p key={line}>{line}</p>)}
                    <img className="pipeline-arrow pipeline-training-arrow" src={trainingArrow} width="21.5" height="9.50006" alt="" />
                  </li>
                  <li className="pipeline-model-card">
                    <h5>{path.model}</h5>
                    <p>{path.modelDetail}</p>
                    <img className="pipeline-arrow pipeline-test-arrow" src={testArrow} width="59.5" height="9.50006" alt="" />
                  </li>
                </ol>
              ))}
            </div>
            <p className="pipeline-training-note">Keep all learned transforms and model fitting inside this pool.</p>
          </section>

          <div className="pipeline-training-to-test" aria-hidden="true">
            <img className="pipeline-arrow" src={testArrow} width="59.5" height="9.50006" alt="" />
          </div>

          <section className="pipeline-test" aria-labelledby="pipeline-test-title">
            <h4 id="pipeline-test-title">1 held-out person</h4>
            <p className="pipeline-test-subtitle">Primary benchmark test only</p>
            <div className="pipeline-test-actions">
              <h5>Apply fitted transforms</h5>
              <p>Predict held-out labels</p>
            </div>
            <div className="pipeline-test-conditions">
              <p>No fitting or tuning</p>
              <p>No validation or early stopping</p>
            </div>
            <p className="pipeline-test-score">Score held-out predictions</p>
          </section>
        </div>
        <div className="pipeline-aggregation-connector" aria-hidden="true">
          <img className="pipeline-arrow" src={aggregationArrow} width="9.50006" height="20.5" alt="" />
        </div>
        <div className="pipeline-aggregation">
          <h4>Repeat for all {people} people</h4>
          <p>→ Subject means + pooled predictions → Saved results</p>
        </div>
      </section>

      <section className="pipeline-evidence" aria-labelledby="pipeline-evidence-title">
        <header className="pipeline-section-heading">
          <h3 id="pipeline-evidence-title">Recorded evidence</h3>
          <p>Saved benchmark · {benchmarkDate}</p>
        </header>
        <ul className="pipeline-evidence-cards" aria-label="Saved benchmark results">
          {evidence.map(({ model, value, measure, detail }) => (
            <li key={measure + model} className="pipeline-evidence-card">
              <h4>{model}</h4>
              <p className="pipeline-evidence-value">{value}</p>
              <p className="pipeline-evidence-measure">{measure}</p>
              <p className="pipeline-evidence-detail">{detail}</p>
            </li>
          ))}
        </ul>
        <p className="pipeline-statistics">Four feature models: Friedman p = {omnibus_friedman.p_value.toFixed(3)}; {pairwiseSummary}</p>
        <p className="pipeline-scope">Separate analyses and full-data RF / SHAP fits are outside this primary benchmark flow.</p>
      </section>
    </section>
  );
}
