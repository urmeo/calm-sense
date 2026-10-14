import results from '../data';
import { formatPercent } from '../lib/benchmarks';
import './ResearchPipeline.css';

const { binary, multiclass } = results;
const people = binary.n_subjects;
const randomForest = binary.models.find(({ model }) => model === 'Random Forest');
const lightGBM = multiclass.models.find(({ model }) => model === 'LightGBM');
const evidence = [
  {
    model: 'Random Forest · binary',
    value: randomForest?.accuracy_mean,
    measure: 'Mean accuracy across people',
  },
  {
    model: 'Random Forest · binary',
    value: randomForest?.balanced_accuracy,
    measure: 'Pooled balanced accuracy',
  },
  {
    model: 'LightGBM · three-class',
    value: lightGBM?.accuracy_mean,
    measure: 'Mean accuracy across people',
  },
];

export default function ResearchPipeline() {
  return (
    <section className="research-pipeline" aria-labelledby="pipeline-title">
      <header className="pipeline-intro">
        <p className="pipeline-kicker">Research pipeline</p>
        <h2 id="pipeline-title">From recorded signals to held-out results</h2>
        <p>Baseline and stress; add amusement for the three-class task.</p>
      </header>

      <ol className="pipeline-steps" aria-label="Research pipeline in five steps" role="list">
        <li className="pipeline-step">
          <span className="pipeline-step-number" aria-hidden="true">1</span>
          <div>
            <h3>Recorded signals</h3>
            <p>WESAD chest · {people ? `${people} people` : 'recorded people'} · 700 Hz.</p>
            <p>ECG, skin conductance, temperature, respiration, and motion.</p>
          </div>
        </li>

        <li className="pipeline-step">
          <span className="pipeline-step-number" aria-hidden="true">2</span>
          <div>
            <h3>Prepare windows</h3>
            <p>Filter signals. Use 60-second windows with a 30-second step.</p>
            <p>Keep windows with at least 90% label agreement.</p>
          </div>
        </li>

        <li className="pipeline-step pipeline-step-training">
          <span className="pipeline-step-number" aria-hidden="true">3</span>
          <div>
            <h3>{people ? `Train on ${people - 1} people` : 'Train on the remaining people'}</h3>
            <p>Reserve one complete person before fitting. Compare five classifiers in two independent paths.</p>
            <div className="pipeline-models">
              <article className="pipeline-model">
                <h4>Four feature classifiers</h4>
                <p>{binary.n_features} features · Logistic Regression, Random Forest, XGBoost, and LightGBM.</p>
                <p>Fit median imputation and scaling on training people only.</p>
              </article>
              <article className="pipeline-model">
                <h4>1D CNN</h4>
                <p>5 channels × 1,024 samples. Use separate inner training and validation people.</p>
                <p>Normalize with inner training windows only. Stop early using validation loss.</p>
              </article>
            </div>
          </div>
        </li>

        <li className="pipeline-step pipeline-step-test">
          <span className="pipeline-step-number" aria-hidden="true">4</span>
          <div>
            <h3>Test one unseen person</h3>
            <p>Apply the fitted transforms and models to the reserved person.</p>
            <p>No fitting, tuning, validation, or early stopping on this person.</p>
          </div>
        </li>

        <li className="pipeline-step">
          <span className="pipeline-step-number" aria-hidden="true">5</span>
          <div>
            <h3>Score and save</h3>
            <p>{people ? `Repeat for all ${people} held-out people.` : 'Repeat for every held-out person.'} Accuracy and macro-F1 are means across people. Balanced accuracy pools held-out predictions.</p>
            <p>Save JSON/CSV results and provenance. Build plots and the static dashboard. No live inference.</p>
          </div>
        </li>
      </ol>

      <section className="pipeline-evidence" aria-labelledby="pipeline-evidence-title">
        <h3 id="pipeline-evidence-title">Saved benchmark results</h3>
        <ul className="pipeline-evidence-cards" aria-label="Saved benchmark results" role="list">
          {evidence.map(({ model, value, measure }) => (
            <li key={model + measure} className="pipeline-evidence-card">
              <h4>{model}</h4>
              <p className="pipeline-evidence-value">{formatPercent(value)}</p>
              <p>{measure}</p>
            </li>
          ))}
        </ul>
      </section>
    </section>
  );
}
