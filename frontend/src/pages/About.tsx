import { Heart, BookOpen, Database, Cpu, ExternalLink, ArrowRight, Activity, Layers, Users, BarChart3 } from 'lucide-react';
import { Link } from 'react-router-dom';
import results, { shapContext } from '../data';
import Panel from '../components/Panel';

const pipeline = [
  { title: 'Recordings', icon: Database, description: '15 WESAD subjects. Chest ECG, EDA, temperature, respiration, and motion.' },
  { title: 'Prepare windows', icon: Activity, description: 'Filter signals. Keep 60 s windows with 50% overlap and at least 90% label purity.' },
  { title: 'Two model paths', icon: Layers, description: `${results.binary.n_features} features for four classifiers, or 5 channels × 1,024 samples for the CNN.` },
  { title: 'Hold out one person', icon: Users, description: 'Fit using the other subjects. Test the held-out person. Repeat for all 15 subjects.' },
  { title: 'Held-out scores', icon: BarChart3, description: 'Report subject scores, pooled metrics, and binary ROC/AUC for the four feature models.' },
];

const foldSteps = [
  {
    title: 'Training pool · 14 subjects',
    items: [
      'Feature models: fit median imputation and scaling, then Logistic Regression, Random Forest, XGBoost, or LightGBM.',
      'CNN: split the training pool by subject for training and validation. Normalize with inner training windows only.',
    ],
  },
  {
    title: 'Test · 1 untouched subject',
    items: [
      'Apply the fitted transforms and model to this person’s eligible windows.',
      'Keep this person outside model fitting, validation, and global calibration.',
    ],
  },
  {
    title: 'Repeat · 15 folds',
    items: [
      'Average accuracy and macro-F1 across subjects. Pool held-out predictions for balanced accuracy.',
      'Binary ROC curves and AUROC/AUPRC scores use the four feature models. CNN AUROC/AUPRC was not computed.',
    ],
  },
];

const analyses = [
  {
    title: 'Calibration and personalization',
    items: [
      'Global calibration of binary Random Forest probabilities: fit to grouped out-of-fold predictions from training subjects inside each LOSO fold.',
      'Personalization: use disjoint, non-overlapping enrollment and evaluation windows from the held-out person. Recalibrate probabilities; keep the classifier fixed.',
    ],
  },
  {
    title: 'Robustness',
    items: [
      'Rerun feature ablations and compare the separate chest and wrist feature pipelines.',
      'Test transfer between WESAD wrist and PhysioNet Non-EEG with 18 shared features. Devices, labels, and units differ.',
    ],
  },
  {
    title: 'Evaluation checks',
    items: [
      'Bootstrap subjects and compare feature models with Friedman and Holm-corrected paired Wilcoxon tests.',
      'Compare matched LOSO and subject-mixed splits. Run tuning separately, with grouped inner folds inside outer LOSO.',
    ],
  },
];

const repositoryUrl = 'https://github.com/urmeo/calm-sense';
const links = [
  ['Repository', repositoryUrl],
  ['Pipeline', `${repositoryUrl}#pipeline`],
  ['Methodology', `${repositoryUrl}/blob/main/docs/methodology.md`],
  ['References', `${repositoryUrl}#references`],
  ['License', `${repositoryUrl}/blob/main/LICENSE`],
];

export default function About() {
  return (
    <div className="space-y-8 animate-fade-in max-w-4xl mx-auto">
      <div className="text-center">
        <div className="flex items-center justify-center space-x-3 mb-4">
          <Heart className="w-12 h-12 text-red-500" />
          <h1 className="text-4xl font-bold text-gradient">CalmSense</h1>
        </div>
        <p className="text-xl text-gray-600 dark:text-gray-400">Wearable stress classification research</p>
      </div>

      <Panel title="Research pipeline">
        <ol className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 140px), 1fr))' }} aria-label="Research pipeline in five steps">
          {pipeline.map(({ title, icon: Icon, description }, index) => (
            <li key={title} className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3 text-blue-600 dark:text-blue-400">
                <Icon aria-hidden="true" className="w-5 h-5" />
                <span className="font-mono text-sm">{String(index + 1).padStart(2, '0')}</span>
                {index < pipeline.length - 1 && <ArrowRight aria-hidden="true" className="w-4 h-4 ml-auto" />}
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{title}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
            </li>
          ))}
        </ol>

        <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <h3 className="font-semibold text-gray-900 dark:text-white mb-3">How leave-one-subject-out (LOSO) evaluation works</h3>
          <ol className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 210px), 1fr))' }} aria-label="Training, test, and aggregation">
            {foldSteps.map(({ title, items }, index) => (
              <li key={title}>
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
                  <span className="font-mono mr-2">{index + 1}.</span>{title}
                </h4>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  {items.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </li>
            ))}
          </ol>
        </div>

        <div className="mt-4">
          <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Separate research runs</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">These experiments extend the benchmark with their own splits and evaluation conditions.</p>
          <ul className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 210px), 1fr))' }} aria-label="Additional research analyses">
            {analyses.map(({ title, items }) => (
              <li key={title} className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">{title}</h4>
                <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  {items.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-4">
          <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Separate fits and saved outputs</h3>
          <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 280px), 1fr))' }}>
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h4 className="flex items-center gap-2 font-semibold text-gray-900 dark:text-white mb-2"><Cpu aria-hidden="true" className="w-5 h-5" />Full-data fits</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">In the saved run, an uncalibrated Random Forest was fitted on all binary windows for export. It uses fixed defaults, separate from the tuning experiment.</p>
              <p className="text-sm text-gray-600 dark:text-gray-400"><span className="font-medium">{shapContext.title}.</span> {shapContext.description} These contributions describe that fit; they are not held-out or causal estimates.</p>
            </div>
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h4 className="flex items-center gap-2 font-semibold text-gray-900 dark:text-white mb-2"><BarChart3 aria-hidden="true" className="w-5 h-5" />Static dashboard</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Saved JSON and CSV results become charts and tables. Recorded signal clips provide examples.</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">The browser displays these saved outputs. It does not run predictions or diagnose stress.</p>
              <Link to="/models" className="inline-block mt-4 text-sm text-blue-600 dark:text-blue-400 underline">Compare model accuracy, macro-F1, and transfer results</Link>
            </div>
          </div>
        </div>
      </Panel>

      <Panel title={<span className="flex items-center gap-3"><BookOpen aria-hidden="true" className="w-6 h-6 text-blue-500" />Core tools</span>}>
        <ul className="grid gap-4 text-sm text-gray-600 dark:text-gray-400" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 210px), 1fr))' }}>
          <li><span className="font-medium text-gray-900 dark:text-white">Signals and features.</span> Python, NumPy, pandas, SciPy, and NeuroKit2.</li>
          <li><span className="font-medium text-gray-900 dark:text-white">Models.</span> scikit-learn, XGBoost, and LightGBM; PyTorch for the 1D-CNN.</li>
          <li><span className="font-medium text-gray-900 dark:text-white">Dashboard.</span> React and ECharts display the saved research outputs.</li>
        </ul>
      </Panel>

      <nav aria-label="Project resources" className="flex flex-wrap justify-center gap-4">
        {links.map(([label, href]) => (
          <a key={label} href={href} target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-2 text-blue-600 dark:text-blue-400 underline">
            {label}<ExternalLink aria-hidden="true" className="w-4 h-4" />
          </a>
        ))}
      </nav>
    </div>
  );
}
