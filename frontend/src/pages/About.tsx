import { Heart, BookOpen, Cpu, ExternalLink, BarChart3 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { shapContext } from '../data';
import Panel from '../components/Panel';
import ResearchPipeline from '../components/ResearchPipeline';

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
    <div className="about-page space-y-8 animate-fade-in mx-auto">
      <div className="text-center">
        <div className="flex items-center justify-center space-x-3 mb-4">
          <Heart className="w-12 h-12 text-red-500" />
          <h1 className="text-4xl font-bold text-gradient">CalmSense</h1>
        </div>
        <p className="text-xl text-gray-600 dark:text-gray-400">Wearable stress classification research</p>
      </div>

      <div className="space-y-3">
        <ResearchPipeline />
        <p className="pipeline-caption text-gray-600 dark:text-gray-400">
          <a href="https://www.figma.com/design/tfPEQsLyRoAj9yT3HrPaob?node-id=3-2" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 underline">View the pipeline in Figma</a>
          {' · '}Binary ROC/AUC curves cover the four feature models. CNN AUROC/AUPRC was not computed.
        </p>
      </div>

      <Panel title="Separate research runs">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">These experiments extend the benchmark with their own splits and evaluation conditions.</p>
          <ul className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 210px), 1fr))' }} aria-label="Additional research analyses">
            {analyses.map(({ title, items }) => (
              <li key={title} className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{title}</h3>
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
