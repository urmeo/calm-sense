import { Heart, BookOpen, Database, Cpu, Shield, ExternalLink, ArrowRight, Activity, Layers, Users, BarChart3 } from 'lucide-react';
import { Link } from 'react-router-dom';
import results, { shapContext } from '../data';
import Panel from '../components/Panel';

const details = [
  {
    title: 'Dataset', icon: Database,
    items: ['WESAD: Wearable Stress and Affect Detection', `${results.binary.n_subjects === undefined ? 'Subject count not recorded' : `${results.binary.n_subjects} laboratory subjects`}; chest ECG, EDA, temperature, respiration, motion`, results.multiclass.classes.join(', ')],
  },
  {
    title: 'Models', icon: Cpu,
    items: [results.binary.models.map(({ model }) => model).join(', '), `Saved inference model: ${results.binary.inference_model ?? 'not recorded'}`, 'Imputation and scaling fit on the training subjects in each LOSO fold'],
  },
  {
    title: 'Tools', icon: Cpu,
    items: ['Python, NumPy, pandas, SciPy', 'scikit-learn, XGBoost, LightGBM; PyTorch for the 1D-CNN', 'NeuroKit2 for physiological features; React and ECharts for this dashboard'],
  },
  {
    title: 'Features', icon: BookOpen,
    items: [`${results.binary.n_features} saved features; 60 s windows, 50% overlap`, 'HRV, EDA, temperature, respiration, motion', 'EDA tonic/phasic decomposition'],
  },
  {
    title: 'Explainability', icon: Shield,
    items: [shapContext.title, shapContext.description, 'Descriptive values; no held-out or causal interpretation'],
  },
];

const pipeline = [
  { title: 'Recordings', icon: Database, description: 'WESAD chest signals: ECG, EDA, temperature, respiration, and motion.' },
  { title: 'Prepare windows', icon: Activity, description: 'Filter signals. Keep 60 s windows with 50% overlap and at least 90% label purity.' },
  { title: 'Model inputs', icon: Layers, description: `${results.binary.n_features} features for four classifiers; 5 × 1,024 signal samples for the 1D-CNN.` },
  { title: 'LOSO evaluation', icon: Users, description: 'Hold out one person. Fit preprocessing and models on the other subjects. Repeat for each person.' },
  { title: 'Saved results', icon: BarChart3, description: 'Save subject scores, pooled metrics, and calibration analyses for the dashboard.' },
];

const repositoryUrl = 'https://github.com/urmeo/calm-sense';
const links = [
  ['Repository', repositoryUrl],
  ['Data flow', `${repositoryUrl}#data-flow`],
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
        <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
          This dashboard displays saved research results and signal clips.
          It does not run predictions or diagnose stress.
        </p>
        <Link to="/models" className="inline-block mt-4 text-blue-600 dark:text-blue-400 underline">
          Compare model accuracy, macro-F1, and transfer results
        </Link>
      </Panel>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {details.map(({ title, icon: Icon, items }) => (
          <Panel key={title} title={<span className="flex items-center gap-3"><Icon className="w-6 h-6 text-blue-500" />{title}</span>}>
            <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-400">
              {items.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </Panel>
        ))}
      </div>

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
