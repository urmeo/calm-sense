import React, { lazy, Suspense, useState, useEffect, useCallback, useRef } from 'react';
import { BrowserRouter as Router, Link, Routes, Route, useLocation } from 'react-router-dom';

import Dashboard from './pages/Dashboard';
import ModelComparison from './pages/ModelComparison';
import About from './pages/About';
import ErrorBoundary from './components/ErrorBoundary';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import results from './data';
import { readDarkMode, saveDarkMode } from './lib/preferences';
import { benchmarkStatus } from './lib/benchmarks';
import { dashboardRoutes } from './lib/routes';
import { generated_at, provenance_kind } from '../../outputs/results/provenance.json';

const SignalExplorer = lazy(() => import('./pages/SignalExplorer'));
const ExplainabilityDashboard = lazy(() => import('./pages/ExplainabilityDashboard'));
const CalibrationPanel = lazy(() => import('./pages/CalibrationPanel'));

const pages = {
  '/': Dashboard,
  '/signals': SignalExplorer,
  '/explain': ExplainabilityDashboard,
  '/models': ModelComparison,
  '/calibration': CalibrationPanel,
  '/about': About,
};
const hasCalibration = Boolean(results.calibration);
const availableRoutes = dashboardRoutes.filter(({ path }) => hasCalibration || path !== '/calibration');
const status = benchmarkStatus(results,
  provenance_kind === 'validated_reproduction' ? generated_at : undefined);

function PageContent({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation();
  const content = useRef<HTMLElement>(null);
  const previousPath = useRef(pathname);

  useEffect(() => {
    if (previousPath.current === pathname) return;
    previousPath.current = pathname;
    content.current?.scrollTo(0, 0);
    content.current?.focus({ preventScroll: true });
  }, [pathname]);

  return <main ref={content} tabIndex={-1} className="flex-1 overflow-y-auto p-4 lg:p-6">{children}</main>;
}

const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(readDarkMode);

  useEffect(() => {
    saveDarkMode(darkMode);
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode((current) => !current);
  const closeSidebar = useCallback(() => setSidebarOpen(false), []);

  return (
    <Router basename={import.meta.env.BASE_URL}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="flex h-screen overflow-hidden">
          <Sidebar
            hasCalibration={hasCalibration}
            isOpen={sidebarOpen}
            onClose={closeSidebar}
            darkMode={darkMode}
            toggleDarkMode={toggleDarkMode}
          />

          <div id="dashboard-content" className="flex-1 flex flex-col overflow-hidden">
            <Header sidebarOpen={sidebarOpen} onMenuClick={() => setSidebarOpen(true)} />

            <PageContent>
              <aside aria-label="Benchmark provenance" className="mb-6 space-y-2 text-sm">
                <p role="note" className={`rounded-lg border p-3 ${status.historical
                  ? 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-200'
                  : 'border-gray-200 bg-white text-gray-600 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
                  {status.primary}
                </p>
                {status.ancillary && (
                  <p role="note" className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-amber-900 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-200">
                    {status.ancillary}
                  </p>
                )}
              </aside>
              <Suspense fallback={<p role="status" className="text-gray-600 dark:text-gray-300">Loading charts…</p>}>
                <Routes>
                  {availableRoutes.map(({ path }) => {
                    const Page = pages[path];
                    return <Route key={path} path={path} element={<ErrorBoundary key={path}><Page /></ErrorBoundary>} />;
                  })}
                  <Route path="*" element={(
                    <div className="space-y-3 text-gray-900 dark:text-white">
                      <h1 className="text-2xl font-bold">Page not found</h1>
                      <Link to="/" className="text-blue-600 dark:text-blue-400 underline">Return to dashboard</Link>
                    </div>
                  )} />
                </Routes>
              </Suspense>
            </PageContent>

            <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-4 py-3">
              <div className="text-center text-sm text-gray-500 dark:text-gray-400">
                CalmSense · Saved WESAD research results · By{' '}
                <a href="https://github.com/urmeo" className="underline hover:text-blue-600 dark:hover:text-blue-400">urme</a>
              </div>
            </footer>
          </div>
        </div>
      </div>
    </Router>
  );
};

export default App;
