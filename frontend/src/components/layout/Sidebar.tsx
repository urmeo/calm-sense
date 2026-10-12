import React, { useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Activity,
  FileSearch,
  BarChart3,
  Gauge,
  Info,
  Sun,
  Moon,
  X,
  Heart,
} from 'lucide-react';
import { containNavigationFocus } from '../../lib/navigation';
import { dashboardRoutes } from '../../lib/routes';

const routeIcons = {
  '/': LayoutDashboard,
  '/signals': Activity,
  '/explain': FileSearch,
  '/models': BarChart3,
  '/calibration': Gauge,
  '/about': Info,
};

const Sidebar: React.FC<{
  hasCalibration: boolean;
  isOpen: boolean;
  onClose: () => void;
  darkMode: boolean;
  toggleDarkMode: () => void;
}> = ({ hasCalibration, isOpen, onClose, darkMode, toggleDarkMode }) => {
  const sidebar = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!isOpen) return;
    const desktop = window.matchMedia('(min-width: 1024px)');
    if (desktop.matches) {
      onClose();
      return;
    }
    const background = document.getElementById('dashboard-content');
    if (!sidebar.current || !background) return;
    const releaseFocus = containNavigationFocus(sidebar.current, background, onClose);
    const onResize = () => { if (desktop.matches) onClose(); };
    desktop.addEventListener('change', onResize);
    return () => {
      desktop.removeEventListener('change', onResize);
      releaseFocus();
    };
  }, [isOpen, onClose]);

  const navItems = dashboardRoutes.filter(({ path }) => hasCalibration || path !== '/calibration');
  const { pathname } = useLocation();
  const currentPath = pathname.replace(/\/+$/, '') || '/';
  const ThemeIcon = darkMode ? Sun : Moon;

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-20 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        ref={sidebar}
        id="navigation-sidebar"
        role={isOpen ? 'dialog' : undefined}
        aria-modal={isOpen ? true : undefined}
        aria-label="Navigation"
        className={`
          fixed top-0 left-0 z-30 h-full w-64 flex flex-col
          bg-primary-700 dark:bg-gray-900
          transform transition-transform duration-300 ease-in-out
          lg:translate-x-0 lg:static
          ${isOpen ? 'translate-x-0 visible' : '-translate-x-full invisible lg:visible'}
        `}
      >
        <div className="flex items-center justify-between shrink-0 p-4 border-b border-primary-600 dark:border-gray-700">
          <div className="flex items-center space-x-2">
            <Heart className="w-8 h-8 text-red-400" />
            <span className="text-xl font-bold text-white">CalmSense</span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close navigation menu"
            className="lg:hidden text-white hover:text-gray-300"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <nav aria-label="Main navigation" className="flex-1 overflow-y-auto p-4 space-y-2">
          {navItems.map((item) => {
            const isActive = currentPath === item.path;
            const Icon = routeIcons[item.path];

            return (
              <Link
                key={item.path}
                to={item.path}
                aria-current={isActive ? 'page' : undefined}
                onClick={onClose}
                className={`
                  flex items-center space-x-3 px-4 py-3 rounded-lg
                  transition-colors duration-200
                  ${isActive
                    ? 'bg-accent-500 text-white'
                    : 'text-gray-300 hover:bg-primary-600 dark:hover:bg-gray-800'
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="shrink-0 p-4">
          <button
            onClick={toggleDarkMode}
            className="flex items-center justify-center w-full px-4 py-3
                       bg-primary-600 dark:bg-gray-800 rounded-lg
                       text-white hover:bg-primary-500 dark:hover:bg-gray-700
                       transition-colors duration-200"
          >
            <ThemeIcon className="w-5 h-5 mr-2" />
            <span>{darkMode ? 'Light Mode' : 'Dark Mode'}</span>
          </button>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
