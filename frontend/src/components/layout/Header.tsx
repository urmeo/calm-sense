import { Menu } from 'lucide-react';

export default function Header({ sidebarOpen, onMenuClick }: {
  sidebarOpen: boolean;
  onMenuClick: () => void;
}) {
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between px-4 py-3">
        <button
          onClick={onMenuClick}
          aria-label="Open navigation menu"
          aria-expanded={sidebarOpen}
          aria-controls="navigation-sidebar"
          className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <Menu className="w-6 h-6 text-gray-600 dark:text-gray-300" />
        </button>

        <span className="hidden sm:flex text-sm text-gray-500 dark:text-gray-400">Saved experiment results</span>
      </div>
    </header>
  );
}
