import { type ReactNode } from 'react';

export default function Panel({ title, children }: { title?: ReactNode; children: ReactNode }) {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      {title && <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">{title}</h2>}
      {children}
    </section>
  );
}
