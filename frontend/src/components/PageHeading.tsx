import type { ReactNode } from 'react';

export default function PageHeading({ title, icon, children }: {
  title: string;
  icon?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div>
      <h1 className={`text-2xl font-bold text-gray-900 dark:text-white ${icon ? 'flex items-center gap-2' : ''}`}>
        {icon}{title}
      </h1>
      <p className="text-gray-500 dark:text-gray-400">{children}</p>
    </div>
  );
}
