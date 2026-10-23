import type { ReactNode } from 'react';

interface ResultRow {
  key: string;
  cells: ReactNode[];
  highlighted?: boolean;
}

export default function ResultsTable({ columns, rows }: { columns: string[]; rows: ResultRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700 text-left text-gray-600 dark:text-gray-300">
            {columns.map((column) => <th key={column} scope="col" className="px-4 py-2">{column}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ key, cells, highlighted }) => (
            <tr key={key} className={`border-b border-gray-100 dark:border-gray-700 ${highlighted ? 'bg-green-50 dark:bg-green-900/20' : ''}`}>
              {cells.map((cell, index) => (
                <td key={columns[index]} className={`px-4 py-2 ${index === 0 ? 'font-medium text-gray-900 dark:text-white' : ''}`}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
