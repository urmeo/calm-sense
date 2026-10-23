import type { ChartOption } from '../components/Chart';

export function barChartOption({ labels, values, colors = ['#3182CE'], horizontal = false,
  axisName = '', maximum, percent = false, precision = 3, showValues = false }: {
  labels: string[];
  values: (number | null)[];
  colors?: string[];
  horizontal?: boolean;
  axisName?: string;
  maximum?: number;
  percent?: boolean;
  precision?: number;
  showValues?: boolean;
}): ChartOption {
  const format = (value: number) => `${value.toFixed(precision)}${percent ? '%' : ''}`;
  const category = {
    type: 'category' as const,
    data: labels,
    inverse: horizontal,
    axisLabel: horizontal ? { fontSize: 11, width: 175, overflow: 'truncate' as const } : {
      fontSize: 11, interval: 0, rotate: labels.length > 3 ? 30 : 0,
    },
  };
  const numeric = {
    type: 'value' as const, min: 0, max: maximum, name: axisName,
    nameLocation: 'middle' as const, nameGap: 35,
    axisLabel: { formatter: (value: number) => percent ? `${value}%` : String(value) },
    splitLine: { lineStyle: { type: 'dashed' as const, opacity: 0.3 } },
  };
  return {
    grid: { left: horizontal ? 185 : 50, right: showValues ? 40 : 20, top: 20, bottom: horizontal ? 45 : labels.length > 3 ? 80 : 55 },
    tooltip: { trigger: 'axis', confine: true, renderMode: 'richText', valueFormatter: (value) =>
      typeof value === 'number' ? format(value) : 'Unavailable' },
    xAxis: horizontal ? numeric : category,
    yAxis: horizontal ? category : numeric,
    series: [{
      name: axisName || 'Value', type: 'bar',
      data: values.map((value, i) => ({ value, itemStyle: {
        color: colors[i % colors.length], borderRadius: horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0],
      } })),
      label: { show: showValues, position: horizontal ? 'right' : 'top', formatter: (item) =>
        typeof item.value === 'number' ? format(item.value) : '' },
    }],
  };
}
