import { useState, useEffect, useMemo, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import Card from './Card';
import InteractiveChart from './InteractiveChart';
import DataTable from './DataTable';
import ExportButton from './ExportButton';
import { ExportColumn } from '../utils/exportUtils';
import AdvancedFilters, { FilterOption, FilterValue } from './AdvancedFilters';
import DashboardWidget, { Widget } from './DashboardWidget';

interface AnalyticsDashboardProps {
  className?: string;
}

const WIDGET_STORAGE_KEY = 'analytics-dashboard-widgets';

const defaultWidgets: Widget[] = [
  {
    id: 'avg-response',
    type: 'metric',
    title: 'Learning Speed',
    size: 'small',
    position: { x: 0, y: 0 },
    data: { value: '0ms', label: 'Learning Speed', change: 0 }
  },
  {
    id: 'total-requests',
    type: 'metric',
    title: 'Training Examples',
    size: 'small',
    position: { x: 1, y: 0 },
    data: { value: '0', label: 'Training Examples', change: 0 }
  },
  {
    id: 'error-rate',
    type: 'metric',
    title: 'Learning Accuracy',
    size: 'small',
    position: { x: 2, y: 0 },
    data: { value: '0%', label: 'Learning Accuracy', change: 0 }
  },
  {
    id: 'uptime',
    type: 'metric',
    title: 'System Reliability',
    size: 'small',
    position: { x: 3, y: 0 },
    data: { value: '0%', label: 'System Reliability', change: 0 }
  }
];

const AnalyticsDashboard = ({ className = '' }: AnalyticsDashboardProps) => {
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [filterValues, setFilterValues] = useState<FilterValue[]>([]);
  const [widgets, setWidgets] = useState<Widget[]>(() => {
    if (typeof window === 'undefined') {
      return defaultWidgets;
    }

    try {
      const stored = localStorage.getItem(WIDGET_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Widget[];
        if (Array.isArray(parsed) && parsed.length) {
          return parsed;
        }
      }
    } catch {
      // ignore storage errors
    }

    return defaultWidgets;
  });

  // Fetch analytics data
  const { data: analyticsData, isLoading } = useQuery({
    queryKey: ['analytics', timeRange],
    queryFn: () => api.getAnalytics(timeRange),
    refetchInterval: 30000
  });

  const { data: performanceData } = useQuery({
    queryKey: ['performance-analytics', timeRange],
    queryFn: () => api.getPerformanceAnalytics(timeRange),
    refetchInterval: 60000
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem(WIDGET_STORAGE_KEY, JSON.stringify(widgets));
    } catch {
      // ignore storage failures
    }
  }, [widgets]);

  // Process data for charts
  const chartData = useMemo(() => {
    if (!analyticsData) return null;

    const timestamps = analyticsData.timestamps || [];
    const labels = timestamps.map(ts => new Date(ts * 1000).toLocaleTimeString());

    return {
      performance: {
        labels,
        datasets: [
          {
            label: 'Response Time (ms)',
            data: analyticsData.response_times || [],
            borderColor: '#4a9eff',
            backgroundColor: 'rgba(74, 158, 255, 0.1)',
            fill: true,
            tension: 0.4
          },
          {
            label: 'Throughput (req/s)',
            data: analyticsData.throughput || [],
            borderColor: '#00d4aa',
            backgroundColor: 'rgba(0, 212, 170, 0.1)',
            fill: true,
            tension: 0.4,
            yAxisID: 'y1'
          }
        ]
      },
      errors: {
        labels,
        datasets: [
          {
            label: 'Error Rate (%)',
            data: analyticsData.error_rates || [],
            borderColor: '#ff4757',
            backgroundColor: 'rgba(255, 71, 87, 0.1)',
            fill: true,
            tension: 0.4
          }
        ]
      },
      distribution: {
        labels: ['Success', 'Warning', 'Error', 'Critical'],
        datasets: [
          {
            label: 'Status Distribution',
            data: [
              analyticsData.success_count || 0,
              analyticsData.warning_count || 0,
              analyticsData.error_count || 0,
              analyticsData.critical_count || 0
            ],
            backgroundColor: [
              '#00d4aa',
              '#ffb800',
              '#ff4757',
              '#8b5cf6'
            ]
          }
        ]
      }
    };
  }, [analyticsData]);

  // Prepare table data
  const tableData = useMemo(() => {
    if (!performanceData) return [];

    return performanceData.top_performers?.map((item: any, index: number) => ({
      id: index + 1,
      name: item.name,
      performance: item.performance,
      throughput: item.throughput,
      errors: item.errors,
      uptime: item.uptime,
      lastUpdate: new Date(item.last_update * 1000).toLocaleString()
    })) || [];
  }, [performanceData]);

  const tableColumns: ExportColumn[] = [
    { key: 'name', title: 'Service' },
    { key: 'performance', title: 'Performance', render: (value) => `${value}%` },
    { key: 'throughput', title: 'Throughput', render: (value) => `${value} req/s` },
    { key: 'errors', title: 'Errors', render: (value) => value.toString() },
    { key: 'uptime', title: 'Uptime', render: (value) => `${value}%` },
    { key: 'lastUpdate', title: 'Last Update' }
  ];

  const filterOptions = useMemo<FilterOption[]>(() => [
    {
      key: 'name',
      label: 'Service Name',
      type: 'text',
      placeholder: 'Search by service'
    },
    {
      key: 'performance',
      label: 'Performance',
      type: 'number',
      min: 0,
      max: 100
    },
    {
      key: 'throughput',
      label: 'Throughput',
      type: 'number',
      min: 0
    },
    {
      key: 'errors',
      label: 'Errors',
      type: 'number',
      min: 0
    },
    {
      key: 'uptime',
      label: 'Uptime',
      type: 'number',
      min: 0,
      max: 100
    }
  ], []);

  const filteredTableData = useMemo(() => {
    if (!filterValues.length) return tableData;

    const compareNumber = (recordValue: number, filterValue: number, operator: FilterValue['operator']) => {
      switch (operator) {
        case 'gt':
          return recordValue > filterValue;
        case 'gte':
          return recordValue >= filterValue;
        case 'lt':
          return recordValue < filterValue;
        case 'lte':
          return recordValue <= filterValue;
        case 'equals':
        default:
          return recordValue === filterValue;
      }
    };

    return tableData.filter(row => {
      return filterValues.every(filter => {
        if (filter.value === null || filter.value === undefined || filter.value === '') {
          return true;
        }

        const value = row[filter.key as keyof typeof row];

        if (filter.key === 'name' && typeof value === 'string') {
          return value.toLowerCase().includes(String(filter.value).toLowerCase());
        }

        if (['performance', 'throughput', 'errors', 'uptime'].includes(filter.key)) {
          const numericValue = Number(value ?? 0);
          const target = Number(filter.value);
          if (Number.isNaN(numericValue) || Number.isNaN(target)) {
            return true;
          }
          return compareNumber(numericValue, target, filter.operator);
        }

        return true;
      });
    });
  }, [filterValues, tableData]);

  const summaryMetrics = useMemo(() => {
    if (!analyticsData) return [];

    return [
      {
        label: 'Learning Speed',
        value: `${analyticsData.avg_response_time || 0}ms`,
        change: analyticsData.response_time_change || 0,
        trend: analyticsData.response_time_change > 0 ? 'up' : 'down'
      },
      {
        label: 'Training Examples',
        value: analyticsData.total_requests?.toLocaleString() || '0',
        change: analyticsData.request_change || 0,
        trend: analyticsData.request_change > 0 ? 'up' : 'down'
      },
      {
        label: 'Learning Accuracy',
        value: `${100 - (analyticsData.error_rate || 0)}%`,
        change: -(analyticsData.error_rate_change || 0),
        trend: analyticsData.error_rate_change > 0 ? 'down' : 'up'
      },
      {
        label: 'System Reliability',
        value: `${analyticsData.uptime || 0}%`,
        change: analyticsData.uptime_change || 0,
        trend: analyticsData.uptime_change > 0 ? 'up' : 'down'
      }
    ];
  }, [analyticsData]);

  useEffect(() => {
    if (!analyticsData) return;

    setWidgets(prev => prev.map(widget => {
      switch (widget.id) {
        case 'avg-response': {
          const nextData = {
            value: `${analyticsData.avg_response_time || 0}ms`,
            label: 'Learning Speed',
            change: analyticsData.response_time_change || 0
          };
          if (
            widget.data?.value === nextData.value &&
            widget.data?.change === nextData.change
          ) {
            return widget;
          }
          return { ...widget, data: nextData };
        }
        case 'total-requests': {
          const nextData = {
            value: analyticsData.total_requests?.toLocaleString() || '0',
            label: 'Training Examples',
            change: analyticsData.request_change || 0
          };
          if (
            widget.data?.value === nextData.value &&
            widget.data?.change === nextData.change
          ) {
            return widget;
          }
          return { ...widget, data: nextData };
        }
        case 'error-rate': {
          const nextData = {
            value: `${100 - (analyticsData.error_rate || 0)}%`,
            label: 'Learning Accuracy',
            change: -(analyticsData.error_rate_change || 0)
          };
          if (
            widget.data?.value === nextData.value &&
            widget.data?.change === nextData.change
          ) {
            return widget;
          }
          return { ...widget, data: nextData };
        }
        case 'uptime': {
          const nextData = {
            value: `${analyticsData.uptime || 0}%`,
            label: 'System Reliability',
            change: analyticsData.uptime_change || 0
          };
          if (
            widget.data?.value === nextData.value &&
            widget.data?.change === nextData.change
          ) {
            return widget;
          }
          return { ...widget, data: nextData };
        }
        default:
          return widget;
      }
    }));
  }, [analyticsData]);

  const handleWidgetMove = useCallback((widgetId: string, newPosition: { x: number; y: number }) => {
    setWidgets(prev => {
      const index = prev.findIndex(widget => widget.id === widgetId);
      if (index === -1) return prev;

      const newIndex = Math.min(prev.length - 1, Math.max(0, newPosition.y * 4 + newPosition.x));
      const updated = [...prev];
      const [moved] = updated.splice(index, 1);
      updated.splice(newIndex, 0, { ...moved, position: newPosition });
      return updated;
    });
  }, []);

  const handleWidgetResize = useCallback((widgetId: string, newSize: Widget['size']) => {
    setWidgets(prev => prev.map(widget => widget.id === widgetId ? { ...widget, size: newSize } : widget));
  }, []);

  const handleWidgetRemove = useCallback((widgetId: string) => {
    setWidgets(prev => prev.filter(widget => widget.id !== widgetId));
  }, []);

  const handleWidgetEdit = useCallback((widgetId: string, config: Record<string, any>) => {
    setWidgets(prev => prev.map(widget => {
      if (widget.id !== widgetId) return widget;
      const newTitle = typeof window !== 'undefined' ? window.prompt('Update widget title', widget.title) : null;
      if (!newTitle) return widget;
      return { ...widget, title: newTitle, config };
    }));
  }, []);

  const handleAddWidget = useCallback(() => {
    setWidgets(prev => {
      const nextIndex = prev.length;
      return [
        ...prev,
        {
          id: `custom-${Date.now()}`,
          type: 'text',
          title: 'Custom Notes',
          size: 'medium',
          position: {
            x: nextIndex % 4,
            y: Math.floor(nextIndex / 4)
          },
          data: {
            content: 'Use the edit action to rename or resize this widget.'
          }
        }
      ];
    });
  }, []);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Training Reports
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Detailed insights into your AI's learning progress and performance
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="form-select"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>

          <ExportButton
            data={filteredTableData}
            columns={tableColumns}
            options={{
              filename: `analytics-${timeRange}-${new Date().toISOString().split('T')[0]}`,
              title: 'Analytics Report',
              subtitle: `Data for ${timeRange} period`
            }}
          />
          <button
            onClick={handleAddWidget}
            className="btn btn-secondary"
            type="button"
          >
            Add Widget
          </button>
        </div>
      </div>

      {/* Customizable Widgets */}
      <Card title="Custom Dashboard" subtitle="Drag, resize, and personalize analytics widgets">
        <DashboardWidget
          widgets={widgets}
          onWidgetMove={handleWidgetMove}
          onWidgetResize={handleWidgetResize}
          onWidgetRemove={handleWidgetRemove}
          onWidgetEdit={handleWidgetEdit}
          editable
        />
      </Card>

      {/* Filters */}
      <AdvancedFilters
        filters={filterOptions}
        values={filterValues}
        onChange={setFilterValues}
        onClear={() => setFilterValues([])}
        className="border border-gray-200 dark:border-gray-800"
      />

      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryMetrics.map((metric, index) => (
          <Card key={index} className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {metric.label}
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metric.value}
                </p>
              </div>
              <div className={`flex items-center gap-1 text-sm ${
                metric.trend === 'up' ? 'text-green-600' : 'text-red-600'
              }`}>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={
                    metric.trend === 'up' ? "M7 17l9.2-9.2M17 17V7H7" : "M7 7l9.2 9.2M7 7v10h10"
                  } />
                </svg>
                <span>{Math.abs(metric.change)}%</span>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Learning Performance" subtitle="Training speed and processing efficiency over time">
          {chartData?.performance && (
            <InteractiveChart
              type="line"
              data={chartData.performance}
              height={300}
              exportable
              options={{
                scales: {
                  y: {
                    title: { text: 'Processing Time (ms)' }
                  },
                  y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { text: 'Learning Rate (examples/s)' },
                    grid: { drawOnChartArea: false }
                  }
                }
              }}
            />
          )}
        </Card>

        <Card title="Learning Accuracy" subtitle="Success rate and error trends over time">
          {chartData?.errors && (
            <InteractiveChart
              type="line"
              data={chartData.errors}
              height={300}
              exportable
              options={{
                scales: {
                  y: {
                    title: { text: 'Accuracy (%)' },
                    min: 0,
                    max: 100
                  }
                }
              }}
            />
          )}
        </Card>
      </div>

      {/* Learning Outcomes Distribution */}
      <Card title="Learning Outcomes" subtitle="Distribution of successful vs unsuccessful learning attempts">
        {chartData?.distribution && (
          <InteractiveChart
            type="doughnut"
            data={chartData.distribution}
            height={300}
            exportable
            options={{
              plugins: {
                legend: {
                  position: 'bottom'
                }
              }
            }}
          />
        )}
      </Card>

      {/* Performance Table */}
      <Card title="Learning Models" subtitle="Performance of different learning approaches">
        <DataTable
          data={filteredTableData}
          columns={[
            { key: 'name', title: 'Learning Model', dataIndex: 'name' },
            { key: 'performance', title: 'Accuracy', dataIndex: 'performance', render: (value) => `${value}%` },
            { key: 'throughput', title: 'Learning Rate', dataIndex: 'throughput', render: (value) => `${value} examples/s` },
            { key: 'errors', title: 'Mistakes', dataIndex: 'errors' },
            { key: 'uptime', title: 'Reliability', dataIndex: 'uptime', render: (value) => `${value}%` },
            { key: 'lastUpdate', title: 'Last Trained', dataIndex: 'lastUpdate' }
          ]}
          loading={isLoading}
          pagination={{
            current: 1,
            pageSize: 10,
            total: filteredTableData.length,
            showSizeChanger: true,
            showTotal: (total, range) => `Showing ${range[0]}-${range[1]} of ${total} services`
          }}
          rowKey="id"
          size="small"
          bordered
          striped
        />
      </Card>
    </div>
  );
};

export default AnalyticsDashboard;
