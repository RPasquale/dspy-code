import { useMemo, useState, useRef, useEffect } from 'react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement,
} from 'chart.js';
import { ensureChartsRegistered } from '../lib/registerCharts';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement
);

ensureChartsRegistered();

export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    borderColor?: string;
    backgroundColor?: string;
    fill?: boolean;
    tension?: number;
    pointRadius?: number;
    pointHoverRadius?: number;
    pointBackgroundColor?: string;
    pointBorderColor?: string;
    pointBorderWidth?: number;
    pointHoverBackgroundColor?: string;
    pointHoverBorderColor?: string;
    pointHoverBorderWidth?: number;
    yAxisID?: string;
  }[];
}

export interface ChartOptions {
  responsive?: boolean;
  maintainAspectRatio?: boolean;
  interaction?: {
    intersect?: boolean;
    mode?: 'index' | 'point' | 'nearest' | 'x' | 'y' | 'dataset';
  };
  plugins?: {
    legend?: {
      display?: boolean;
      position?: 'top' | 'bottom' | 'left' | 'right';
      labels?: {
        usePointStyle?: boolean;
        padding?: number;
        color?: string;
        font?: {
          size?: number;
          family?: string;
        };
      };
    };
    title?: {
      display?: boolean;
      text?: string;
      color?: string;
      font?: {
        size?: number;
        family?: string;
      };
    };
    tooltip?: {
      enabled?: boolean;
      mode?: 'index' | 'point' | 'nearest' | 'x' | 'y' | 'dataset';
      intersect?: boolean;
      backgroundColor?: string;
      titleColor?: string;
      bodyColor?: string;
      borderColor?: string;
      borderWidth?: number;
      cornerRadius?: number;
      displayColors?: boolean;
      callbacks?: {
        title?: (context: any[]) => string;
        label?: (context: any) => string;
        afterLabel?: (context: any) => string;
        footer?: (context: any[]) => string;
      };
    };
  };
  scales?: {
    x?: {
      display?: boolean;
      title?: {
        display?: boolean;
        text?: string;
        color?: string;
      };
      ticks?: {
        color?: string;
        maxRotation?: number;
        minRotation?: number;
        font?: {
          size?: number;
          family?: string;
        };
      };
      grid?: {
        display?: boolean;
        color?: string;
        drawBorder?: boolean;
        drawOnChartArea?: boolean;
      };
    };
    y?: {
      display?: boolean;
      position?: 'left' | 'right';
      title?: {
        display?: boolean;
        text?: string;
        color?: string;
      };
      ticks?: {
        color?: string;
        font?: {
          size?: number;
          family?: string;
        };
        callback?: (value: any) => string;
      };
      grid?: {
        display?: boolean;
        color?: string;
        drawBorder?: boolean;
        drawOnChartArea?: boolean;
      };
    };
    y1?: {
      display?: boolean;
      position?: 'left' | 'right';
      title?: {
        display?: boolean;
        text?: string;
        color?: string;
      };
      ticks?: {
        color?: string;
        font?: {
          size?: number;
          family?: string;
        };
        callback?: (value: any) => string;
      };
      grid?: {
        display?: boolean;
        color?: string;
        drawBorder?: boolean;
        drawOnChartArea?: boolean;
      };
    };
  };
  elements?: {
    point?: {
      radius?: number;
      hoverRadius?: number;
      backgroundColor?: string;
      borderColor?: string;
      borderWidth?: number;
    };
    line?: {
      tension?: number;
      borderWidth?: number;
    };
  };
  animation?: {
    duration?: number;
    easing?: string;
  };
  hover?: {
    animationDuration?: number;
  };
}

export type ChartType = 'line' | 'bar' | 'doughnut';

interface InteractiveChartProps {
  type: ChartType;
  data: ChartData;
  options?: ChartOptions;
  height?: number;
  className?: string;
  loading?: boolean;
  error?: string;
  onDataPointClick?: (event: any, elements: any[]) => void;
  onDataPointHover?: (event: any, elements: any[]) => void;
  exportable?: boolean;
  title?: string;
}

const InteractiveChart = ({
  type,
  data,
  options = {},
  height = 300,
  className = '',
  loading = false,
  error,
  onDataPointClick,
  onDataPointHover,
  exportable = false,
  title
}: InteractiveChartProps) => {
  const chartRef = useRef<ChartJS>(null);
  const [isExporting, setIsExporting] = useState(false);

  const defaultOptions: ChartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index'
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
          color: '#6b7280',
          font: {
            size: 12,
            family: 'Inter, system-ui, sans-serif'
          }
        }
      },
      title: title ? {
        display: true,
        text: title,
        color: '#111827',
        font: {
          size: 16,
          family: 'Inter, system-ui, sans-serif'
        }
      } : undefined,
      tooltip: {
        enabled: true,
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#374151',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          title: (context) => {
            return context[0]?.label || '';
          },
          label: (context) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y || context.parsed;
            return `${label}: ${typeof value === 'number' ? value.toFixed(2) : value}`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        ticks: {
          color: '#6b7280',
          maxRotation: 0,
          minRotation: 0,
          font: {
            size: 11,
            family: 'Inter, system-ui, sans-serif'
          }
        },
        grid: {
          display: false,
          drawBorder: false
        }
      },
      y: {
        display: true,
        ticks: {
          color: '#6b7280',
          font: {
            size: 11,
            family: 'Inter, system-ui, sans-serif'
          }
        },
        grid: {
          display: true,
          color: 'rgba(107, 114, 128, 0.1)',
          drawBorder: false
        }
      }
    },
    elements: {
      point: {
        radius: 3,
        hoverRadius: 6,
        backgroundColor: '#ffffff',
        borderColor: '#4a9eff',
        borderWidth: 2
      },
      line: {
        tension: 0.4,
        borderWidth: 2
      }
    },
    animation: {
      duration: 750,
      easing: 'easeInOutQuart'
    },
    hover: {
      animationDuration: 200
    },
    onClick: onDataPointClick,
    onHover: onDataPointHover
  }), [title, onDataPointClick, onDataPointHover]);

  const mergedOptions = useMemo(() => ({
    ...defaultOptions,
    ...options,
    plugins: {
      ...defaultOptions.plugins,
      ...options.plugins
    },
    scales: {
      ...defaultOptions.scales,
      ...options.scales
    }
  }), [defaultOptions, options]);

  const handleExport = async () => {
    if (!chartRef.current) return;
    
    setIsExporting(true);
    try {
      const canvas = chartRef.current.canvas;
      const url = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = `${title || 'chart'}-${new Date().toISOString().split('T')[0]}.png`;
      link.href = url;
      link.click();
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const renderChart = () => {
    const commonProps = {
      ref: chartRef,
      data,
      options: mergedOptions
    };

    switch (type) {
      case 'line':
        return <Line {...commonProps} />;
      case 'bar':
        return <Bar {...commonProps} />;
      case 'doughnut':
        return <Doughnut {...commonProps} />;
      default:
        return <Line {...commonProps} />;
    }
  };

  if (loading) {
    return (
      <div className={`relative ${className}`} style={{ height }}>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="loading-spinner"></div>
        </div>
        <div className="skeleton w-full h-full rounded-lg"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`relative ${className}`} style={{ height }}>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <svg className="w-12 h-12 mx-auto mb-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <p className="text-sm text-red-600">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      {exportable && (
        <div className="absolute top-2 right-2 z-10">
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="btn btn-secondary btn-sm"
          >
            {isExporting ? (
              <div className="loading-spinner w-4 h-4"></div>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            )}
          </button>
        </div>
      )}
      <div style={{ height }}>
        {renderChart()}
      </div>
    </div>
  );
};

export default InteractiveChart;
