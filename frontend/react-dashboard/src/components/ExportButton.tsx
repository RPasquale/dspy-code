import { useState } from 'react';
import { 
  exportToCSV, 
  exportToJSON, 
  exportToPDF, 
  exportToExcel,
  exportMultipleFormats,
  ExportColumn,
  ExportOptions 
} from '../utils/exportUtils';
import { useToast } from './ToastProvider';

interface ExportButtonProps {
  data: any[];
  columns: ExportColumn[];
  options?: ExportOptions;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'primary' | 'secondary' | 'ghost';
  showDropdown?: boolean;
  customFormats?: ('csv' | 'json' | 'pdf' | 'excel')[];
}

const ExportButton = ({
  data,
  columns,
  options = {},
  className = '',
  size = 'md',
  variant = 'secondary',
  showDropdown = true,
  customFormats = ['csv', 'json', 'pdf', 'excel']
}: ExportButtonProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const { showToast } = useToast();

  const sizeClasses = {
    sm: 'btn-sm',
    md: '',
    lg: 'btn-lg'
  };

  const variantClasses = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    ghost: 'btn-ghost'
  };

  const handleExport = async (format: string) => {
    if (data.length === 0) {
      showToast({
        type: 'warning',
        title: 'No Data',
        message: 'There is no data to export.'
      });
      return;
    }

    setIsExporting(true);
    setIsOpen(false);

    try {
      switch (format) {
        case 'csv':
          exportToCSV(data, columns, options);
          break;
        case 'json':
          exportToJSON(data, options);
          break;
        case 'pdf':
          exportToPDF(data, columns, options);
          break;
        case 'excel':
          exportToExcel(data, columns, options);
          break;
        case 'all':
          await exportMultipleFormats(data, columns, customFormats, options);
          break;
      }

      showToast({
        type: 'success',
        title: 'Export Complete',
        message: `Data exported successfully as ${format.toUpperCase()}`
      });
    } catch (error) {
      showToast({
        type: 'error',
        title: 'Export Failed',
        message: 'An error occurred while exporting the data.'
      });
    } finally {
      setIsExporting(false);
    }
  };

  const formatOptions = [
    { key: 'csv', label: 'CSV', icon: '📊' },
    { key: 'json', label: 'JSON', icon: '📄' },
    { key: 'pdf', label: 'PDF', icon: '📋' },
    { key: 'excel', label: 'Excel', icon: '📈' }
  ];

  if (!showDropdown) {
    return (
      <button
        onClick={() => handleExport('csv')}
        disabled={isExporting || data.length === 0}
        className={`btn ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      >
        {isExporting ? (
          <div className="loading-spinner w-4 h-4"></div>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Export
          </>
        )}
      </button>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isExporting || data.length === 0}
        className={`btn ${variantClasses[variant]} ${sizeClasses[size]}`}
      >
        {isExporting ? (
          <div className="loading-spinner w-4 h-4"></div>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Export
            <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </>
        )}
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-20">
            <div className="py-1">
              {formatOptions.map(option => (
                <button
                  key={option.key}
                  onClick={() => handleExport(option.key)}
                  className="w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-3"
                >
                  <span className="text-lg">{option.icon}</span>
                  <span>{option.label}</span>
                </button>
              ))}
              <div className="border-t border-gray-200 dark:border-gray-700 my-1"></div>
              <button
                onClick={() => handleExport('all')}
                className="w-full px-4 py-2 text-left text-sm text-blue-600 dark:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-3"
              >
                <span className="text-lg">📦</span>
                <span>Export All Formats</span>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ExportButton;
