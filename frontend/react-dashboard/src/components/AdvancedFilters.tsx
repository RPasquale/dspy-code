import { useState, useEffect, useMemo } from 'react';

export interface FilterOption {
  key: string;
  label: string;
  type: 'text' | 'select' | 'multiselect' | 'date' | 'daterange' | 'number' | 'boolean';
  options?: { value: any; label: string }[];
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
}

export interface FilterValue {
  key: string;
  value: any;
  operator?: 'equals' | 'contains' | 'startsWith' | 'endsWith' | 'gt' | 'gte' | 'lt' | 'lte' | 'between' | 'in' | 'notIn';
}

export interface AdvancedFiltersProps {
  filters: FilterOption[];
  values: FilterValue[];
  onChange: (values: FilterValue[]) => void;
  onClear: () => void;
  className?: string;
  collapsible?: boolean;
  showCount?: boolean;
}

const AdvancedFilters = ({
  filters,
  values,
  onChange,
  onClear,
  className = '',
  collapsible = true,
  showCount = true
}: AdvancedFiltersProps) => {
  const [isExpanded, setIsExpanded] = useState(!collapsible);
  const [localValues, setLocalValues] = useState<FilterValue[]>(values);

  const activeFiltersCount = useMemo(() => {
    return values.filter(v => {
      if (v.value === null || v.value === undefined || v.value === '') return false;
      if (Array.isArray(v.value) && v.value.length === 0) return false;
      if (typeof v.value === 'object' && v.value.start && v.value.end) return true;
      return true;
    }).length;
  }, [values]);

  useEffect(() => {
    setLocalValues(values);
  }, [values]);

  const handleFilterChange = (key: string, value: any, operator?: string) => {
    const newValues = localValues.filter(v => v.key !== key);
    
    if (value !== null && value !== undefined && value !== '' && 
        !(Array.isArray(value) && value.length === 0)) {
      newValues.push({ key, value, operator: operator as any });
    }
    
    setLocalValues(newValues);
    onChange(newValues);
  };

  const handleClearFilter = (key: string) => {
    const newValues = localValues.filter(v => v.key !== key);
    setLocalValues(newValues);
    onChange(newValues);
  };

  const handleClearAll = () => {
    setLocalValues([]);
    onChange([]);
    onClear();
  };

  const renderFilterInput = (filter: FilterOption) => {
    const currentValue = localValues.find(v => v.key === filter.key)?.value;
    const currentOperator = localValues.find(v => v.key === filter.key)?.operator;

    switch (filter.type) {
      case 'text':
        return (
          <input
            type="text"
            value={currentValue || ''}
            onChange={(e) => handleFilterChange(filter.key, e.target.value)}
            placeholder={filter.placeholder || `Filter by ${filter.label}`}
            className="form-input"
          />
        );

      case 'select':
        return (
          <select
            value={currentValue || ''}
            onChange={(e) => handleFilterChange(filter.key, e.target.value)}
            className="form-select"
          >
            <option value="">All {filter.label}</option>
            {filter.options?.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        );

      case 'multiselect':
        const selectedValues = Array.isArray(currentValue) ? currentValue : [];
        return (
          <div className="space-y-2">
            <select
              multiple
              value={selectedValues}
              onChange={(e) => {
                const selected = Array.from(e.target.selectedOptions, option => option.value);
                handleFilterChange(filter.key, selected);
              }}
              className="form-select"
              size={Math.min(filter.options?.length || 1, 5)}
            >
              {filter.options?.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            {selectedValues.length > 0 && (
              <div className="text-xs text-gray-500">
                {selectedValues.length} selected
              </div>
            )}
          </div>
        );

      case 'date':
        return (
          <input
            type="date"
            value={currentValue || ''}
            onChange={(e) => handleFilterChange(filter.key, e.target.value)}
            className="form-input"
          />
        );

      case 'daterange':
        const rangeValue = currentValue || { start: '', end: '' };
        return (
          <div className="flex items-center gap-2">
            <input
              type="date"
              value={rangeValue.start || ''}
              onChange={(e) => handleFilterChange(filter.key, { 
                ...rangeValue, 
                start: e.target.value 
              })}
              className="form-input"
              placeholder="Start date"
            />
            <span className="text-gray-400">to</span>
            <input
              type="date"
              value={rangeValue.end || ''}
              onChange={(e) => handleFilterChange(filter.key, { 
                ...rangeValue, 
                end: e.target.value 
              })}
              className="form-input"
              placeholder="End date"
            />
          </div>
        );

      case 'number':
        return (
          <div className="flex items-center gap-2">
            <select
              value={currentOperator || 'equals'}
              onChange={(e) => {
                const newValues = localValues.filter(v => v.key !== filter.key);
                newValues.push({ 
                  key: filter.key, 
                  value: currentValue, 
                  operator: e.target.value as any 
                });
                setLocalValues(newValues);
                onChange(newValues);
              }}
              className="form-select w-24"
            >
              <option value="equals">=</option>
              <option value="gt">&gt;</option>
              <option value="gte">&gt;=</option>
              <option value="lt">&lt;</option>
              <option value="lte">&lt;=</option>
              <option value="between">between</option>
            </select>
            <input
              type="number"
              value={currentValue || ''}
              onChange={(e) => handleFilterChange(filter.key, parseFloat(e.target.value) || null)}
              placeholder={filter.placeholder || `Filter by ${filter.label}`}
              min={filter.min}
              max={filter.max}
              step={filter.step}
              className="form-input"
            />
          </div>
        );

      case 'boolean':
        return (
          <select
            value={currentValue === null ? '' : currentValue}
            onChange={(e) => handleFilterChange(filter.key, e.target.value === '' ? null : e.target.value === 'true')}
            className="form-select"
          >
            <option value="">All</option>
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
        );

      default:
        return null;
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg ${className}`}>
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Filters
            </h3>
            {showCount && activeFiltersCount > 0 && (
              <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-1 rounded-full">
                {activeFiltersCount}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {activeFiltersCount > 0 && (
              <button
                onClick={handleClearAll}
                className="btn btn-secondary btn-sm"
              >
                Clear All
              </button>
            )}
            {collapsible && (
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="btn btn-ghost btn-sm"
              >
                <svg
                  className={`w-4 h-4 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="p-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filters.map(filter => {
              const hasValue = localValues.some(v => v.key === filter.key && v.value !== null && v.value !== undefined && v.value !== '');
              
              return (
                <div key={filter.key} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {filter.label}
                    </label>
                    {hasValue && (
                      <button
                        onClick={() => handleClearFilter(filter.key)}
                        className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                  {renderFilterInput(filter)}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedFilters;
