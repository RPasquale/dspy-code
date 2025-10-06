import { useState, useMemo } from 'react';

export interface Column<T> {
  key: keyof T | string;
  title: string;
  dataIndex: keyof T | string;
  render?: (value: any, record: T, index: number) => React.ReactNode;
  sorter?: (a: T, b: T) => number;
  filterable?: boolean;
  width?: string | number;
  align?: 'left' | 'center' | 'right';
  fixed?: 'left' | 'right';
  ellipsis?: boolean;
}

export interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  loading?: boolean;
  error?: string;
  pagination?: {
    current: number;
    pageSize: number;
    total: number;
    showSizeChanger?: boolean;
    showQuickJumper?: boolean;
    showTotal?: (total: number, range: [number, number]) => string;
  };
  rowKey?: keyof T | ((record: T) => string);
  rowSelection?: {
    selectedRowKeys: string[];
    onChange: (selectedRowKeys: string[], selectedRows: T[]) => void;
    getCheckboxProps?: (record: T) => { disabled?: boolean; name?: string };
  };
  scroll?: {
    x?: number | string;
    y?: number | string;
  };
  size?: 'small' | 'middle' | 'large';
  bordered?: boolean;
  striped?: boolean;
  hoverable?: boolean;
  className?: string;
  onRow?: (record: T, index: number) => {
    onClick?: (event: React.MouseEvent) => void;
    onDoubleClick?: (event: React.MouseEvent) => void;
    onMouseEnter?: (event: React.MouseEvent) => void;
    onMouseLeave?: (event: React.MouseEvent) => void;
  };
  expandable?: {
    expandedRowKeys: string[];
    onExpandedRowsChange: (expandedRowKeys: string[]) => void;
    expandedRowRender?: (record: T, index: number) => React.ReactNode;
    expandRowByClick?: boolean;
    rowExpandable?: (record: T) => boolean;
  };
}

const DataTable = <T extends Record<string, any>>({
  data,
  columns,
  loading = false,
  error,
  pagination,
  rowKey = 'id',
  rowSelection,
  scroll,
  size = 'middle',
  bordered = false,
  striped = true,
  hoverable = true,
  className = '',
  onRow,
  expandable
}: DataTableProps<T>) => {
  const [sortField, setSortField] = useState<string>('');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(pagination?.pageSize || 10);

  const getRowKey = (record: T, index: number): string => {
    if (typeof rowKey === 'function') {
      return rowKey(record);
    }
    return String(record[rowKey as keyof T] || index);
  };

  const sortedData = useMemo(() => {
    if (!sortField) return data;
    
    const column = columns.find(col => col.key === sortField);
    if (!column?.sorter) return data;

    return [...data].sort((a, b) => {
      const result = column.sorter!(a, b);
      return sortDirection === 'asc' ? result : -result;
    });
  }, [data, sortField, sortDirection, columns]);

  const paginatedData = useMemo(() => {
    if (!pagination) return sortedData;
    
    const start = (currentPage - 1) * pageSize;
    const end = start + pageSize;
    return sortedData.slice(start, end);
  }, [sortedData, currentPage, pageSize, pagination]);

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handlePageSizeChange = (size: number) => {
    setPageSize(size);
    setCurrentPage(1);
  };

  const getValue = (record: T, dataIndex: keyof T | string) => {
    if (typeof dataIndex === 'string' && dataIndex.includes('.')) {
      return dataIndex.split('.').reduce((obj, key) => obj?.[key], record);
    }
    return record[dataIndex as keyof T];
  };

  const sizeClasses = {
    small: 'text-xs',
    middle: 'text-sm',
    large: 'text-base'
  };

  if (loading) {
    return (
      <div className={`border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden ${className}`}>
        <div className="p-4">
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="skeleton h-8 w-full rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`border border-red-200 dark:border-red-800 rounded-lg overflow-hidden ${className}`}>
        <div className="p-8 text-center">
          <svg className="w-12 h-12 mx-auto mb-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <p className="text-sm text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <div className={`border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden ${bordered ? 'border-2' : ''}`}>
        <div className="overflow-x-auto">
          <table className={`w-full ${sizeClasses[size]}`}>
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                {rowSelection && (
                  <th className="px-4 py-3 text-left">
                    <input
                      type="checkbox"
                      checked={rowSelection.selectedRowKeys.length === data.length && data.length > 0}
                      onChange={(e) => {
                        if (e.target.checked) {
                          const allKeys = data.map((record, index) => getRowKey(record, index));
                          rowSelection.onChange(allKeys, data);
                        } else {
                          rowSelection.onChange([], []);
                        }
                      }}
                      className="rounded border-gray-300"
                    />
                  </th>
                )}
                {expandable && (
                  <th className="px-4 py-3 text-left w-12"></th>
                )}
                {columns.map((column) => (
                  <th
                    key={String(column.key)}
                    className={`px-4 py-3 text-${column.align || 'left'} ${
                      column.sorter ? 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700' : ''
                    }`}
                    style={{ width: column.width }}
                    onClick={() => column.sorter && handleSort(String(column.key))}
                  >
                    <div className="flex items-center gap-2">
                      <span>{column.title}</span>
                      {column.sorter && (
                        <span className="text-xs text-gray-400">
                          {sortField === column.key ? (sortDirection === 'asc' ? '↑' : '↓') : '↕'}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className={`divide-y divide-gray-200 dark:divide-gray-700 ${striped ? 'even:bg-gray-50 dark:even:bg-gray-800' : ''}`}>
              {paginatedData.map((record, index) => {
                const key = getRowKey(record, index);
                const isExpanded = expandable?.expandedRowKeys.includes(key);
                const rowProps = onRow?.(record, index) || {};

                return (
                  <React.Fragment key={key}>
                    <tr
                      className={`${hoverable ? 'hover:bg-gray-50 dark:hover:bg-gray-800' : ''}`}
                      {...rowProps}
                    >
                      {rowSelection && (
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={rowSelection.selectedRowKeys.includes(key)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                rowSelection.onChange(
                                  [...rowSelection.selectedRowKeys, key],
                                  [...rowSelection.selectedRowKeys.map(k => data.find((r, i) => getRowKey(r, i) === k)!).filter(Boolean), record]
                                );
                              } else {
                                rowSelection.onChange(
                                  rowSelection.selectedRowKeys.filter(k => k !== key),
                                  rowSelection.selectedRowKeys.filter(k => k !== key).map(k => data.find((r, i) => getRowKey(r, i) === k)!).filter(Boolean)
                                );
                              }
                            }}
                            className="rounded border-gray-300"
                          />
                        </td>
                      )}
                      {expandable && (
                        <td className="px-4 py-3">
                          {expandable.rowExpandable?.(record) !== false && (
                            <button
                              onClick={() => {
                                const newExpandedKeys = isExpanded
                                  ? expandable.expandedRowKeys.filter(k => k !== key)
                                  : [...expandable.expandedRowKeys, key];
                                expandable.onExpandedRowsChange(newExpandedKeys);
                              }}
                              className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                            >
                              <svg
                                className={`w-4 h-4 transform transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                              </svg>
                            </button>
                          )}
                        </td>
                      )}
                      {columns.map((column) => {
                        const value = getValue(record, column.dataIndex);
                        const content = column.render ? column.render(value, record, index) : value;

                        return (
                          <td
                            key={String(column.key)}
                            className={`px-4 py-3 text-${column.align || 'left'} ${
                              column.ellipsis ? 'truncate max-w-xs' : ''
                            }`}
                          >
                            {content}
                          </td>
                        );
                      })}
                    </tr>
                    {expandable && isExpanded && expandable.expandedRowRender && (
                      <tr>
                        <td colSpan={columns.length + (rowSelection ? 1 : 0) + (expandable ? 1 : 0)} className="p-0">
                          <div className="bg-gray-50 dark:bg-gray-800 p-4">
                            {expandable.expandedRowRender(record, index)}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        {pagination && (
          <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-700 dark:text-gray-300">
                {pagination.showTotal ? (
                  pagination.showTotal(pagination.total, [
                    (currentPage - 1) * pageSize + 1,
                    Math.min(currentPage * pageSize, pagination.total)
                  ])
                ) : (
                  `Showing ${(currentPage - 1) * pageSize + 1} to ${Math.min(currentPage * pageSize, pagination.total)} of ${pagination.total} entries`
                )}
              </div>
              <div className="flex items-center gap-2">
                {pagination.showSizeChanger && (
                  <select
                    value={pageSize}
                    onChange={(e) => handlePageSizeChange(Number(e.target.value))}
                    className="form-select text-sm"
                  >
                    <option value={10}>10 / page</option>
                    <option value={20}>20 / page</option>
                    <option value={50}>50 / page</option>
                    <option value={100}>100 / page</option>
                  </select>
                )}
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="btn btn-secondary btn-sm"
                  >
                    Previous
                  </button>
                  <span className="px-3 py-1 text-sm">
                    Page {currentPage} of {Math.ceil(pagination.total / pageSize)}
                  </span>
                  <button
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage >= Math.ceil(pagination.total / pageSize)}
                    className="btn btn-secondary btn-sm"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataTable;
