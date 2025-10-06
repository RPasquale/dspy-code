import { useMemo, useState } from 'react'

interface EventsTableProps {
  rows: any[];
  onCopy?: (subset: any[]) => void;
  loading?: boolean;
  error?: string;
}

export default function EventsTable({ rows, onCopy, loading, error }: EventsTableProps) {
  const [sel, setSel] = useState<Set<number>>(new Set())
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set())
  const [sortField, setSortField] = useState<string>('timestamp')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')
  
  const toggle = (i: number) => setSel(prev => { 
    const n = new Set(prev); 
    if (n.has(i)) n.delete(i); 
    else n.add(i); 
    return n 
  })
  
  const toggleExpanded = (i: number) => setExpandedRows(prev => {
    const n = new Set(prev);
    if (n.has(i)) n.delete(i);
    else n.add(i);
    return n;
  })
  
  const all = rows || []
  
  const subset = useMemo(() => all.filter((_, i) => sel.has(i)), [all, sel])
  
  const sortedRows = useMemo(() => {
    return [...all].sort((a, b) => {
      const aVal = getNestedValue(a, sortField)
      const bVal = getNestedValue(b, sortField)
      
      if (aVal === bVal) return 0
      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1
      return sortDirection === 'asc' ? 1 : -1
    })
  }, [all, sortField, sortDirection])
  
  const copyJSON = async () => {
    const text = JSON.stringify(subset.length ? subset : all, null, 2)
    try { 
      await navigator.clipboard.writeText(text)
      // Could add toast notification here
    } catch {}
    onCopy?.(subset.length ? subset : all)
  }
  
  const copyCSV = async () => {
    const cols = ['topic', 'ts', 'service', 'event.name', 'event.action', 'event.status']
    const get = (obj: any, path: string) => path.split('.').reduce((acc, k) => (acc && typeof acc === 'object') ? acc[k] : undefined, obj)
    const lines = [cols.join(',')].concat((subset.length ? subset : all).map(r => cols.map(c => JSON.stringify(get(r, c) ?? '')).join(',')))
    try { 
      await navigator.clipboard.writeText(lines.join('\n'))
      // Could add toast notification here
    } catch {}
    onCopy?.(subset.length ? subset : all)
  }
  
  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }
  
  if (loading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="skeleton h-8 w-32 rounded"></div>
          <div className="skeleton h-8 w-24 rounded"></div>
        </div>
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="skeleton h-12 w-full rounded"></div>
          ))}
        </div>
      </div>
    )
  }
  
  if (error) {
    return (
      <div className="flex items-center justify-center p-8 text-red-500">
        <div className="text-center">
          <svg className="w-12 h-12 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <p className="text-sm">{error}</p>
        </div>
      </div>
    )
  }
  
  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        <button 
          className="btn btn-secondary btn-sm" 
          onClick={() => setSel(new Set())}
        >
          Clear Selection
        </button>
        <button 
          className="btn btn-secondary btn-sm" 
          onClick={() => setSel(new Set(all.map((_, i) => i)))}
        >
          Select All
        </button>
        <div className="flex-1"></div>
        <button 
          className="btn btn-primary btn-sm" 
          onClick={copyJSON}
          disabled={all.length === 0}
        >
          Copy JSON
        </button>
        <button 
          className="btn btn-primary btn-sm" 
          onClick={copyCSV}
          disabled={all.length === 0}
        >
          Copy CSV
        </button>
      </div>
      
      {/* Table */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        <div className="max-h-96 overflow-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
              <tr>
                <th className="px-4 py-3 text-left">
                  <input 
                    type="checkbox" 
                    checked={sel.size === all.length && all.length > 0}
                    onChange={() => sel.size === all.length ? setSel(new Set()) : setSel(new Set(all.map((_, i) => i)))}
                    className="rounded border-gray-300"
                  />
                </th>
                <th 
                  className="px-4 py-3 text-left cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                  onClick={() => handleSort('topic')}
                >
                  <div className="flex items-center gap-2">
                    Topic
                    {sortField === 'topic' && (
                      <span className="text-xs">
                        {sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th 
                  className="px-4 py-3 text-left cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                  onClick={() => handleSort('timestamp')}
                >
                  <div className="flex items-center gap-2">
                    Time
                    {sortField === 'timestamp' && (
                      <span className="text-xs">
                        {sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th className="px-4 py-3 text-left">Summary</th>
                <th className="px-4 py-3 text-left w-20">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {sortedRows.map((rec, i) => {
                const e: any = rec?.event || rec
                const ts = typeof rec?.ts === 'number' ? new Date(rec.ts * 1000).toLocaleString() : 
                          (typeof rec?.timestamp === 'number' ? new Date(rec.timestamp * 1000).toLocaleString() : '')
                const topic = rec?.topic || e?.topic || ''
                const summary = e?.action || e?.name || e?.event || e?.status || e?.message || JSON.stringify(e).slice(0, 120)
                const isExpanded = expandedRows.has(i)
                
                return (
                  <tr key={`ev-${i}`} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-4 py-3">
                      <input 
                        type="checkbox" 
                        checked={sel.has(i)} 
                        onChange={() => toggle(i)}
                        className="rounded border-gray-300"
                      />
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-gray-600 dark:text-gray-400 font-mono text-xs">
                        {topic}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-gray-500 dark:text-gray-400 text-xs">
                        {ts}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="max-w-xs truncate">
                        {summary}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => toggleExpanded(i)}
                        className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 text-xs"
                      >
                        {isExpanded ? 'Hide' : 'Show'}
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        
        {/* Expanded details */}
        {sortedRows.map((rec, i) => {
          if (!expandedRows.has(i)) return null
          
          return (
            <div key={`details-${i}`} className="border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white">Event Details</h4>
                <button
                  onClick={() => toggleExpanded(i)}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <pre className="text-xs text-gray-600 dark:text-gray-300 whitespace-pre-wrap overflow-x-auto">
                {JSON.stringify(rec, null, 2)}
              </pre>
            </div>
          )
        })}
      </div>
      
      {all.length === 0 && (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          No events found
        </div>
      )}
    </div>
  )
}

function getNestedValue(obj: any, path: string): any {
  return path.split('.').reduce((acc, k) => (acc && typeof acc === 'object') ? acc[k] : undefined, obj)
}

