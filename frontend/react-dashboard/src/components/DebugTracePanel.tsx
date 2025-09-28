import { useEffect, useRef, useState } from 'react'
import Card from './Card'

export default function DebugTracePanel() {
  const [lines, setLines] = useState<string[]>([])
  const [filter, setFilter] = useState<string>('')
  const esRef = useRef<EventSource | null>(null)
  useEffect(() => {
    try {
      const es = new EventSource('/api/debug/trace/stream')
      esRef.current = es
      es.onmessage = (ev) => {
        const txt = (ev?.data || '').toString()
        if (!txt) return
        setLines((prev) => {
          const next = [...prev, txt]
          return next.length > 50 ? next.slice(-50) : next
        })
      }
      es.onerror = () => {}
      return () => { try { es.close() } catch {} }
    } catch {
      // ignore
    }
  }, [])

  const clear = async () => {
    try {
      await fetch('/api/debug/trace', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clear: true }) })
      setLines([])
    } catch {}
  }

  return (
    <Card title="Debug Trace" subtitle="Recent server events (live)">
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, gap: 8, alignItems: 'center' }}>
        <span style={{ color: '#9ca3af' }}>Last {lines.length} events</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: '#9ca3af' }}>Filter</span>
          <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="substring (e.g., /api/grpo)" style={{ width: 220, padding: 6, borderRadius: 6, border: '1px solid #233', background: '#0b0f17', color: '#e5e7eb' }} />
        </div>
        <button onClick={clear} style={{ background: '#111827', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}>Clear</button>
      </div>
      <pre style={{ maxHeight: 200, overflow: 'auto', background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af', margin: 0 }}>
        {lines.filter((ln) => !filter || (ln.toLowerCase().includes(filter.toLowerCase()))).map((ln, i) => ln + '\n')}
      </pre>
    </Card>
  )
}
