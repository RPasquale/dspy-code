import { useEffect, useRef, useState } from 'react'
import Card from './Card'

export default function DevCyclePanel() {
  const [lines, setLines] = useState<string[]>([])
  const [running, setRunning] = useState<boolean>(false)
  const [msg, setMsg] = useState<string>('')
  const esRef = useRef<EventSource | null>(null)

  // Attach SSE
  useEffect(() => {
    try {
      const es = new EventSource('/api/dev-cycle/stream')
      esRef.current = es
      es.onmessage = (ev) => {
        const txt = (ev?.data || '').toString()
        if (!txt) return
        setLines((prev) => {
          const next = [...prev, txt]
          return next.length > 200 ? next.slice(-200) : next
        })
      }
      es.onerror = () => {}
      return () => { try { es.close() } catch {} }
    } catch {}
  }, [])

  const refreshStatus = async () => {
    try {
      const resp = await fetch('/api/dev-cycle/status', { headers: { 'Accept': 'application/json' } })
      const j = await resp.json()
      setRunning(!!j?.running)
    } catch {}
  }
  useEffect(() => { refreshStatus(); const t = setInterval(refreshStatus, 3000); return () => clearInterval(t) }, [])

  const start = async () => {
    try {
      setMsg('')
      const headers: any = { 'Content-Type': 'application/json' }
      const k = localStorage.getItem('ADMIN_KEY') || ''
      if (k) headers['X-Admin-Key'] = k
      const resp = await fetch('/api/dev-cycle/start', { method: 'POST', headers, body: '{}' })
      const j = await resp.json().catch(() => ({}))
      if (!resp.ok || j?.ok === false) {
        setMsg(j?.error || 'Failed to start')
      } else {
        setMsg('Started')
        setRunning(true)
      }
    } catch (e: any) {
      setMsg(e?.message || String(e))
    }
  }

  return (
    <Card title="Dev Cycle" subtitle="Build → Test → Docker → Push">
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <button onClick={start} disabled={running} style={{ background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}>{running ? 'Running…' : 'Start'}</button>
        {msg && <span style={{ color: '#93c5fd' }}>{msg}</span>}
      </div>
      <pre style={{ maxHeight: 220, overflow: 'auto', background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af', margin: 0 }}>
        {lines.map((ln) => ln + '\n')}
      </pre>
    </Card>
  )
}

