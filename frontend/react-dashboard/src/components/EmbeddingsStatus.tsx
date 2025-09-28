import { useEffect, useState } from 'react'
import Card from './Card'

type EmbStatus = {
  exists?: boolean
  status?: string
  count?: number
  ts?: number
  model?: string
  url?: string
  out?: string
  error?: string
}

function fmtTime(ts?: number) {
  try { if (!ts) return '--'; return new Date(ts * 1000).toLocaleString() } catch { return '--' }
}

export default function EmbeddingsStatus() {
  const [st, setSt] = useState<EmbStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    async function tick() {
      try {
        const r = await fetch('/api/embedding/index/status')
        const j = await r.json()
        if (!alive) return
        setSt(j)
      } catch {
        if (!alive) return
        setSt({ exists: false, status: 'idle' })
      } finally {
        if (alive) setLoading(false)
      }
    }
    tick()
    const id = setInterval(tick, 5000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const status = st?.status || 'idle'
  const color = status === 'done' ? '#34d399' : (status === 'running' ? '#f59e0b' : (status === 'error' ? '#f87171' : '#9ca3af'))

  return (
    <Card title="Embeddings Index" subtitle="InferMesh‐backed code memory" dense>
      {loading ? (
        <div className="skeleton skeleton-block" />
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div>
            <div style={{ color: '#9ca3af', marginBottom: 4 }}>Status</div>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
              <span style={{ fontWeight: 600 }}>{status}</span>
            </div>
          </div>
          <div>
            <div style={{ color: '#9ca3af', marginBottom: 4 }}>Chunks</div>
            <div>{typeof st?.count === 'number' ? st.count : '--'}</div>
          </div>
          <div>
            <div style={{ color: '#9ca3af', marginBottom: 4 }}>Last Rebuilt</div>
            <div>{fmtTime(st?.ts)}</div>
          </div>
          <div>
            <div style={{ color: '#9ca3af', marginBottom: 4 }}>Model</div>
            <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{st?.model || '--'}</div>
          </div>
          {st?.error && (
            <div style={{ gridColumn: '1 / span 2', color: '#f87171', fontSize: 12 }}>Error: {st.error}</div>
          )}
        </div>
      )}
    </Card>
  )
}

