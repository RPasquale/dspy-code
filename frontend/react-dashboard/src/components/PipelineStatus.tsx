import { useQuery } from '@tanstack/react-query'
import Card from './Card'
import { useToast } from './ToastProvider'

async function api(path: string) {
  const r = await fetch(path)
  if (!r.ok) throw new Error(await r.text())
  return await r.json()
}

export default function PipelineStatus() {
  const { notify } = useToast()
  const q = useQuery({ queryKey: ['pipeline-status'], queryFn: () => api('/api/pipeline/status'), refetchInterval: 5000 })
  const st = (q.data || {}) as any
  const vec = st.vectorizer || {}
  const emb = st.embed_worker || {}
  const ok = (emb.ok === true) && (vec.enabled !== false)

  function copy(txt: string) {
    try { navigator.clipboard.writeText(txt); notify('Copied to clipboard', 'ok') } catch { notify('Copy failed', 'err') }
  }

  return (
    <Card title="Data Pipeline" subtitle="Vectorizer + Embed worker" dense>
      {q.isLoading ? (<div className="skeleton skeleton-block" />) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div>
            <div style={{ color: '#9ca3af' }}>Embed Worker</div>
            <div style={{ color: emb.ok ? '#34d399' : '#f87171' }}>{emb.ok ? 'reachable' : 'unreachable'}</div>
          </div>
          <div>
            <div style={{ color: '#9ca3af' }}>Vectorizer Rows (est)</div>
            <div>{typeof vec.rows_est === 'number' ? vec.rows_est : '--'}</div>
          </div>
          <div style={{ gridColumn: '1 / span 2', display: 'flex', gap: 8, marginTop: 6 }}>
            <button className="btn" onClick={() => copy('docker compose -f docker/lightweight/docker-compose.yml up -d embed-worker spark-vectorizer')}>Copy Start Commands</button>
            <button className="btn" onClick={() => copy('docker compose -f docker/lightweight/docker-compose.yml stop embed-worker spark-vectorizer')}>Copy Stop Commands</button>
          </div>
        </div>
      )}
    </Card>
  )
}

