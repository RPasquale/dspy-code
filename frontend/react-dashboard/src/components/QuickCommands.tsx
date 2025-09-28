import { useState } from 'react'
import Card from './Card'
import { useToast } from './ToastProvider'

export default function QuickCommands() {
  const { notify } = useToast()
  const [pending, setPending] = useState(false)
  const [n, setN] = useState(5)
  const [embPending, setEmbPending] = useState(false)

  async function runSmoke() {
    setPending(true)
    try {
      const r = await fetch('/api/stack/smoke', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ n_messages: n, topic: 'agent.results.app' }) })
      const j = await r.json()
      if (j.ok) notify(`Smoke produced ${j.produced} msgs`, 'ok')
      else notify(`Smoke failed: ${j.error || 'error'}`, 'err')
    } catch (e) {
      notify(`Smoke error: ${String(e)}`, 'err')
    } finally {
      setPending(false)
    }
  }

  async function setGuard() {
    try {
      const r = await fetch('/api/system/guard', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ min_free_gb: 2, min_ram_gb: 1, min_vram_mb: 0 }) })
      const j = await r.json(); notify(j.ok ? 'Guard set' : 'Guard error', j.ok ? 'ok' : 'err')
    } catch { notify('Guard error', 'err') }
  }

  async function setWorkspace() {
    try {
      const r = await fetch('/api/system/workspace', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: '/workspace' }) })
      const j = await r.json(); notify(j.ok ? 'Workspace saved' : 'Workspace error', j.ok ? 'ok' : 'err')
    } catch { notify('Workspace error', 'err') }
  }

  async function rebuildEmbeddings() {
    setEmbPending(true)
    try {
      const r = await fetch('/api/embedding/index/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ infermesh: true }) })
      const j = await r.json(); if (j.ok) notify('Embedding index started', 'ok'); else { notify(`Emb index failed: ${j.error||'error'}`, 'err'); setEmbPending(false); return }
      // Poll status until done or error (timeout ~60s)
      const started = Date.now()
      const poll = async () => {
        try {
          const s = await fetch('/api/embedding/index/status')
          const js = await s.json()
          if (js.status === 'running') {
            if (Date.now() - started > 60000) { notify('Emb index still running…', 'info'); setEmbPending(false); return }
            setTimeout(poll, 2000)
          } else if (js.status === 'done') {
            notify(`Embeddings rebuilt: ${js.count} chunks`, 'ok')
            setEmbPending(false)
          } else if (js.status === 'error') {
            notify(`Embeddings error: ${js.error}`, 'err')
            setEmbPending(false)
          } else {
            setTimeout(poll, 2000)
          }
        } catch {
          setTimeout(poll, 2000)
        }
      }
      setTimeout(poll, 2000)
    } catch { notify('Emb index error', 'err'); setEmbPending(false) }
  }

  return (
    <Card title="Quick Commands" subtitle="One-click helpers">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>Smoke Test</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input className="input" type="number" min={1} step={1} value={n} onChange={(e) => setN(parseInt(e.target.value || '5', 10))} style={{ width: 100 }} />
            <button className="btn btn-primary" onClick={runSmoke} disabled={pending}>{pending ? 'Running…' : 'Run Smoke'}</button>
          </div>
        </div>
        <div>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>Defaults</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={setGuard}>Set Guard</button>
            <button className="btn" onClick={setWorkspace}>Use /workspace</button>
          </div>
        </div>
        <div>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>Embeddings</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-primary" onClick={rebuildEmbeddings} disabled={embPending}>{embPending ? 'Rebuilding…' : 'Rebuild Embeddings'}</button>
          </div>
        </div>
      </div>
    </Card>
  )
}
