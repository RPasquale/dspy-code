import { useEffect, useState } from 'react'
import CapacityPanel from '@/components/CapacityPanel'
import SystemResourcesCard from '@/components/SystemResourcesCard'

const CapacityPage = () => {
  const [adminKey, setAdminKey] = useState<string>('')
  const [cfg, setCfg] = useState<any>(null)
  const [snap, setSnap] = useState<any>(null)
  const [props, setProps] = useState<any[]>([])
  const [err, setErr] = useState<string>('')
  const [bundle, setBundle] = useState<string>('deploy/provision/aws/dspy-agent')
  const [blocked, setBlocked] = useState<boolean>(false)
  useEffect(() => {
    let t = 0 as any
    const fn = async () => {
      try { const r = await fetch('/api/system/resources'); const j = await r.json(); setBlocked(j?.host?.ok === false) } catch {}
      t = setTimeout(fn, 5000)
    }
    fn(); return () => clearTimeout(t)
  }, [])
  async function applyApproved() {
    try {
      const res = await fetch('/api/capacity/apply-approved', { method: 'POST', headers: { ...headers(), 'Content-Type': 'application/json' }, body: JSON.stringify({ bundle, yes: true }) })
      const j = await res.json(); alert(JSON.stringify(j));
    } catch(e: any) {
      alert('Error: '+e)
    }
  }

  function headers(): HeadersInit {
    const k = localStorage.getItem('ADMIN_KEY') || ''
    return k ? { 'X-Admin-Key': k } : {}
  }
  function save() {
    localStorage.setItem('ADMIN_KEY', adminKey)
    load()
  }
  async function load() {
    try {
      const r = await fetch('/api/capacity/status', { headers: headers() })
      const j = await r.json()
      setCfg(j.config); setSnap(j.snapshot); setProps(j.proposals || []); setErr('')
    } catch (e: any) {
      setErr(String(e))
    }
  }
  async function act(kind: 'approve'|'deny', p: any) {
    const url = kind === 'approve' ? '/api/capacity/approve' : '/api/capacity/deny'
    const res = await fetch(url, { method: 'POST', headers: { ...headers(), 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: p.kind, params: p }) })
    const j = await res.json(); alert(JSON.stringify(j)); load()
  }
  async function applyAws() {
    try {
      const res = await fetch('/api/capacity/apply', { method: 'POST', headers: { ...headers(), 'Content-Type': 'application/json' }, body: JSON.stringify({ bundle, yes: true }) })
      const j = await res.json(); alert(JSON.stringify(j));
    } catch(e: any) {
      alert('Error: '+e)
    }
  }
  useEffect(() => { setAdminKey(localStorage.getItem('ADMIN_KEY') || ''); load() }, [])
  return (
    <div style={{ padding: '1rem' }}>
      <h2>Capacity & Budgets</h2>
      {blocked && (
        <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fecaca', padding: 8, borderRadius: 6, marginBottom: 12 }}>
          Blocked: free disk below threshold. <a href="/dashboard" style={{ color: '#93c5fd' }}>Resolve in Cleanup →</a>
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <CapacityPanel />
        <SystemResourcesCard />
      </div>
      <div style={{ marginTop: 16 }}>
        <h3>Proposals</h3>
        {err && <div style={{ color: 'tomato' }}>{err}</div>}
        {!props.length && <div style={{ color: '#9ca3af' }}>No proposals.</div>}
        {props.map((p, i) => (
          <div key={i} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <pre style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af' }}>{JSON.stringify(p, null, 2)}</pre>
            <button onClick={() => act('approve', p)}>Approve</button>
            <button onClick={() => act('deny', p)}>Deny</button>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 16 }}>
        <h3>Apply Infrastructure (AWS)</h3>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <input value={bundle} onChange={e => setBundle(e.target.value)} style={{ minWidth: '28rem' }} />
          <button onClick={applyAws}>Apply</button>
          <button onClick={applyApproved}>Apply Approved Set</button>
        </div>
      </div>
    </div>
  )
}

export default CapacityPage
