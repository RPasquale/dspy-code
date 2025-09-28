import { useEffect, useMemo, useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import Card from './Card'
import { useToast } from './ToastProvider'

async function api(path: string) {
  const r = await fetch(path)
  if (!r.ok) throw new Error(await r.text())
  return await r.json()
}

export default function MeshStatusCard() {
  const { notify } = useToast()
  const st = useQuery({ queryKey: ['mesh-status'], queryFn: () => api('/api/mesh/status'), refetchInterval: 8000 })
  const tp = useQuery({ queryKey: ['mesh-topics'], queryFn: () => api('/api/mesh/topics'), refetchInterval: 15000 })
  const status = (st.data?.status || {}) as any
  const topics = (tp.data?.topics || {}) as any
  const ok = !!status.ok
  const [tailTopic, setTailTopic] = useState<string>('')
  const [tailLimit, setTailLimit] = useState<number>(50)
  const tail = useQuery({
    queryKey: ['mesh-tail', tailTopic, tailLimit],
    queryFn: () => api(`/api/mesh/tail?topic=${encodeURIComponent(tailTopic)}&limit=${tailLimit}`),
    enabled: !!tailTopic,
    refetchInterval: 10000
  })
  const [live, setLive] = useState(false)
  const [showCompact, setShowCompact] = useState(true)
  const [liveTrim, setLiveTrim] = useState<number>(300)
  const [liveItems, setLiveItems] = useState<any[]>([])
  const [liveItemsRaw, setLiveItemsRaw] = useState<any[]>([])
  const [sigFilter, setSigFilter] = useState<string>('')
  const [typeFilter, setTypeFilter] = useState<string>('')
  const [minR, setMinR] = useState<string>('')
  const [maxR, setMaxR] = useState<string>('')
  const [outPath, setOutPath] = useState<string>('')
  const [minKSeed, setMinKSeed] = useState<number>(2)
  useEffect(() => {
    if (!live || !tailTopic) return
    let es: EventSource | null = null
    try {
      es = new EventSource(`/api/mesh/tail/stream?topic=${encodeURIComponent(tailTopic)}&limit=${tailLimit}`)
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          const deltaFmt = Array.isArray(msg?.delta_fmt) ? msg.delta_fmt : []
          const deltaRaw = Array.isArray(msg?.delta) ? msg.delta : []
          if (deltaFmt.length) setLiveItems((prev) => [...prev, ...deltaFmt].slice(-Math.max(1, liveTrim)))
          if (deltaRaw.length) setLiveItemsRaw((prev) => [...prev, ...deltaRaw].slice(-Math.max(1, liveTrim)))
        } catch {}
      }
      es.onerror = () => {}
    } catch {}
    return () => { try { es?.close() } catch {} }
  }, [live, tailTopic, tailLimit])
  const topicList = useMemo(() => {
    try {
      const keys = Object.keys(topics || {})
      return keys
    } catch { return [] }
  }, [topics])
  const applyFilters = useCallback((arr: any[]) => {
    try {
      const minRv = (minR || '').trim() ? parseFloat(minR) : undefined
      const maxRv = (maxR || '').trim() ? parseFloat(maxR) : undefined
      const sig = (sigFilter || '').toLowerCase()
      const typ = (typeFilter || '').toLowerCase()
      return (arr || []).filter((it: any) => {
        if (sig && !String(it.signature || '').toLowerCase().includes(sig)) return false
        if (typ && !String(it.action_type || '').toLowerCase().includes(typ)) return false
        const r = (typeof it.reward === 'number') ? it.reward : undefined
        if (minRv != null && (r == null || r < minRv)) return false
        if (maxRv != null && (r == null || r > maxRv)) return false
        return true
      })
    } catch { return arr }
  }, [sigFilter, typeFilter, minR, maxR])
  return (
    <Card title="Mesh Core" subtitle="Service status + topics">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>Status</div>
          {st.isLoading ? (
            <div className="skeleton skeleton-block" />
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <KV label="OK" value={ok ? 'yes' : 'no'} color={ok ? '#34d399' : '#f87171'} />
              <KV label="RTT (ms)" value={String(status.rtt_ms ?? '--')} />
              <KV label="Endpoint" value={String(status.endpoint ?? '-') } />
            </div>
          )}
        </div>
        <div>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>Topics</div>
          {tp.isLoading ? (
            <div className="skeleton skeleton-block" />
          ) : (
            <pre style={{ maxHeight: 160, overflow: 'auto', background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af' }}>{JSON.stringify(topics, null, 2)}</pre>
          )}
        </div>
      </div>
      <div style={{ marginTop: 10 }}>
        <div style={{ color: '#9ca3af', marginBottom: 6 }}>Tail Topic</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <input className="input" placeholder="topic name" value={tailTopic} onChange={(e) => setTailTopic(e.target.value)} style={{ maxWidth: 260 }} list="mesh-topic-list" />
          <datalist id="mesh-topic-list">
            {topicList.map((t: string) => <option key={t} value={t} />)}
          </datalist>
          <input className="input" type="number" min={1} step={1} value={tailLimit} onChange={(e) => setTailLimit(parseInt(e.target.value || '50', 10))} style={{ width: 100 }} />
        </div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <button className="btn" onClick={() => { setTailTopic('rl_actions') }}>rl_actions</button>
          <button className="btn" onClick={() => { setTailTopic('retrieval_events') }}>retrieval_events</button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
          <label style={{ color: '#9ca3af' }}><input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} /> Live</label>
          <label style={{ color: '#9ca3af' }}><input type="checkbox" checked={showCompact} onChange={(e) => setShowCompact(e.target.checked)} /> Compact</label>
          <span style={{ color: '#9ca3af' }}>Trim</span>
          <input className="input" type="number" min={50} step={50} value={liveTrim} onChange={(e) => setLiveTrim(parseInt(e.target.value || '300', 10))} style={{ width: 100 }} />
          {live && <button className="btn" onClick={() => { setLiveItems([]); setLiveItemsRaw([]) }}>Clear</button>}
          {!!tailTopic && !live && <span style={{ color: '#9ca3af' }}>Live off (show snapshot)</span>}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
          <span style={{ color: '#9ca3af' }}>Filters:</span>
          <input className="input" placeholder="signature contains" value={sigFilter} onChange={(e) => setSigFilter(e.target.value)} style={{ width: 200 }} />
          <input className="input" placeholder="type contains" value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} style={{ width: 160 }} />
          <input className="input" placeholder="min reward" type="number" value={minR} onChange={(e) => setMinR(e.target.value)} style={{ width: 120 }} />
          <input className="input" placeholder="max reward" type="number" value={maxR} onChange={(e) => setMaxR(e.target.value)} style={{ width: 120 }} />
          <button className="btn" onClick={() => { setSigFilter(''); setTypeFilter(''); setMinR(''); setMaxR('') }}>Reset</button>
        </div>
        {(!tailTopic) ? (
          <div className="skeleton skeleton-block" />
        ) : live ? (
          showCompact ? <CompactList items={applyFilters(liveItems)} /> : (
            <pre style={{ maxHeight: 200, overflow: 'auto', background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af' }}>{JSON.stringify(liveItemsRaw.slice().reverse(), null, 2)}</pre>
          )
        ) : (tail.isInitialLoading ? (
          <div className="skeleton skeleton-block" />
        ) : (
          <>
            {showCompact && Array.isArray((tail.data||{}).items_fmt) ? (
              <CompactList items={applyFilters((tail.data as any).items_fmt)} />
            ) : (
              <pre style={{ maxHeight: 200, overflow: 'auto', background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af' }}>{JSON.stringify(tail.data || {}, null, 2)}</pre>
            )}
          </>
        ))}
        {!!tailTopic && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
            <input className="input" placeholder="out path (optional) e.g. /app/.grpo/seed/tail.jsonl" value={outPath} onChange={(e) => setOutPath(e.target.value)} style={{ flex: 1, minWidth: 280 }} />
            <span style={{ color: '#9ca3af' }}>min_k</span>
            <input className="input" type="number" min={1} step={1} value={minKSeed} onChange={(e) => setMinKSeed(parseInt(e.target.value || '2', 10))} style={{ width: 100 }} />
            <button className="btn" onClick={async () => {
              try {
                const items = live ? (showCompact ? liveItems : liveItemsRaw) : ((showCompact && Array.isArray((tail.data||{}).items_fmt)) ? (tail.data as any).items_fmt : ((tail.data as any)?.items || []))
                const payload: any = { items, ...(outPath ? { out: outPath } : {}), min_k: minKSeed }
                const r = await fetch('/api/mesh/tail/to-grpo', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
                const j = await r.json()
                if (j.ok) {
                  notify(`Seeded GRPO dataset → ${j.path}`, 'ok')
                  try { localStorage.setItem('grpo_dataset_path', j.path) } catch {}
                  // open stats in new tab for quick inspection
                  try {
                    const url = `/api/grpo/dataset-stats?path=${encodeURIComponent(j.path)}`
                    window.open(url, '_blank')
                  } catch {}
                } else {
                  notify(`Failed: ${j.error || 'seed error'}`, 'err')
                }
              } catch { notify('Seed error', 'err') }
            }}>Send to Miner</button>
          </div>
        )}
      </div>
    </Card>
  )
}

function CompactList({ items }: { items: any[] }) {
  if (!items || !items.length) return <div style={{ color: '#9ca3af' }}>No items.</div>
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 6, maxHeight: 220, overflow: 'auto' }}>
      {items.slice().reverse().map((it, i) => (
        <div key={i} style={{ display: 'grid', gridTemplateColumns: '1fr 80px', gap: 8, background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8 }}>
          <div>
            <div style={{ color: '#9ca3af', fontSize: 12, marginBottom: 4 }}>{fmtTs(it.timestamp)} {it.signature ? `• ${it.signature}` : ''} {it.action_type ? `• ${it.action_type}` : ''}</div>
            <div style={{ color: '#e5e7eb' }} title={it.prompt}>{truncate(it.prompt, 200)}</div>
            {!!it.text && <div style={{ color: '#9ca3af', fontSize: 12 }} title={it.text}>{truncate(it.text, 240)}</div>}
          </div>
          <div style={{ textAlign: 'right', color: colorReward(it.reward) }}>{fmtReward(it.reward)}</div>
        </div>
      ))}
    </div>
  )
}

// filtering handled inline via applyFilters in component

function fmtTs(ts?: number) {
  try { if (!ts) return ''; return new Date(ts * 1000).toLocaleTimeString() } catch { return '' }
}
function truncate(s?: string, n: number = 120) { if (!s) return ''; return s.length > n ? (s.slice(0, n) + '…') : s }
function fmtReward(r?: number) { if (typeof r !== 'number' || !isFinite(r)) return '--'; return r.toFixed(3) }
function colorReward(r?: number) { if (typeof r !== 'number') return '#9ca3af'; return r >= 0 ? '#34d399' : '#f87171' }

function KV({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8 }}>
      <div style={{ color: '#93c5fd', fontSize: 12 }}>{label}</div>
      <div style={{ color: color || '#e5e7eb', fontSize: 16, marginTop: 2 }}>{value}</div>
    </div>
  )
}
