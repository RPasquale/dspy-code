import React, { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Card from './Card'
import { api } from '@/api/client'

export default function SystemResourcesCard() {
  const { data, isLoading } = useQuery({ queryKey: ['system-resources'], queryFn: api.getSystemResources, refetchInterval: 4000 })
  const [topicsStr, setTopicsStr] = useState<string>(() => localStorage.getItem('kafka_topics') || 'agent.results,embeddings')
  const [composeFile, setComposeFile] = useState<string>(() => localStorage.getItem('kafka_compose_file') || '')
  const [composeService, setComposeService] = useState<string>(() => localStorage.getItem('kafka_compose_service') || 'kafka')
  const [adminKey, setAdminKey] = useState<string>(() => localStorage.getItem('ADMIN_KEY') || '')
  const [toast, setToast] = useState<{ kind: 'ok'|'err'; text: string } | null>(null)
  const [savingKafka, setSavingKafka] = useState(false)
  const [showAllRaw, setShowAllRaw] = useState(false)
  // Load server defaults once
  useQuery({ queryKey: ['kafka-settings'], queryFn: api.getKafkaSettings, onSuccess: (d) => {
    const s = d?.settings || {}
    if (typeof s.compose_file === 'string' && !composeFile) setComposeFile(s.compose_file)
    if (typeof s.service === 'string' && !composeService) setComposeService(s.service)
  } })
  const topics = useMemo(() => topicsStr.split(',').map(s => s.trim()).filter(Boolean), [topicsStr])
  const kafka = useQuery({ queryKey: ['kafka-configs', topicsStr, composeFile, composeService], queryFn: () => api.getKafkaConfigs(topics, composeFile || undefined, composeService || undefined), refetchInterval: 15000 })
  const host = data?.host
  const disk = host?.disk
  const warn = host && host.ok === false
  const containers = (data?.containers || []).slice(0, 8)
  return (
    <Card title="System Resources" subtitle={isLoading ? 'Loading…' : ''}>
      {toast && (
        <div style={{ position: 'relative' }}>
          <div style={{ position: 'absolute', right: 6, top: -6, background: toast.kind==='ok' ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)', border: '1px solid rgba(148,163,184,0.25)', color: toast.kind==='ok' ? '#86efac' : '#fecaca', borderRadius: 8, padding: '6px 10px' }}>{toast.text}</div>
        </div>
      )}
      {isLoading ? (
        <div className="anim-fade-in" style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 12 }}>
          <div className="skeleton skeleton-block" />
          <div className="skeleton skeleton-block" />
          <div className="skeleton skeleton-block" />
          <div className="skeleton skeleton-block" />
        </div>
      ) : (disk && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 12 }}>
          <KV label="Disk (GB)" value={`${fmt(disk.used_gb)}/${fmt(disk.total_gb)} (${fmt(disk.pct_used)}%)`} />
          <KV label="Free (GB)" value={fmt(disk.free_gb)} />
          <KV label="Threshold (GB)" value={fmt(host?.threshold_free_gb)} />
          <KV label="Status" value={warn ? 'INSUFFICIENT' : 'OK'} color={warn ? '#f87171' : '#34d399'} />
        </div>
      ))}
      {!!(host?.gpu?.length) && !isLoading && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>GPU</div>
          {host.gpu.map((g, i) => (
            <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 4 }}>
              <span style={{ color: '#e5e7eb' }}>{g.name}</span>
              <span style={{ color: '#9ca3af' }}>util {fmt(g.util_pct)}% mem {fmt(g.mem_used_mb)}/{fmt(g.mem_total_mb)} MB</span>
            </div>
          ))}
        </div>
      )}
      <div>
        {!!host?.memory && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 12 }}>
            <KV label="RAM (GB)" value={`${fmt(host.memory.used_gb)}/${fmt(host.memory.total_gb)} (${fmt(host.memory.pct_used)}%)`} />
            <KV label="Free RAM (GB)" value={fmt(host.memory.free_gb)} />
            {!!host.cpu && <KV label="CPU load (1m)" value={String(host.cpu.load1 ?? '--')} />}
            {!!host.cpu && <KV label="CPU load (5m)" value={String(host.cpu.load5 ?? '--')} />}
          </div>
        )}
        <div style={{ color: '#9ca3af', marginBottom: 6 }}>Containers (top 8)</div>
        {isLoading ? (
          <div className="skeleton skeleton-block" />
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', gap: 6 }}>
            <HeadCell text="Name" /><HeadCell text="CPU%" /><HeadCell text="Mem%" /><HeadCell text="Mem (MB)" />
            {containers.map((c) => (
              <React.Fragment key={c.name}>
                <Cell text={c.name} />
                <Cell text={fmt(c.cpu_pct)} />
                <Cell text={fmt(c.mem_pct)} />
                <Cell text={`${fmt(c.mem_used_mb)}/${fmt(c.mem_limit_mb)}`} />
              </React.Fragment>
            ))}
          </div>
        )}
      </div>
      <div style={{ marginTop: 10, color: '#9ca3af' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span>Kafka topics:</span>
          <input value={topicsStr} onChange={(e) => { setTopicsStr(e.target.value); localStorage.setItem('kafka_topics', e.target.value) }} style={{ width: 260, padding: 6, borderRadius: 6, border: '1px solid #233', background: '#0b0f17', color: '#e5e7eb' }} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span>Compose file:</span>
          <input value={composeFile} onChange={(e) => setComposeFile(e.target.value)} placeholder="docker/lightweight/docker-compose.yml" style={{ width: 280, padding: 6, borderRadius: 6, border: '1px solid #233', background: '#0b0f17', color: '#e5e7eb' }} />
          <span>Service:</span>
          <input value={composeService} onChange={(e) => setComposeService(e.target.value)} placeholder="kafka" style={{ width: 140, padding: 6, borderRadius: 6, border: '1px solid #233', background: '#0b0f17', color: '#e5e7eb' }} />
          <button onClick={async () => {
            try { setSavingKafka(true); localStorage.setItem('kafka_compose_file', composeFile); localStorage.setItem('kafka_compose_service', composeService); await api.setKafkaSettings({ compose_file: composeFile, service: composeService }) } finally { setSavingKafka(false); kafka.refetch() }
          }} style={{ background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}>{savingKafka ? 'Saving…' : 'Save'}</button>
          <button onClick={() => kafka.refetch()} style={{ background: '#111827', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}>Refresh</button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span>Kafka admin key:</span>
          <input type="password" value={adminKey} onChange={(e) => setAdminKey(e.target.value)} placeholder="X-Admin-Key" style={{ width: 220, padding: 6, borderRadius: 6, border: '1px solid #233', background: '#0b0f17', color: '#e5e7eb' }} />
          <button onClick={() => { localStorage.setItem('ADMIN_KEY', adminKey); setToast({ kind: 'ok', text: 'Admin key saved' }); setTimeout(() => setToast(null), 1500) }} style={{ background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}>Save</button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>Kafka retention:</span>
          <span>{formatKafka(kafka.data)}</span>
          <label style={{ marginLeft: 'auto', color: '#9ca3af' }}>
            <input type="checkbox" checked={showAllRaw} onChange={(e) => setShowAllRaw(e.target.checked)} /> Show raw for all
          </label>
          <span style={{ color: '#9ca3af' }}>Apply all:</span>
          <input id="apply-all-ret" placeholder="retention.ms" style={{ width: 140, padding: 6, borderRadius: 6, border: '1px solid #233', background: '#0b0f17', color: '#e5e7eb' }} />
          <label style={{ color: '#9ca3af' }}><input id="apply-all-dry" type="checkbox" defaultChecked /> dry-run</label>
          <details>
            <summary style={{ cursor: 'pointer', color: '#9ca3af' }}>Overrides</summary>
            <div style={{ marginTop: 6, color: '#9ca3af' }}>Format: topic=ms, one per line</div>
            <textarea id="apply-all-over" placeholder={"agent.results=60000\nembeddings=300000"} style={{ width: 360, height: 80, padding: 6, borderRadius: 6, border: '1px solid #233', background: '#0b0f17', color: '#e5e7eb' }} />
          </details>
          <button onClick={async () => {
            try {
              const inp = (document.getElementById('apply-all-ret') as HTMLInputElement | null)
              const dryEl = (document.getElementById('apply-all-dry') as HTMLInputElement | null)
              const over = (document.getElementById('apply-all-over') as HTMLTextAreaElement | null)
              const val = inp?.value?.trim() || ''
              const applyDry = !!(dryEl?.checked)
              const cf = localStorage.getItem('kafka_compose_file') || ''
              const cs = localStorage.getItem('kafka_compose_service') || 'kafka'
              const overrides: Record<string, number> = {}
              if (over?.value?.trim()) {
                over.value.split(/\n+/).forEach((ln) => {
                  const m = ln.split('='); if (m.length === 2) { const k = m[0].trim(); const v = parseInt(m[1].trim(), 10); if (k && Number.isFinite(v)) overrides[k] = v }
                })
              }
              if (Object.keys(overrides).length > 0) {
                await Promise.all(Object.entries(overrides).map(([k, v]) => api.runCleanup({ dry_run: applyDry, actions: { kafka_prune: { topics: [k], retention_ms: v, ...(cf ? { compose_file: cf } : {}), ...(cs ? { service: cs } : {}) } } })))
                setToast({ kind: 'ok', text: (applyDry ? 'Previewed' : 'Applied') + ' overrides' }); setTimeout(() => setToast(null), 1500)
              } else {
                if (!/^\d+$/.test(val)) { return }
                const num = parseInt(val, 10)
                await api.runCleanup({ dry_run: applyDry, actions: { kafka_prune: { topics, retention_ms: num, ...(cf ? { compose_file: cf } : {}), ...(cs ? { service: cs } : {}) } } })
                setToast({ kind: 'ok', text: (applyDry ? 'Previewed' : 'Applied') + ' all' }); setTimeout(() => setToast(null), 1500)
              }
            } catch { setToast({ kind: 'err', text: 'Apply all failed' }); setTimeout(() => setToast(null), 1500) }
          }} style={{ background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}>Apply</button>
        </div>
        <div style={{ marginTop: 6 }}>
          {Object.entries(kafka.data?.topics || {}).map(([name, info]: any) => (
            <PerTopicRetention key={name} name={name} retention={info?.retention_ms} raw={(info as any)?.raw} showAllRaw={showAllRaw} onToast={(ok, txt) => { setToast({ kind: ok ? 'ok' : 'err', text: txt }); setTimeout(() => setToast(null), 1500) }} />
          ))}
        </div>
        {renderSparklines(kafka.data)}
      </div>
      {warn && (
        <div style={{ marginTop: 12 }}>
          <div style={{ color: '#f87171' }}>
            Free disk below threshold. Training and heavy operations are blocked until more storage is available.
          </div>
          <div style={{ marginTop: 6, display: 'flex', gap: 8 }}>
            <a href="/dashboard" style={btnStyle}>Open Cleanup</a>
            <a href="/admin/capacity" style={btnStyle}>Capacity Admin</a>
          </div>
        </div>
      )}
    </Card>
  )
}

function fmt(n?: number, d = 1) {
  if (typeof n !== 'number' || !isFinite(n)) return '--'
  if (Math.abs(n) >= 100) return n.toFixed(0)
  return n.toFixed(d)
}

function KV({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8 }}>
      <div style={{ color: '#93c5fd', fontSize: 12 }}>{label}</div>
      <div style={{ color: color || '#e5e7eb', fontSize: 16, marginTop: 2 }}>{value}</div>
    </div>
  )
}

function HeadCell({ text }: { text: string }) {
  return <div style={{ color: '#9ca3af', fontSize: 12 }}>{text}</div>
}

function Cell({ text }: { text: string }) {
  return <div style={{ color: '#e5e7eb', fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis' }}>{text}</div>
}

const btnStyle: any = { background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer', textDecoration: 'none' }

function formatKafka(data?: any): string {
  try {
    const t = data?.topics || {}
    const ents = Object.entries(t) as [string, any][]
    if (!ents.length) return 'n/a'
    return ents.map(([k, v]) => `${k}=${(v?.retention_ms != null) ? v.retention_ms : 'n/a'}`).join(' · ')
  } catch {
    return 'n/a'
  }
}

function PerTopicRetention({ name, retention, raw, showAllRaw, onToast }: { name: string; retention?: number; raw?: string; showAllRaw?: boolean; onToast?: (ok: boolean, text: string) => void }) {
  const [val, setVal] = useState<string>(retention != null ? String(retention) : '')
  const [dry, setDry] = useState<boolean>(true)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [showRaw, setShowRaw] = useState(false)
  return (
    <div style={{ margin: '4px 0' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ color: '#e5e7eb', minWidth: 160 }}>{name}</span>
        <input value={val} onChange={(e) => setVal(e.target.value)} placeholder="retention.ms" style={{ width: 160, padding: 6, borderRadius: 6, border: '1px solid #233', background: '#0b0f17', color: '#e5e7eb' }} />
        <label style={{ color: '#9ca3af' }}><input type="checkbox" checked={dry} onChange={(e) => setDry(e.target.checked)} /> dry-run</label>
        <button onClick={() => setShowRaw(!showRaw)} style={{ background: '#111827', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}>{(showAllRaw || showRaw) ? 'Hide' : 'Show'}</button>
      <button disabled={busy || !/^[0-9]+$/.test(val)} onClick={async () => {
        try {
          setBusy(true)
          const composeFile = localStorage.getItem('kafka_compose_file') || ''
          const composeService = localStorage.getItem('kafka_compose_service') || 'kafka'
          await api.runCleanup({ dry_run: !!dry, actions: { kafka_prune: { topics: [name], retention_ms: parseInt(val, 10), ...(composeFile ? { compose_file: composeFile } : {}), ...(composeService ? { service: composeService } : {}) } } })
          const t = dry ? 'Previewed' : 'Applied'
          setMsg(t)
          onToast && onToast(true, `${t} ${name}`)
          setTimeout(() => setMsg(''), 1500)
        } catch (e: any) {
          setMsg('Failed')
          onToast && onToast(false, `Failed ${name}`)
          setTimeout(() => setMsg(''), 2000)
        } finally { setBusy(false) }
      }} style={{ background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }}>{busy ? 'Applying…' : 'Apply'}</button>
      {msg && <span style={{ color: '#93c5fd' }}>{msg}</span>}
      </div>
      {(showAllRaw || showRaw) && raw && (
        <pre style={{ whiteSpace: 'pre-wrap', background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af', marginTop: 8 }}>{raw}</pre>
      )}
    </div>
  )
}

function renderSparklines(data?: any) {
  try {
    const hist = data?.history || {}
    const keys: string[] = Object.keys(hist)
    if (!keys.length) return null
    return (
      <div style={{ display: 'flex', gap: 20, marginTop: 6 }}>
        {keys.map((k) => {
          const arr = (hist[k] || []) as { ts: number; retention_ms: number }[]
          const points = arr.slice(-30)
          if (!points.length) return null
          const min = Math.min(...points.map(p => p.retention_ms))
          const max = Math.max(...points.map(p => p.retention_ms))
          const w = 120, h = 30
          const norm = (v: number) => (max === min ? h/2 : h - ((v - min) / (max - min)) * h)
          const step = w / Math.max(1, points.length - 1)
          const d = points.map((p, i) => `${i*step},${norm(p.retention_ms)}`).join(' ')
          return (
            <div key={k} style={{ color: '#9ca3af' }}>
              <div style={{ fontSize: 12, marginBottom: 2 }}>{k}</div>
              <svg width={w} height={h} style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 4 }}>
                <polyline fill="none" stroke="#34d399" strokeWidth="2" points={d} />
              </svg>
            </div>
          )
        })}
      </div>
    )
  } catch {
    return null
  }
}
