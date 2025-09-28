import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import Card from './Card'
import { api } from '@/api/client'
import { useToast } from './ToastProvider'

export default function GuardSettingsPanel() {
  const { notify } = useToast()
  const q = useQuery({ queryKey: ['system-resources'], queryFn: api.getSystemResources, refetchInterval: 5000 })
  const ws = useQuery({ queryKey: ['workspace-path'], queryFn: api.getWorkspace, refetchInterval: 15000 })
  const [wspath, setWspath] = useState<string>('')
  const saveWs = useMutation({ mutationFn: async () => { const res = await api.setWorkspace(wspath); return res }, onSuccess: () => { setMsg('Workspace saved'); notify('Workspace saved', 'ok'); setTimeout(() => setMsg(''), 1500) }, onError: (e: any) => { const t = String(e?.message || e); setMsg(t); notify(`Workspace save failed: ${t}`, 'err'); setTimeout(() => setMsg(''), 3000) } })
  // Initialize from query
  useState(() => { if (typeof ws.data?.path === 'string') setWspath(ws.data.path); return undefined })
  const [minFree, setMinFree] = useState<number>(() => Number(localStorage.getItem('guard_min_free_gb') || '2'))
  const [minRam, setMinRam] = useState<number>(() => Number(localStorage.getItem('guard_min_ram_gb') || '0'))
  const [minVram, setMinVram] = useState<number>(() => Number(localStorage.getItem('guard_min_vram_mb') || '0'))
  const [msg, setMsg] = useState<string>('')
  const save = useMutation({ mutationFn: async () => {
    localStorage.setItem('guard_min_free_gb', String(minFree))
    localStorage.setItem('guard_min_ram_gb', String(minRam))
    localStorage.setItem('guard_min_vram_mb', String(minVram))
    // POST to server (system/guard)
    const res = await fetch('/api/system/guard', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ min_free_gb: minFree, min_ram_gb: minRam, min_vram_mb: minVram }) })
    return await res.json()
  }, onSuccess: () => { setMsg('Saved'); notify('Guard thresholds saved', 'ok'); setTimeout(() => setMsg(''), 1500) }, onError: (e: any) => { const t = String(e?.message || e); setMsg(t); notify(`Save failed: ${t}`, 'err'); setTimeout(() => setMsg(''), 3000) } })
  const guard = useQuery({ queryKey: ['system-guard-mini', minFree, minRam, minVram], queryFn: () => api.guardSystem({ min_free_gb: minFree, min_ram_gb: minRam, min_vram_mb: minVram }), refetchInterval: 5000 })
  const warn = guard.data ? !guard.data.ok : false
  return (
    <Card title="Guard Settings" subtitle="Thresholds for blocking heavy operations">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div>
          <Row label="Workspace path">
            <input value={wspath} onChange={(e) => setWspath(e.target.value)} className="input" placeholder="/absolute/or/container/path" />
          </Row>
          <button onClick={() => saveWs.mutate()} disabled={saveWs.isPending} className="btn">{saveWs.isPending ? 'Saving…' : 'Save Workspace'}</button>
          <div style={{ height: 8 }} />
          <Row label="Min Free Disk (GB)"><input type="number" min={0} step={0.5} value={minFree} onChange={(e) => setMinFree(parseFloat(e.target.value || '0'))} className="input" /></Row>
          <Row label="Min Free RAM (GB)"><input type="number" min={0} step={0.5} value={minRam} onChange={(e) => setMinRam(parseFloat(e.target.value || '0'))} className="input" /></Row>
          <Row label="Min Free VRAM (MB)"><input type="number" min={0} step={64} value={minVram} onChange={(e) => setMinVram(parseFloat(e.target.value || '0'))} className="input" /></Row>
          <button onClick={() => save.mutate()} disabled={save.isPending} className="btn">{save.isPending ? 'Saving…' : 'Save'}</button>
          {msg && <span style={{ marginLeft: 8, color: '#93c5fd' }}>{msg}</span>}
        </div>
        <div>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>Status</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8 }}>
            <KV label="Disk" value={guard.data?.disk_ok ? 'OK' : 'BLOCK'} color={guard.data?.disk_ok ? '#34d399' : '#f87171'} />
            <KV label="RAM" value={guard.data?.ram_ok ? 'OK' : 'BLOCK'} color={guard.data?.ram_ok ? '#34d399' : '#f87171'} />
            <KV label="GPU" value={guard.data?.gpu_ok ? 'OK' : 'BLOCK'} color={guard.data?.gpu_ok ? '#34d399' : '#f87171'} />
          </div>
        </div>
      </div>
    </Card>
  )
}

function Row({ label, children }: { label: string; children: any }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: 10, alignItems: 'center', marginBottom: 8 }}>
      <div style={{ color: '#9ca3af' }}>{label}</div>
      <div>{children}</div>
    </div>
  )
}

function KV({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8 }}>
      <div style={{ color: '#93c5fd', fontSize: 12 }}>{label}</div>
      <div style={{ color: color || '#e5e7eb', fontSize: 16, marginTop: 2 }}>{value}</div>
    </div>
  )
}

const inputStyle: any = {}
const btnStyle: any = {}
