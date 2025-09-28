import React, { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import Card from '@/components/Card'
import CleanupPanel from '@/components/CleanupPanel'

type Sample = { ts: number; cpu?: number; ramPct?: number; diskPct?: number; gpuPct?: number }

export default function SystemStatsPage() {
  const { data } = useQuery({ queryKey: ['system-resources-page'], queryFn: api.getSystemResources, refetchInterval: 5000 })
  const [history, setHistory] = useState<Sample[]>([])
  const [adminKey, setAdminKey] = useState<string>('')
  useEffect(() => { try { setAdminKey(localStorage.getItem('ADMIN_KEY') || '') } catch {} }, [])

  useEffect(() => {
    if (!data) return
    const now = Date.now() / 1000
    const cpu = data?.host?.cpu?.cpu_pct ?? data?.host?.cpu?.load1 ?? undefined
    const ram = data?.host?.memory?.pct_used ?? data?.host?.mem?.mem_pct ?? undefined
    const disk = data?.host?.disk?.pct_used ?? undefined
    const g = Array.isArray(data?.host?.gpu) ? data.host.gpu[0] : undefined
    const gpuPct = (g?.util_pct != null) ? g.util_pct : (g?.mem_total_mb ? ((g.mem_used_mb || 0) / g.mem_total_mb) * 100.0 : undefined)
    setHistory((prev) => {
      const next = [...prev, { ts: now, cpu: toNum(cpu), ramPct: toNum(ram), diskPct: toNum(disk), gpuPct: toNum(gpuPct) }].slice(-120)
      return next
    })
  }, [data])

  const last = history[history.length - 1] || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <MetricCard title="CPU %" value={fmt(last.cpu)} series={history.map(h => h.cpu ?? null)} />
        <MetricCard title="RAM %" value={fmt(last.ramPct)} series={history.map(h => h.ramPct ?? null)} />
        <MetricCard title="Disk %" value={fmt(last.diskPct)} series={history.map(h => h.diskPct ?? null)} />
        <MetricCard title="GPU %" value={fmt(last.gpuPct)} series={history.map(h => h.gpuPct ?? null)} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="Top Containers" subtitle="CPU and memory usage">
          <ContainerTable containers={Array.isArray(data?.containers) ? data?.containers : []} />
        </Card>
        <Card title="Cleanup" subtitle="Admin only">
          {!adminKey && (
            <div style={{ marginBottom: 8, color: '#fbbf24' }}>
              Admin key not set. Go to Capacity page to save your X-Admin-Key.
            </div>
          )}
          <CleanupPanel />
        </Card>
      </div>
    </div>
  )
}

function MetricCard({ title, value, series }: { title: string; value: string; series: (number|null)[] }) {
  return (
    <Card title={title} subtitle="last 10m">
      <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: 12, alignItems: 'center' }}>
        <div style={{ fontSize: 28, color: '#e5e7eb' }}>{value}</div>
        <LineChart series={series} />
      </div>
    </Card>
  )
}

function LineChart({ series }: { series: (number|null)[] }) {
  const points = useMemo(() => {
    const vals = series.filter((x): x is number => typeof x === 'number' && isFinite(x))
    const min = Math.min(0, ...vals)
    const max = Math.max(100, ...vals)
    const w = 420, h = 80
    const N = series.length || 1
    const toX = (i: number) => (i / Math.max(1, N - 1)) * (w - 4) + 2
    const toY = (v: number) => h - 2 - ((v - min) / Math.max(1e-6, max - min)) * (h - 4)
    const pts: string[] = []
    series.forEach((v, i) => {
      if (typeof v === 'number' && isFinite(v)) {
        pts.push(`${toX(i)},${toY(v)}`)
      }
    })
    return { d: pts.join(' '), w, h }
  }, [series])
  return (
    <svg width={points.w} height={points.h} viewBox={`0 0 ${points.w} ${points.h}`} preserveAspectRatio="none" style={{ width: '100%', height: 80, background: '#0b0f17', border: '1px solid #233', borderRadius: 6 }}>
      <polyline fill="none" stroke="#60a5fa" strokeWidth="2" points={points.d} />
    </svg>
  )
}

function ContainerTable({ containers }: { containers: any[] }) {
  const top = containers.slice(0, 8)
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', gap: 6 }}>
      <HeadCell text="Name" /><HeadCell text="CPU%" /><HeadCell text="Mem%" /><HeadCell text="Mem (MB)" />
      {top.map((c) => (
        <React.Fragment key={c.name}>
          <Cell text={c.name} />
          <Cell text={fmt(c.cpu_pct)} />
          <Cell text={fmt(c.mem_pct)} />
          <Cell text={`${fmt(c.mem_used_mb)}/${fmt(c.mem_limit_mb)}`} />
        </React.Fragment>
      ))}
    </div>
  )
}

function HeadCell({ text }: { text: string }) {
  return <div style={{ color: '#9ca3af', fontSize: 12 }}>{text}</div>
}
function Cell({ text }: { text: string }) { return <div style={{ color: '#e5e7eb', fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis' }}>{text}</div> }

function toNum(x: any): number | undefined {
  const n = Number(x)
  return Number.isFinite(n) ? n : undefined
}
function fmt(n?: number): string {
  if (typeof n !== 'number' || !isFinite(n)) return '--'
  return `${n.toFixed(1)}%`
}
