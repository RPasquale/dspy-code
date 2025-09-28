import { useEffect, useMemo, useState } from 'react'
import { api } from '@/api/client'
import Card from './Card'

type Sample = { ts: number; cpu?: number; ram?: number }

export default function MiniSystemCharts() {
  const [hist, setHist] = useState<Sample[]>([])
  useEffect(() => {
    let es: EventSource | null = null
    try {
      es = new EventSource('/api/system/resources/stream')
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          const now = msg?.host?.timestamp || (Date.now() / 1000)
          const cpu = msg?.host?.cpu?.cpu_pct ?? msg?.host?.cpu?.load1
          const ram = msg?.host?.memory?.pct_used ?? msg?.host?.memory?.mem_pct ?? msg?.host?.mem?.mem_pct
          setHist((prev) => [...prev, { ts: now, cpu: num(cpu), ram: num(ram) }].slice(-120))
        } catch {}
      }
      es.onerror = () => {}
    } catch {
      // Fallback: poll occasionally if SSE fails
      const t = setInterval(async () => {
        try {
          const res: any = await api.getSystemResources()
          const now = res?.host?.timestamp || (Date.now() / 1000)
          const cpu = res?.host?.cpu?.cpu_pct ?? res?.host?.cpu?.load1
          const ram = res?.host?.memory?.pct_used ?? res?.host?.mem?.mem_pct
          setHist((prev) => [...prev, { ts: now, cpu: num(cpu), ram: num(ram) }].slice(-120))
        } catch {}
      }, 5000)
      return () => clearInterval(t)
    }
    return () => { try { es?.close() } catch {} }
  }, [])

  const cpuSeries = hist.map(h => h.cpu ?? null)
  const ramSeries = hist.map(h => h.ram ?? null)

  return (
    <Card title="System Trends" subtitle="10 min CPU/RAM">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <Metric title="CPU %" series={cpuSeries} />
        <Metric title="RAM %" series={ramSeries} />
      </div>
    </Card>
  )
}

function Metric({ title, series }: { title: string; series: (number|null)[] }) {
  const last = series.length ? series[series.length - 1] : undefined
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <div style={{ color: '#e5e7eb', fontSize: 20 }}>{fmt(last)}</div>
        <div style={{ color: '#9ca3af', fontSize: 12 }}>{title}</div>
      </div>
      <LineChart series={series} />
    </div>
  )
}

function LineChart({ series }: { series: (number|null)[] }) {
  const points = useMemo(() => {
    const vals = series.filter((x): x is number => typeof x === 'number' && isFinite(x))
    const min = Math.min(0, ...vals)
    const max = Math.max(100, ...vals)
    const w = 320, h = 68
    const N = series.length || 1
    const toX = (i: number) => (i / Math.max(1, N - 1)) * (w - 4) + 2
    const toY = (v: number) => h - 2 - ((v - min) / Math.max(1e-6, max - min)) * (h - 4)
    const pts: string[] = []
    series.forEach((v, i) => { if (typeof v === 'number' && isFinite(v)) pts.push(`${toX(i)},${toY(v)}`) })
    return { d: pts.join(' '), w, h }
  }, [series])
  return (
    <svg width={points.w} height={points.h} viewBox={`0 0 ${points.w} ${points.h}`} preserveAspectRatio="none" style={{ width: '100%', height: 68, background: '#0b0f17', border: '1px solid #233', borderRadius: 6 }}>
      <polyline fill="none" stroke="#34d399" strokeWidth="2" points={points.d} />
    </svg>
  )
}

function num(x: any): number | undefined { const n = Number(x); return Number.isFinite(n) ? n : undefined }
function fmt(n?: number | null): string { return typeof n === 'number' && isFinite(n) ? `${n.toFixed(1)}%` : '--' }

