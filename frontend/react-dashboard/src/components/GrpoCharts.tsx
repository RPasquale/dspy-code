import { useEffect, useMemo } from 'react'
import { Line } from 'react-chartjs-2'
import { ensureChartsRegistered } from '@/lib/registerCharts'

type M = { step: number; loss?: number; kl?: number; adv_mean?: number }

export default function GrpoCharts({ metrics }: { metrics: M[] }) {
  ensureChartsRegistered()
  const steps = metrics.map(m => m.step)
  const loss = metrics.map(m => safe(m.loss))
  const kl = metrics.map(m => safe(m.kl))
  const adv = metrics.map(m => safe(m.adv_mean))
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12 }} className="anim-slide-up">
      <Chart title="Loss" labels={steps} data={loss} color="#60a5fa" />
      <Chart title="KL" labels={steps} data={kl} color="#34d399" />
      <Chart title="Advantage (mean)" labels={steps} data={adv} color="#fbbf24" />
    </div>
  )
}

function Chart({ title, labels, data, color }: { title: string; labels: (number|string)[]; data: number[]; color: string }) {
  return (
    <div style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8 }} className="anim-fade-in">
      <div style={{ color: '#9ca3af', marginBottom: 6 }}>{title}</div>
      <div className="chart-box">
        <Line
          data={{ labels, datasets: [{ label: title, data, borderColor: color, backgroundColor: 'transparent', tension: 0.35, pointRadius: 0 }] }}
          options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { enabled: true } }, scales: { x: { grid: { display: false } }, y: { grid: { color: 'rgba(255,255,255,0.06)' } } } }}
        />
      </div>
    </div>
  )
}

function safe(n?: number): number { return typeof n === 'number' && isFinite(n) ? n : 0 }
