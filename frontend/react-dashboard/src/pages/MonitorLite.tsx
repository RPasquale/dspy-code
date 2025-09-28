import React, { useEffect, useState } from 'react'

type HW = { cpu_percent?: number; mem_percent?: number; gpu_name?: string; gpu_util_percent?: number; gpu_mem_used_mb?: number; gpu_mem_total_mb?: number }
type StreamRL = { rate_per_sec?: number; total?: number }
type OnlineBandit = { tools?: Record<string, any> }

export default function MonitorLite() {
  const [hw, setHW] = useState<HW>({})
  const [srl, setSRL] = useState<StreamRL>({})
  const [bandit, setBandit] = useState<OnlineBandit>({})

  useEffect(() => {
    const es = new EventSource('/api/metrics/stream')
    es.onmessage = ev => {
      try {
        const data = JSON.parse(ev.data || '{}')
        if (data.hw) setHW(data.hw)
        if (data.stream_rl) setSRL(data.stream_rl)
        if (data.online_bandit) setBandit(data.online_bandit)
      } catch {}
    }
    return () => es.close()
  }, [])

  const fmt = (n?: number, d = 1) => (typeof n === 'number' ? n.toFixed(d) : '-')

  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif' }}>
      <h2>DSPy Monitor Lite</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div style={card}>
          <h3>Hardware</h3>
          <KV k="CPU %" v={fmt(hw.cpu_percent)} />
          <KV k="Mem %" v={fmt(hw.mem_percent)} />
          <KV k="GPU" v={`${hw.gpu_name || ''} util ${fmt(hw.gpu_util_percent)}% mem ${fmt(hw.gpu_mem_used_mb)}/${fmt(hw.gpu_mem_total_mb)} MB`} />
        </div>
        <div style={card}>
          <h3>Stream RL</h3>
          <KV k="Rate / sec" v={fmt(srl.rate_per_sec, 2)} />
          <KV k="Total" v={`${srl.total ?? '-'}`} />
        </div>
        <div style={card}>
          <h3>Online Bandit</h3>
          <pre style={{ margin: 0 }}>{JSON.stringify(bandit.tools || {}, null, 2)}</pre>
        </div>
      </div>
    </div>
  )
}

const card: React.CSSProperties = { border: '1px solid #ddd', borderRadius: 6, padding: 12, minWidth: 280 }

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: 'flex', gap: 8 }}>
      <div style={{ minWidth: 120, fontWeight: 600 }}>{k}</div>
      <div>{v}</div>
    </div>
  )
}

