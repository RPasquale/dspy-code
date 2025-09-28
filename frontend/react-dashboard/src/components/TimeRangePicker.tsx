import { useMemo, useState } from 'react'

export type TimeRange = { label: string; since?: number; until?: number; live?: boolean }

const presets = [
  { label: 'Last 5m', minutes: 5 },
  { label: 'Last 15m', minutes: 15 },
  { label: 'Last 1h', minutes: 60 },
  { label: 'Last 24h', minutes: 24*60 },
]

export default function TimeRangePicker({ value, onChange }: { value?: TimeRange; onChange: (tr: TimeRange) => void }) {
  const [custom, setCustom] = useState<{ since?: string; until?: string }>({})

  const now = useMemo(()=> Date.now()/1000, [])
  return (
    <div className="flex items-center space-x-2">
      {presets.map(p => (
        <button key={p.label} className="btn-secondary" onClick={()=> onChange({ label: p.label, since: now - p.minutes*60, until: now })}>{p.label}</button>
      ))}
      <button className="btn-secondary" onClick={()=> onChange({ label: 'Follow', live: true })}>Follow</button>
      <div className="flex items-center space-x-1">
        <input className="input w-44" placeholder="since ISO or epoch" value={custom.since||''} onChange={e=>setCustom(prev=>({ ...prev, since: e.target.value }))} />
        <input className="input w-44" placeholder="until ISO or epoch" value={custom.until||''} onChange={e=>setCustom(prev=>({ ...prev, until: e.target.value }))} />
        <button className="btn-primary" onClick={()=>{
          const parse = (s?: string) => {
            if (!s) return undefined
            const n = Number(s); if (!Number.isNaN(n) && n>0) return n
            const d = new Date(s); if (!isNaN(d.getTime())) return d.getTime()/1000
            return undefined
          }
          onChange({ label: 'Custom', since: parse(custom.since), until: parse(custom.until) })
        }}>Apply</button>
      </div>
    </div>
  )
}

