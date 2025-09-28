import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'

type Row = { topic: string; rec: any }

export default function EventsPage() {
  const [topics, setTopics] = useState('spark.app,ui.action')
  const [limit, setLimit] = useState(200)
  const [follow, setFollow] = useState(true)
  const [live, setLive] = useState(true)
  const [events, setEvents] = useState<Row[]>([])
  const [q, setQ] = useState('')
  const [es, setEs] = useState<EventSource|null>(null)
  const [downloading, setDownloading] = useState(false)

  useEffect(() => {
    if (!live) { if (es) { es.close(); setEs(null) } return }
    const list = topics.split(',').map(s=>s.trim()).filter(Boolean)
    const ev = api.streamEvents({ topics: list, limit, follow }, (data) => {
      try {
        const out: Row[] = []
        if (data?.topics && typeof data.topics === 'object') {
          for (const [t, obj] of Object.entries<any>(data.topics)) {
            const items = Array.isArray(obj?.items) ? obj.items : []
            for (const it of items) out.push({ topic: t, rec: it })
          }
          setEvents(out.slice(-limit))
          return
        }
        if (data?.delta && typeof data.delta === 'object') {
          for (const [t, arr] of Object.entries<any>(data.delta)) {
            const items = Array.isArray(arr) ? arr : []
            for (const it of items) out.push({ topic: t, rec: it })
          }
          setEvents(prev => [...prev, ...out].slice(-Math.max(limit, 50)))
        }
      } catch {}
    })
    setEs(ev)
    return () => { ev.close(); setEs(null) }
  }, [topics, limit, follow, live])

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase()
    if (!s) return events
    return events.filter(r => JSON.stringify(r.rec).toLowerCase().includes(s))
  }, [events, q])

  return (
    <div className="space-y-6">
      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Events Stream</h2>
        <div className="mt-2 flex items-center space-x-2">
          <label className="text-slate-300 text-sm">Topics</label>
          <input className="input w-96" value={topics} onChange={e=>setTopics(e.target.value)} placeholder="topic1,topic2" />
          <label className="text-slate-300 text-sm">Limit</label>
          <input className="input w-24" value={limit} onChange={e=>setLimit(Number(e.target.value)||200)} />
          <label className="text-slate-300 text-sm">Follow</label>
          <input type="checkbox" checked={follow} onChange={e=>setFollow(e.target.checked)} />
          <label className="text-slate-300 text-sm">Live</label>
          <input type="checkbox" checked={live} onChange={e=>setLive(e.target.checked)} />
          <input className="input w-64" placeholder="Search (JSON includes)" value={q} onChange={e=>setQ(e.target.value)} />
          <button className="btn-secondary" onClick={async ()=>{
            try {
              setDownloading(true)
              const qs = new URLSearchParams()
              if (topics) qs.set('topics', topics)
              qs.set('limit', String(limit))
              if (q) qs.set('q', q)
              qs.set('download', '1')
              // Use browser navigation to stream download
              window.open(`/api/events/export?${qs.toString()}`, '_blank')
            } finally { setDownloading(false) }
          }}>{downloading? 'Preparing…' : 'Download'}</button>
        </div>
        <div className="mt-3 max-h-[32rem] overflow-auto bg-black/20 rounded p-2 text-xs text-slate-200">
          {filtered.length===0 ? <div className="text-slate-400">No events</div> : (
            <table className="w-full text-xs">
              <thead><tr className="text-slate-400"><th className="text-left p-1">Topic</th><th className="text-left p-1">Time</th><th className="text-left p-1">Summary</th></tr></thead>
              <tbody>
                {filtered.map((r, i)=>{
                  const ev: any = r.rec?.event || r.rec
                  const ts = typeof r.rec?.ts==='number' ? new Date(r.rec.ts*1000).toLocaleTimeString() : (typeof r.rec?.timestamp==='number'? new Date(r.rec.timestamp*1000).toLocaleTimeString():'')
                  const summary = ev?.action || ev?.name || ev?.event || ev?.status || ev?.message || JSON.stringify(ev).slice(0,120)
                  return <>
                    <tr key={`row-${i}`} className="border-t border-white/10 hover:bg-white/5 cursor-pointer" onClick={(evn)=>{ const n = (evn.currentTarget.nextSibling as HTMLElement|undefined); if(n) n.classList.toggle('hidden') }}>
                      <td className="p-1 text-slate-400">{r.topic}</td>
                      <td className="p-1 text-slate-400">{ts}</td>
                      <td className="p-1">{summary}</td>
                    </tr>
                    <tr className="hidden"><td colSpan={3} className="p-2"><pre className="whitespace-pre-wrap">{JSON.stringify(r.rec, null, 2)}</pre></td></tr>
                  </>
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
