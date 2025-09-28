import { useEffect, useMemo, useState } from 'react'
import TopicMultiSelect from '../components/TopicMultiSelect'
import TimeRangePicker, { TimeRange } from '../components/TimeRangePicker'
import EventsTable from '../components/EventsTable'
import { api } from '../api/client'

export default function StreamsExplorerPage() {
  const [topics, setTopics] = useState<string[]>(['spark.app'])
  const [range, setRange] = useState<TimeRange>({ label: 'Last 15m', since: (Date.now()/1000) - 15*60, until: Date.now()/1000 })
  const [q, setQ] = useState('')
  const [fields, setFields] = useState('')
  const [rows, setRows] = useState<any[]>([])
  const [live, setLive] = useState(false)
  const [es, setEs] = useState<EventSource|null>(null)
  const [presets, setPresets] = useState<{ name: string; topics: string[]; q?: string; fields?: string; range?: TimeRange }[]>(()=>{
    try { return JSON.parse(localStorage.getItem('STREAMS_PRESETS')||'[]') } catch { return [] }
  })

  const params = useMemo(()=> ({
    topics,
    since: range?.since,
    until: range?.until,
    follow: !!range?.live,
    q,
    fields,
  }), [topics, range, q, fields])

  useEffect(() => {
    if (es) { es.close(); setEs(null) }
    if (range.live) {
      const opts: any = { topics, follow: true, q }
      const fieldList = fields.split(',').map(s=>s.trim()).filter(Boolean)
      if (fieldList.length) opts.field = fieldList
      const ev = api.streamEvents(opts, data => {
        const acc: any[] = []
        if (data?.topics) {
          for (const [t, obj] of Object.entries<any>(data.topics)) {
            const items = Array.isArray(obj?.items) ? obj.items : []
            acc.push(...items)
          }
          setRows(acc)
        } else if (data?.delta) {
          const delta: any[] = []
          for (const [t, arr] of Object.entries<any>(data.delta)) {
            if (Array.isArray(arr)) delta.push(...arr)
          }
          setRows(prev => [...prev, ...delta].slice(-1000))
        }
      })
      setEs(ev)
      setLive(true)
      return () => { ev.close(); setEs(null); setLive(false) }
    }
    // Not live: fetch tails for each topic with filters
    let aborted = false
    ;(async () => {
      const out: any[] = []
      for (const t of topics) {
        try {
          const opt: any = { limit: 500, q, since: range.since, until: range.until }
          const fieldList = fields.split(',').map(s=>s.trim()).filter(Boolean)
          if (fieldList.length) { opt.keys = fieldList; opt.values = [] }
          const res = await api.getEventsTailEx(t, opt)
          out.push(...(res.items||[]))
        } catch {}
      }
      if (!aborted) setRows(out)
    })()
    return () => { aborted = true }
  }, [params.topics.join(','), params.since, params.until, params.follow, params.q, params.fields])

  return (
    <div className="space-y-6">
      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Streams Explorer</h2>
        <div className="mt-3 space-y-3">
          <div className="flex items-center space-x-3">
            <span className="text-slate-300 text-sm">Topics</span>
            <TopicMultiSelect value={topics} onChange={setTopics} />
          </div>
          <div className="flex items-center space-x-3">
            <span className="text-slate-300 text-sm">Time</span>
            <TimeRangePicker value={range} onChange={(r)=> { setRange(r); setLive(!!r.live) }} />
            <input className="input w-80" placeholder="Search (regex or text)" value={q} onChange={e=>setQ(e.target.value)} />
            <input className="input w-72" placeholder="Fields (dot paths, comma)" value={fields} onChange={e=>setFields(e.target.value)} />
            <button className={`btn-${range.live? 'danger':'success'}`} onClick={()=> setRange(prev => ({ label: prev.live? 'Custom' : 'Follow', live: !prev.live, since: prev.since, until: prev.until }))}>{range.live? 'Stop' : 'Follow'}</button>
            <a className="btn-secondary" href={`/api/events/export?topics=${encodeURIComponent(topics.join(','))}&limit=1000${q?`&q=${encodeURIComponent(q)}`:''}${range.since?`&since=${range.since}`:''}${range.until?`&until=${range.until}`:''}${fields?fields.split(',').map(f=>`&field=${encodeURIComponent(f.trim())}`).join(''):''}&download=1`} target="_blank" rel="noreferrer">Download</a>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-slate-300 text-sm">Presets</span>
            <button className="btn-secondary" onClick={()=>{
              const name = prompt('Preset name?')
              if (!name) return
              const p = { name, topics, q, fields, range }
              const next = [...presets.filter(x=>x.name!==name), p]
              setPresets(next)
              try { localStorage.setItem('STREAMS_PRESETS', JSON.stringify(next)) } catch {}
            }}>Save</button>
            {presets.map((p, i)=> (
              <button key={i} className="btn-secondary" onClick={()=>{ setTopics(p.topics); setQ(p.q||''); setFields(p.fields||''); if (p.range) setRange(p.range) }}>{p.name}</button>
            ))}
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-slate-300 text-sm">Quick Filters</span>
            <button className="btn-secondary" onClick={()=>{ setTopics(['spark.app']); setQ('FAILED') }}>Spark Failed</button>
            <button className="btn-secondary" onClick={()=>{ setTopics(['agent.action']); setQ('~/approve/i') }}>Agent Approvals</button>
            <button className="btn-secondary" onClick={()=>{ setTopics(['training.dataset']); setQ('') }}>Dataset Events</button>
          </div>
        </div>
      </div>

      <div className="glass p-4">
        <EventsTable rows={rows} onCopy={()=> { /* no-op; feedback toast could be added */ }} />
      </div>
    </div>
  )
}
