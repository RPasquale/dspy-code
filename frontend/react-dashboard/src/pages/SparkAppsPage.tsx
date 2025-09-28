import { useEffect, useState } from 'react'
import { api } from '../api/client'

export default function SparkAppsPage() {
  const [ns, setNs] = useState('default')
  const [apps, setApps] = useState<{ items: any[]; scheduled: any[] } | null>(null)
  const [compact, setCompact] = useState<{ items: any[]; scheduled: any[] } | null>(null)
  const [pending, setPending] = useState<{ source_file: string; rows: number; status: string; dir: string }[]>([])
  const [pattern, setPattern] = useState('')
  const [logName, setLogName] = useState('')
  const [logNs, setLogNs] = useState('default')
  const [logTail, setLogTail] = useState(200)
  const [logs, setLogs] = useState<Record<string,string>>({})
  const [toast, setToast] = useState<{ kind: 'success'|'error'|'info'; msg: string }|null>(null)
  const [trainer, setTrainer] = useState<'tiny'|'hf'>('tiny')
  const [hfModel, setHfModel] = useState('mistralai/Mistral-7B-Instruct')
  const [hfEpochs, setHfEpochs] = useState(1)
  const [hfBatch, setHfBatch] = useState(2)
  const [hfMaxLen, setHfMaxLen] = useState(1024)
  const [hfLr, setHfLr] = useState(1e-5)
  const [evTopic, setEvTopic] = useState('ui.action')
  const [evLimit, setEvLimit] = useState(50)
  const [events, setEvents] = useState<any[]>([])
  const [autoEvents, setAutoEvents] = useState(true)
  const [followOnly, setFollowOnly] = useState(false)
  const [liveEvents, setLiveEvents] = useState(true)
  const [eventSource, setEventSource] = useState<EventSource|null>(null)
  const [adminKey, setAdminKey] = useState('')
  const [pendingActs, setPendingActs] = useState<any[]>([])
  const [dsReady, setDsReady] = useState<{ manifest: string; shards: number; rows: number; ts: number }|null>(null)
  const [codeDsReady, setCodeDsReady] = useState<{ datasetDir: string; shards: number; ts: number }|null>(null)
  const [evalCode, setEvalCode] = useState('')
  const [evalOut, setEvalOut] = useState('')
  const [evalTopic, setEvalTopic] = useState('spark.log')
  const [evalLimit, setEvalLimit] = useState(200)
  const [evalScore, setEvalScore] = useState<{ score: number; bleu1: number; rougeL: number; text?: string }|null>(null)
  const [modelsInfo, setModelsInfo] = useState<any>(null)
  const [snippets, setSnippets] = useState<{ name: string; code: string }[]>(()=>{ try { return JSON.parse(localStorage.getItem('CODE_SNIPPETS')||'[]') } catch { return [] } })
  const [newSnippetName, setNewSnippetName] = useState('')
  const [sparkAppName, setSparkAppName] = useState('')
  const [sparkNamespace, setSparkNamespace] = useState('default')
  const [trainStatus, setTrainStatus] = useState<any>(null)
  const load = async () => {
    const a = await api.getSparkApps(ns)
    const c = await api.getSparkAppList(ns)
    const p = await api.getIngestPending()
    setApps(a); setCompact(c); setPending(p.pending || [])
  }
  useEffect(() => { load() }, [ns])
  useEffect(() => {
    let stop = false
    const tick = async () => {
      try { const s = await apiRequestTrainStatus(); if (!stop) setTrainStatus(s) } catch {}
    }
    tick(); const t = setInterval(tick, 15000)
    return () => { stop = true; clearInterval(t) }
  }, [])

  async function apiRequestTrainStatus() {
    try { const res = await fetch('/api/train/status'); if (!res.ok) return null; return await res.json() } catch { return null }
  }
  useEffect(() => {
    let timer: any
    const tick = async () => {
      try { if (evTopic) { const res = await api.getEventsTail(evTopic, evLimit); setEvents(res.items||[]) } }
      catch {}
    }
    if (autoEvents && !liveEvents) {
      // Kick once then poll
      tick()
      timer = setInterval(tick, 5000)
    }
    return () => { if (timer) clearInterval(timer) }
  }, [evTopic, evLimit, autoEvents, liveEvents])

  // Live SSE
  useEffect(() => {
    if (!autoEvents || !liveEvents) { if (eventSource) { eventSource.close(); setEventSource(null) } return }
    const es = new EventSource(`/api/events/stream?topic=${encodeURIComponent(evTopic)}&limit=${encodeURIComponent(String(evLimit))}${followOnly?'&follow=1':''}`)
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        const items: any[] = data.items || []
        const delta: any[] = data.delta || []
        setEvents(prev => {
          let next = prev
          if (items.length) next = items
          if (delta.length) next = [...prev, ...delta].slice(-evLimit)
          return next
        })
      } catch {}
    }
    es.onerror = () => { /* keep alive; browser will auto-retry */ }
    setEventSource(es)
    return () => { es.close(); setEventSource(null) }
  }, [evTopic, evLimit, autoEvents, liveEvents, followOnly])

  // Dataset ready banner: follow training.dataset
  useEffect(() => {
    const es = new EventSource(`/api/events/stream?topic=${encodeURIComponent('training.dataset')}&limit=5&follow=1`)
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        const items: any[] = data.items || []
        const delta: any[] = data.delta || []
        const arr = (delta && Array.isArray(delta) ? delta : []).concat(items && Array.isArray(items) ? items : [])
        const last = arr[arr.length-1]
        const e = last?.event || last
        if (e?.manifest) {
          setDsReady({ manifest: e.manifest, shards: Number(e.shards||0), rows: Number(e.rows||0), ts: Date.now()/1000 })
        } else if (e?.dataset_dir) {
          setCodeDsReady({ datasetDir: e.dataset_dir, shards: Number(e.shards||0), ts: Date.now()/1000 })
        }
      } catch {}
    }
    return () => { es.close() }
  }, [])
  useEffect(() => {
    let stop = false
    const tick = async ()=>{
      try { const info = await api.getModelsInfo(); if (!stop) setModelsInfo(info) } catch {}
    }
    tick(); const t = setInterval(tick, 30000)
    return ()=> { stop = true; clearInterval(t) }
  }, [])
  return (
    <>
      <div className="space-y-6">
      {dsReady && (
        <div className="glass-strong border border-green-500/40 p-3 rounded">
          <div className="flex items-center justify-between">
            <div className="text-green-300 text-sm">Dataset Ready: {dsReady.rows} rows in {dsReady.shards} shards</div>
            <a className="btn-secondary" href={`/api/events/export?topic=training.dataset&limit=20&download=1`} target="_blank" rel="noreferrer">Export Events</a>
          </div>
          <div className="text-slate-400 text-xs mt-1">Manifest: {dsReady.manifest}</div>
        </div>
      )}
      {codeDsReady && (
        <div className="glass-strong border border-cyan-500/40 p-3 rounded">
          <div className="flex items-center justify-between">
            <div className="text-cyan-300 text-sm">Code→Log Dataset Ready: {codeDsReady.shards} shards</div>
            <a className="btn-secondary" href={`/api/events/export?topic=training.dataset&limit=20&download=1`} target="_blank" rel="noreferrer">Export Events</a>
          </div>
          <div className="text-slate-400 text-xs mt-1">Dir: {codeDsReady.datasetDir}</div>
        </div>
      )}
      {trainStatus && (
        <div className="glass p-3">
          <div className="flex items-center justify-between">
            <div className="text-slate-200 text-sm">Training Gate: {trainStatus.ready? 'Ready' : 'Waiting'} • Shards {trainStatus.shards} / {trainStatus?.cfg?.min_shards} • Rows {trainStatus.rows} / {trainStatus?.cfg?.min_rows}</div>
            <div className="text-slate-400 text-xs">Cadence: {Math.round((trainStatus?.cfg?.train_interval_sec||86400)/3600)}h • Freshness: {trainStatus?.cfg?.min_fresh_sec||600}s</div>
          </div>
          {trainStatus.eta_fresh && (
            <div className="text-slate-400 text-xs mt-1">ETA to freshness: {new Date(trainStatus.eta_fresh*1000).toLocaleTimeString()}</div>
          )}
        </div>
      )}
      {modelsInfo && (
        <div className="glass p-3">
          <h3 className="text-white text-sm font-semibold mb-2">Models</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300">
            <div>
              <div className="text-slate-200">Code→Log (HF)</div>
              <div>Dir: {modelsInfo?.code_log?.model_dir}</div>
              <div>Size: {((modelsInfo?.code_log?.size_bytes||0)/1e6).toFixed(1)} MB</div>
              <div>Updated: {modelsInfo?.code_log?.updated_at ? new Date(modelsInfo?.code_log?.updated_at*1000).toLocaleString() : '-'}</div>
            </div>
            <div>
              <div className="text-slate-200">GRPO Tool</div>
              <div>Manifest: {modelsInfo?.grpo?.manifest}</div>
              <div>Manifest mtime: {modelsInfo?.grpo?.manifest_mtime ? new Date(modelsInfo?.grpo?.manifest_mtime*1000).toLocaleString() : '-'}</div>
              <div>Model Dir: {modelsInfo?.grpo?.model_dir || '-'}</div>
            </div>
          </div>
        </div>
      )}
      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Spark Apps</h2>
        <div className="mt-2 flex items-center space-x-2">
          <label className="text-slate-300 text-sm">Namespace</label>
          <input value={ns} onChange={e=>setNs(e.target.value)} className="input" />
          <button onClick={load} className="btn-primary">Refresh</button>
        </div>
        <div className="mt-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-300">
                <th className="p-2 text-left">Kind</th>
                <th className="p-2 text-left">Name</th>
                <th className="p-2 text-left">State/Schedule</th>
                <th className="p-2 text-left">Submitted/LastRun</th>
                <th className="p-2 text-left">Actions</th>
              </tr>
            </thead>
            <tbody>
              {(compact?.items || []).map((r, i) => (
                <tr key={`it-${i}`} className="border-t border-white/10 hover:bg-white/5">
                  <td className="p-2 text-slate-300">{r.kind}</td>
                  <td className="p-2 text-slate-200">{r.name}</td>
                  <td className="p-2">
                    <span className={`status-pill ${r.state==='RUNNING'?'status-success':(r.state==='COMPLETED'?'status-success':(r.state==='FAILED'?'status-error':'status-warning'))}`}>
                      {r.state || '-'}
                    </span>
                  </td>
                  <td className="p-2 text-slate-500 text-xs">{r.submissionTime || '-'}</td>
                  <td className="p-2">
                    <button className="btn-secondary" onClick={async ()=>{
                      try {
                        setLogName(r.name); setLogNs(r.namespace||'default')
                        const res = await api.getSparkAppLogs(r.name, r.namespace||'default', logTail)
                        setLogs(res.logs||{})
                        setToast({ kind:'info', msg: `Fetched logs for ${r.name}` })
                      } catch (e:any) {
                        setToast({ kind:'error', msg: e?.message||'Fetch failed' })
                      }
                    }}>Logs</button>
                  </td>
                </tr>
              ))}
              {(compact?.scheduled || []).map((r, i) => (
                <tr key={`sc-${i}`} className="border-t border-white/10 hover:bg-white/5">
                  <td className="p-2 text-slate-300">{r.kind}</td>
                  <td className="p-2 text-slate-200">{r.name}</td>
                  <td className="p-2"><span className="status-pill status-accent">{r.schedule || '-'}</span></td>
                  <td className="p-2 text-slate-500 text-xs">{r.lastRun || '-'}</td>
                  <td className="p-2 text-slate-500 text-xs">—</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <details className="mt-4">
          <summary className="cursor-pointer text-slate-300">Raw JSON</summary>
          <pre className="mt-3 text-xs text-slate-300 overflow-auto max-h-80 bg-black/20 p-2 rounded">{JSON.stringify(apps, null, 2)}</pre>
        </details>
      </div>

      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Capacity Controls</h2>
        <div className="mt-2 flex items-center space-x-2">
          <label className="text-slate-300 text-sm">Admin Key</label>
          <input className="input w-56" value={adminKey} onChange={e=>setAdminKey(e.target.value)} placeholder="X-Admin-Key" />
          <button className="btn-secondary" onClick={()=>{ try{ localStorage.setItem('ADMIN_KEY', adminKey||''); setToast({kind:'success', msg:'Saved key'}) }catch{} }}>Save</button>
        </div>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex items-center space-x-2">
            <label className="text-slate-300 text-sm">Storage GB</label>
            <input id="cap-gb" className="input w-32" defaultValue={10} />
            <button className="btn-success" onClick={async ()=>{
              const v = Number((document.getElementById('cap-gb') as HTMLInputElement)?.value || '0')
              try{ await api.capacityApprove({ kind: 'storage_budget_increase', params: { to_gb: v } } as any, localStorage.getItem('ADMIN_KEY')||''); setToast({kind:'success', msg:'Approved storage'}) }catch(e:any){ setToast({kind:'error', msg:e?.message||'Approve failed'}) }
            }}>Approve</button>
            <button className="btn-danger" onClick={async ()=>{
              const v = Number((document.getElementById('cap-gb') as HTMLInputElement)?.value || '0')
              try{ await api.capacityDeny({ kind: 'storage_budget_increase', value: v } as any, localStorage.getItem('ADMIN_KEY')||''); setToast({kind:'info', msg:'Denied storage'}) }catch(e:any){ setToast({kind:'error', msg:e?.message||'Deny failed'}) }
            }}>Deny</button>
          </div>
          <div className="flex items-center space-x-2">
            <label className="text-slate-300 text-sm">GPU Hours/Day</label>
            <input id="cap-gpu" className="input w-32" defaultValue={1} />
            <button className="btn-success" onClick={async ()=>{
              const v = Number((document.getElementById('cap-gpu') as HTMLInputElement)?.value || '0')
              try{ await api.capacityApprove({ kind: 'gpu_hours_increase', params: { to_hpd: v } } as any, localStorage.getItem('ADMIN_KEY')||''); setToast({kind:'success', msg:'Approved GPU hours'}) }catch(e:any){ setToast({kind:'error', msg:e?.message||'Approve failed'}) }
            }}>Approve</button>
            <button className="btn-danger" onClick={async ()=>{
              const v = Number((document.getElementById('cap-gpu') as HTMLInputElement)?.value || '0')
              try{ await api.capacityDeny({ kind: 'gpu_hours_increase', value: v } as any, localStorage.getItem('ADMIN_KEY')||''); setToast({kind:'info', msg:'Denied GPU hours'}) }catch(e:any){ setToast({kind:'error', msg:e?.message||'Deny failed'}) }
            }}>Deny</button>
          </div>
        </div>
      </div>

      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Ingest Approvals</h2>
        <div className="mt-2 flex items-center space-x-2">
          <input placeholder="/landing/csv/**/*.csv" value={pattern} onChange={e=>setPattern(e.target.value)} className="input w-96" />
          <button className="btn-success" onClick={async ()=>{ if(!pattern) return; try { await api.approveIngest(pattern); setToast({ kind:'success', msg:`Approved: ${pattern}`}); } catch(e:any){ setToast({kind:'error', msg: e?.message||'Approve failed'})} setPattern(''); await load() }}>Approve</button>
          <button className="btn-danger" onClick={async ()=>{ if(!pattern) return; try { await api.rejectIngest(pattern); setToast({ kind:'info', msg:`Rejected: ${pattern}`}); } catch(e:any){ setToast({kind:'error', msg: e?.message||'Reject failed'})} setPattern(''); await load() }}>Reject</button>
        </div>
        <table className="mt-3 w-full text-sm">
          <thead>
            <tr className="text-slate-300"><th className="text-left p-2">File</th><th className="text-left p-2">Rows</th><th className="text-left p-2">Status</th><th className="text-left p-2">Dir</th></tr>
          </thead>
          <tbody>
            {pending.map((r,i)=> (
              <tr key={i} className="border-t border-white/10 hover:bg-white/5">
                <td className="p-2 text-slate-200">{r.source_file}</td>
                <td className="p-2 text-slate-300">{r.rows}</td>
                <td className="p-2"><span className="status-pill status-warning">{r.status}</span></td>
                <td className="p-2 text-slate-500 text-xs">{r.dir}</td>
                <td className="p-2">
                  <button className="btn-success btn-xs" onClick={async ()=>{ try { await api.approveIngest(r.source_file); setToast({kind:'success', msg:`Approved: ${r.source_file}`}) } catch(e:any){ setToast({kind:'error', msg:e?.message||'Approve failed'}) } await load() }}>Approve</button>
                  <button className="btn-danger btn-xs ml-2" onClick={async ()=>{ try { await api.rejectIngest(r.source_file); setToast({kind:'info', msg:`Rejected: ${r.source_file}`}) } catch(e:any){ setToast({kind:'error', msg:e?.message||'Reject failed'}) } await load() }}>Reject</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Spark App Logs</h2>
        <div className="mt-2 flex items-center space-x-2">
          <label className="text-slate-300 text-sm">Name</label>
          <input value={logName} onChange={e=>setLogName(e.target.value)} className="input w-64" />
          <label className="text-slate-300 text-sm">Namespace</label>
          <input value={logNs} onChange={e=>setLogNs(e.target.value)} className="input w-40" />
          <label className="text-slate-300 text-sm">Tail</label>
          <input value={logTail} onChange={e=>setLogTail(Number(e.target.value)||200)} className="input w-24" />
          <button className="btn-primary" onClick={async ()=>{ if(!logName) return; const r = await api.getSparkAppLogs(logName, logNs, logTail); setLogs(r.logs || {}) }}>Fetch</button>
        </div>
        {Object.keys(logs).length === 0 ? (
          <p className="mt-3 text-slate-400 text-sm">Enter a SparkApplication name and namespace to fetch driver/executor logs.</p>
        ) : (
          <div className="mt-4 space-y-4">
            {Object.entries(logs).map(([pod, text]) => (
              <details key={pod} open className="bg-black/20 rounded">
                <summary className="cursor-pointer px-3 py-2 text-slate-200 text-sm">{pod}</summary>
                <pre className="p-3 text-xs text-slate-300 overflow-auto max-h-80 whitespace-pre-wrap">{text}</pre>
              </details>
            ))}
          </div>
        )}
      </div>

      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Train Tool Policy Now</h2>
        <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex items-center space-x-2">
            <label className="text-slate-300 text-sm">Trainer</label>
            <select className="input" value={trainer} onChange={e=>setTrainer(e.target.value as any)}>
              <option value="tiny">Tiny (GRU)</option>
              <option value="hf">HF (Transformers)</option>
            </select>
          </div>
          {trainer==='hf' && (
            <>
              <div className="flex items-center space-x-2"><label className="text-slate-300 text-sm">Model</label><input className="input w-80" value={hfModel} onChange={e=>setHfModel(e.target.value)} /></div>
              <div className="flex items-center space-x-2"><label className="text-slate-300 text-sm">Epochs</label><input className="input w-28" value={hfEpochs} onChange={e=>setHfEpochs(Number(e.target.value)||1)} /></div>
              <div className="flex items-center space-x-2"><label className="text-slate-300 text-sm">Batch</label><input className="input w-28" value={hfBatch} onChange={e=>setHfBatch(Number(e.target.value)||2)} /></div>
              <div className="flex items-center space-x-2"><label className="text-slate-300 text-sm">MaxLen</label><input className="input w-28" value={hfMaxLen} onChange={e=>setHfMaxLen(Number(e.target.value)||1024)} /></div>
              <div className="flex items-center space-x-2"><label className="text-slate-300 text-sm">LR</label><input className="input w-28" value={hfLr} onChange={e=>setHfLr(Number(e.target.value)||1e-5)} /></div>
            </>
          )}
        </div>
        <div className="mt-3">
          <button className="btn-accent" onClick={async ()=>{
            try {
              const payload:any = { trainer };
              if (trainer==='hf') payload.args = { model: hfModel, epochs: hfEpochs, batch_size: hfBatch, max_len: hfMaxLen, lr: hfLr }
              await api.triggerToolTrain(payload)
              // Log UI action via unified event API (best-effort)
              try { await api.logEvent('ui.action', { name: 'trigger_training', trainer, args: payload.args||{} }) } catch {}
              setToast({ kind:'success', msg: 'Training triggered.' })
            } catch(e:any) {
              setToast({ kind:'error', msg: e?.message||'Training trigger failed' })
            }
          }}>Trigger Training</button>
          <button className="btn-secondary ml-2" onClick={async ()=>{
            try {
              const payload:any = { trainer };
              if (trainer==='hf') payload.args = { model: hfModel, epochs: hfEpochs, batch: hfBatch, max_code: 1024, max_log: 256, lr: hfLr }
              await api.triggerCodeLogTrain(payload)
              try { await api.logEvent('ui.action', { name: 'trigger_code_log_training', trainer, args: payload.args||{} }) } catch {}
              setToast({ kind:'success', msg: 'Code→Log training triggered.' })
            } catch(e:any) {
              setToast({ kind:'error', msg: e?.message||'Code→Log trigger failed' })
            }
          }}>Train Code→Log Now</button>
        </div>
      </div>

      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Code→Log Eval (HF)</h2>
        <div className="mt-2 flex items-start space-x-3">
          <textarea className="input w-full h-32" placeholder="Paste code snippet here" value={evalCode} onChange={e=>setEvalCode(e.target.value)} />
          <div>
            <button className="btn-accent" onClick={async ()=>{
              try { const r = await api.evalCodeLog({ code: evalCode, max_new_tokens: 128 }); setEvalOut(r.text||'') } catch (e:any) { setEvalOut(`Error: ${e?.message||'failed'}`) }
            }}>Generate Logs</button>
            <div className="mt-2 space-x-2">
              <input className="input w-40" placeholder="Topic (spark.log)" value={evalTopic} onChange={e=>setEvalTopic(e.target.value)} />
              <input className="input w-24" placeholder="Limit" value={evalLimit} onChange={e=>setEvalLimit(Number(e.target.value)||200)} />
              <input className="input w-40" placeholder="SparkApp (optional)" value={sparkAppName} onChange={e=>setSparkAppName(e.target.value)} />
              <input className="input w-32" placeholder="Namespace" value={sparkNamespace} onChange={e=>setSparkNamespace(e.target.value)} />
              <button className="btn-secondary" onClick={async ()=>{
                try { const r = await api.evalCodeLogScore({ code: evalCode, topic: evalTopic, limit: evalLimit, max_new_tokens: 128, ...(sparkAppName? { spark_app: sparkAppName, namespace: sparkNamespace } : {}) } as any); setEvalScore(r?.best ? { score: r.best.score, bleu1: r.best.bleu1, rougeL: r.best.rougeL, text: r.best.log?.text } : { score: 0, bleu1: 0, rougeL: 0 }) } catch (e:any) { setEvalScore(null); setToast({kind:'error', msg: e?.message||'Score failed'}) }
              }}>Score vs Recent</button>
            </div>
            <div className="mt-3 space-y-2">
              <div className="text-slate-300 text-xs">Saved Snippets</div>
              <div className="flex items-center space-x-2">
                <select className="input w-64" onChange={e=>{ const idx = Number(e.target.value); if(!isNaN(idx) && snippets[idx]) setEvalCode(snippets[idx].code) }}>
                  <option value="">— select —</option>
                  {snippets.map((s, i)=> <option key={i} value={String(i)}>{s.name}</option>)}
                </select>
                <input className="input w-40" placeholder="Name" value={newSnippetName} onChange={e=>setNewSnippetName(e.target.value)} />
                <button className="btn-secondary" onClick={()=>{
                  const name = newSnippetName.trim() || `snippet_${Date.now()}`
                  const next = [...snippets.filter(s=>s.name!==name), { name, code: evalCode }]
                  setSnippets(next)
                  try { localStorage.setItem('CODE_SNIPPETS', JSON.stringify(next)) } catch {}
                  setNewSnippetName('')
                }}>Save</button>
                <button className="btn-danger" onClick={()=>{
                  const name = newSnippetName.trim()
                  if (!name) return
                  const next = snippets.filter(s=>s.name!==name)
                  setSnippets(next); try { localStorage.setItem('CODE_SNIPPETS', JSON.stringify(next)) } catch {}
                }}>Delete</button>
              </div>
            </div>
          </div>
        </div>
        {evalOut && (
          <div className="mt-3">
            <div className="text-slate-300 text-sm mb-1">Generated</div>
            <pre className="bg-black/20 rounded p-2 text-xs text-slate-200 whitespace-pre-wrap max-h-64 overflow-auto">{evalOut}</pre>
          </div>
        )}
        {evalScore && (
          <div className="mt-3 text-xs text-slate-300">
            <div>Score: {evalScore.score.toFixed(3)} (BLEU-1 {evalScore.bleu1.toFixed(3)} | ROUGE-L {evalScore.rougeL.toFixed(3)})</div>
            {evalScore.text && (
              <div className="mt-1"><div className="text-slate-400">Best Match</div><pre className="bg-black/20 rounded p-2 text-xs text-slate-200 whitespace-pre-wrap max-h-32 overflow-auto">{evalScore.text}</pre></div>
            )}
          </div>
        )}
      </div>

      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Events</h2>
        <div className="mt-2 flex items-center space-x-2">
          <label className="text-slate-300 text-sm">Topic</label>
          <input className="input w-56" value={evTopic} onChange={e=>setEvTopic(e.target.value)} placeholder="ui.action" />
          <label className="text-slate-300 text-sm">Limit</label>
          <input className="input w-24" value={evLimit} onChange={e=>setEvLimit(Number(e.target.value)||50)} />
          <label className="text-slate-300 text-sm">Auto</label>
          <input type="checkbox" checked={autoEvents} onChange={e=>setAutoEvents(e.target.checked)} />
          <label className="text-slate-300 text-sm">Live</label>
          <input type="checkbox" checked={liveEvents} onChange={e=>setLiveEvents(e.target.checked)} />
          <label className="text-slate-300 text-sm">Follow</label>
          <input type="checkbox" checked={followOnly} onChange={e=>setFollowOnly(e.target.checked)} />
          <button className="btn-secondary" onClick={async ()=>{
            try { const res = await api.getEventsTail(evTopic, evLimit); await navigator.clipboard.writeText(JSON.stringify(res.items||[], null, 2)); setToast({kind:'success', msg:'Copied'}) } catch(e:any){ setToast({kind:'error', msg: e?.message||'Copy failed'}) }
          }}>Copy JSON</button>
          <a className="btn-secondary" href={`/api/events/export?topic=${encodeURIComponent(evTopic)}&limit=${encodeURIComponent(String(evLimit))}&download=1`} target="_blank" rel="noreferrer">Download</a>
          <button className="btn-primary" onClick={async ()=>{
            try { const res = await api.getEventsTail(evTopic, evLimit); setEvents(res.items||[]); }
            catch(e:any){ setToast({ kind:'error', msg: e?.message||'Fetch failed' }) }
          }}>Refresh</button>
        </div>
        <div className="mt-3 max-h-80 overflow-auto bg-black/20 rounded p-2 text-xs text-slate-200">
          {events.length===0 ? <div className="text-slate-400">No events</div> : (
            <table className="w-full text-xs">
              <thead><tr className="text-slate-400"><th className="text-left p-1">Time</th><th className="text-left p-1">Summary</th></tr></thead>
              <tbody>
                {events.map((ev, i)=>{
                  const ts = typeof ev?.ts === 'number' ? new Date(ev.ts*1000).toLocaleTimeString() : (typeof ev?.timestamp==='number'? new Date(ev.timestamp*1000).toLocaleTimeString() : '')
                  const e = ev?.event || ev
                  const summary = e?.action || e?.name || e?.event || e?.status || e?.message || JSON.stringify(e).slice(0,120)
                  return <>
                    <tr key={`ev-${i}`} className="border-t border-white/10 hover:bg-white/5 cursor-pointer" onClick={(evn)=>{
                      const row = (evn.currentTarget.nextSibling as HTMLElement|undefined)
                      if (row) row.classList.toggle('hidden')
                    }}>
                      <td className="p-1 text-slate-400">{ts}</td><td className="p-1">{summary}</td>
                    </tr>
                    <tr className="hidden"><td colSpan={2} className="p-2"><pre className="whitespace-pre-wrap">{JSON.stringify(ev, null, 2)}</pre></td></tr>
                  </>
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="glass p-4">
        <h2 className="text-lg font-semibold text-white">Guardrails Actions</h2>
        <div className="mt-2 flex items-center space-x-2">
          <button className="btn-primary" onClick={async ()=>{ try{ const r = await api.getGuardrailsPending(); setPendingActs(r.pending||[]) } catch(e:any){ setToast({kind:'error', msg: e?.message||'Fetch failed'}) } }}>Refresh</button>
        </div>
        <table className="mt-3 w-full text-sm">
          <thead><tr className="text-slate-300"><th className="text-left p-2">ID</th><th className="text-left p-2">Type</th><th className="text-left p-2">Status</th><th className="text-left p-2">Actions</th></tr></thead>
          <tbody>
            {(pendingActs||[]).map((x,i)=> (
              <tr key={`ga-${i}`} className="border-t border-white/10"><td className="p-2">{x.id}</td><td className="p-2">{x.type}</td><td className="p-2">{x.status}</td><td className="p-2">
                <button className="btn-success mr-2" onClick={async ()=>{ try{ await api.approveGuardrailsAction(x.id); setToast({kind:'success', msg:'Approved'}) } catch(e:any){ setToast({kind:'error', msg:e?.message||'Approve failed'}) } }}>
                  Approve
                </button>
                <button className="btn-danger" onClick={async ()=>{ try{ await api.rejectGuardrailsAction(x.id, 'nope'); setToast({kind:'info', msg:'Rejected'}) } catch(e:any){ setToast({kind:'error', msg:e?.message||'Reject failed'}) } }}>
                  Reject
                </button>
              </td>
            </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
    {toast && (
      <div className={`fixed bottom-6 right-6 glass px-4 py-2 rounded border ${toast.kind==='success'?'border-green-400 text-green-300':(toast.kind==='error'?'border-red-400 text-red-300':'border-cyan-400 text-cyan-300')}`}
           onAnimationEnd={()=> setTimeout(()=> setToast(null), 2400)}>
        {toast.msg}
      </div>
    )}
    </>
  )
}
