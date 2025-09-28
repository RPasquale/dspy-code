import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import { useState } from 'react'

export default function EnvQueuePage() {
  const qc = useQueryClient()
  const { data } = useQuery({ queryKey: ['env-queue-stats'], queryFn: api.getEnvQueueStats, refetchInterval: 4000 })
  const [taskId, setTaskId] = useState('')
  const [cls, setCls] = useState<'cpu_short'|'cpu_long'|'gpu'>('cpu_short')
  const [payload, setPayload] = useState('')
  const mut = useMutation({ mutationFn: api.submitEnvTask, onSuccess: ()=> qc.invalidateQueries({ queryKey: ['env-queue-stats'] }) })
  const cfgQ = useQuery({ queryKey: ['env-queue-cfg'], queryFn: async ()=> (await fetch('/api/env-queue/config')).json().then(r=>r.config || {}), refetchInterval: 10000 })
  const [cfg, setCfg] = useState<any>({})
  const saveCfg = useMutation({ mutationFn: async (c:any)=> fetch('/api/env-queue/config', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(c) }), onSuccess: ()=> cfgQ.refetch() })
  const cfgVal = cfgQ.data || cfg
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white">Env Queue</h2>
      <div className="glass p-4 rounded-lg border border-white/10">
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="text-slate-400 text-sm">Pending</div>
            <div className="text-3xl text-yellow-300">{data?.pending ?? 0}</div>
          </div>
          <div>
            <div className="text-slate-400 text-sm">Done</div>
            <div className="text-3xl text-green-300">{data?.done ?? 0}</div>
          </div>
          <div>
            <div className="text-slate-400 text-sm">Last Updated</div>
            <div className="text-lg text-slate-200">{data?.timestamp ? new Date((data.timestamp||0)*1000).toLocaleTimeString() : '-'}</div>
          </div>
        </div>
      </div>
      <div className="glass p-4 rounded-lg border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-2">Submit Task</h3>
        <div className="flex items-center gap-2">
          <input className="input w-48" placeholder="ID (auto)" value={taskId} onChange={e=> setTaskId(e.target.value)} />
          <select className="input w-36" value={cls} onChange={e=> setCls(e.target.value as any)}>
            <option value="cpu_short">CPU Short</option>
            <option value="cpu_long">CPU Long</option>
            <option value="gpu">GPU</option>
          </select>
          <input className="input flex-1" placeholder="Payload" value={payload} onChange={e=> setPayload(e.target.value)} />
          <button className="btn-primary" onClick={()=> mut.mutate({ id: taskId || undefined, class: cls, payload })} disabled={mut.isPending}>
            {mut.isPending ? 'Submitting…' : 'Submit'}
          </button>
        </div>
        {mut.isError && <div className="text-red-300 mt-2">{(mut.error as any)?.message || 'Submit failed'}</div>}
      </div>
      <div className="glass p-4 rounded-lg border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-2">Orchestrator Config</h3>
        <div className="grid grid-cols-2 gap-3">
          <label className="text-slate-300 text-sm">Base Limit<input className="input" defaultValue={cfgVal.BaseLimit ?? ''} onChange={e=> setCfg({...cfgVal, BaseLimit: Number(e.target.value)||0})} /></label>
          <label className="text-slate-300 text-sm">Min Limit<input className="input" defaultValue={cfgVal.MinLimit ?? ''} onChange={e=> setCfg({...cfgVal, MinLimit: Number(e.target.value)||0})} /></label>
          <label className="text-slate-300 text-sm">Max Limit<input className="input" defaultValue={cfgVal.MaxLimit ?? ''} onChange={e=> setCfg({...cfgVal, MaxLimit: Number(e.target.value)||0})} /></label>
          <label className="text-slate-300 text-sm">Increase Step<input className="input" defaultValue={cfgVal.IncreaseStep ?? ''} onChange={e=> setCfg({...cfgVal, IncreaseStep: Number(e.target.value)||0})} /></label>
          <label className="text-slate-300 text-sm">Decrease Step<input className="input" defaultValue={cfgVal.DecreaseStep ?? ''} onChange={e=> setCfg({...cfgVal, DecreaseStep: Number(e.target.value)||0})} /></label>
          <label className="text-slate-300 text-sm">Adaptation (ms)<input className="input" defaultValue={cfgVal.AdaptationIntervalMs ?? ''} onChange={e=> setCfg({...cfgVal, AdaptationIntervalMs: Number(e.target.value)||0})} /></label>
        </div>
        <div className="mt-2">
          <button className="btn-secondary" onClick={()=> saveCfg.mutate(cfg)}>Save Config</button>
        </div>
      </div>
      <div className="glass p-4 rounded-lg border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-2">Recent Items</h3>
        <div className="text-slate-300 text-sm">
          {(data?.items ?? []).slice(-20).reverse().map((it: any, idx: number) => (
            <div key={idx} className="flex items-center justify-between border-b border-white/5 py-1">
              <div>
                <span className={`px-2 py-0.5 rounded text-xs ${it.status==='pending' ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300'}`}>{it.status}</span>
                <span className="ml-2 text-slate-200">{it.id}</span>
              </div>
              <div className="text-slate-400">{it.file}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
