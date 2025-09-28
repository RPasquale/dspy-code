import { useMemo, useState } from 'react'

export default function EventsTable({ rows, onCopy }: { rows: any[]; onCopy?: (subset: any[]) => void }) {
  const [sel, setSel] = useState<Set<number>>(new Set())
  const toggle = (i: number) => setSel(prev => { const n = new Set(prev); if (n.has(i)) n.delete(i); else n.add(i); return n })
  const all = rows || []
  const subset = useMemo(()=> all.filter((_,i)=> sel.has(i)), [all, sel])
  const copyJSON = async () => {
    const text = JSON.stringify(subset.length? subset : all, null, 2)
    try { await navigator.clipboard.writeText(text) } catch {}
    onCopy?.(subset.length? subset : all)
  }
  const copyCSV = async () => {
    const cols = ['topic','ts','service','event.name','event.action','event.status']
    const get = (obj: any, path: string) => path.split('.').reduce((acc,k)=> (acc && typeof acc==='object')? acc[k] : undefined, obj)
    const lines = [cols.join(',')].concat((subset.length? subset : all).map(r => cols.map(c => JSON.stringify(get(r,c) ?? '')).join(',')))
    try { await navigator.clipboard.writeText(lines.join('\n')) } catch {}
    onCopy?.(subset.length? subset : all)
  }
  return (
    <div>
      <div className="flex items-center space-x-2 mb-2">
        <button className="btn-secondary" onClick={()=> setSel(new Set())}>Clear</button>
        <button className="btn-secondary" onClick={()=> setSel(new Set(all.map((_,i)=>i)))}>Select All</button>
        <button className="btn-primary" onClick={copyJSON}>Copy JSON</button>
        <button className="btn-primary" onClick={copyCSV}>Copy CSV</button>
      </div>
      <div className="max-h-[32rem] overflow-auto bg-black/20 rounded p-2 text-xs text-slate-200">
        <table className="w-full text-xs">
          <thead><tr className="text-slate-400"><th className="p-1">Sel</th><th className="text-left p-1">Topic</th><th className="text-left p-1">Time</th><th className="text-left p-1">Summary</th></tr></thead>
          <tbody>
            {all.map((rec, i) => {
              const e: any = rec?.event || rec
              const ts = typeof rec?.ts==='number' ? new Date(rec.ts*1000).toLocaleString() : (typeof rec?.timestamp==='number'? new Date(rec.timestamp*1000).toLocaleString():'')
              const topic = rec?.topic || e?.topic || ''
              const summary = e?.action || e?.name || e?.event || e?.status || e?.message || JSON.stringify(e).slice(0,120)
              return <>
                <tr key={`ev-${i}`} className="border-t border-white/10 hover:bg-white/5">
                  <td className="p-1"><input type="checkbox" checked={sel.has(i)} onChange={()=> toggle(i)} /></td>
                  <td className="p-1 text-slate-400">{topic}</td>
                  <td className="p-1 text-slate-400">{ts}</td>
                  <td className="p-1">{summary}</td>
                </tr>
                <tr><td colSpan={4} className="p-2"><details><summary className="cursor-pointer text-slate-400">Details</summary><pre className="whitespace-pre-wrap">{JSON.stringify(rec, null, 2)}</pre></details></td></tr>
              </>
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

