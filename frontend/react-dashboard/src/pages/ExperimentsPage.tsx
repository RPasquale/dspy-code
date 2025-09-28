import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '@/api/client'

const Field = ({ label, children }: { label: string; children: any }) => (
  <div className="mb-3">
    <div className="text-slate-300 text-sm mb-1">{label}</div>
    {children}
  </div>
)

export default function ExperimentsPage() {
  const [datasetPath, setDatasetPath] = useState('')
  const [preview, setPreview] = useState<string[]>([])
  const [model, setModel] = useState('BAAI/bge-small-en-v1.5')
  const [batchSize, setBatchSize] = useState<number>(32)
  const [maxCount, setMaxCount] = useState<number|undefined>(undefined)
  const [normalize, setNormalize] = useState(false)
  const [runningId, setRunningId] = useState<string|undefined>(undefined)
  const [status, setStatus] = useState<any|undefined>(undefined)
  const [logs, setLogs] = useState<string[]>([])
  const esRef = useRef<EventSource|null>(null)

  const [sweepMode, setSweepMode] = useState(false)
  const [modelsInput, setModelsInput] = useState('BAAI/bge-small-en-v1.5')
  const [batchesInput, setBatchesInput] = useState('16,32,64')

  useEffect(() => {
    return () => { if (esRef.current) { esRef.current.close(); esRef.current = null } }
  }, [])

  const startStream = (id: string) => {
    if (esRef.current) esRef.current.close()
    esRef.current = api.streamExperimentLogs(id, (line) => setLogs((prev) => [...prev.slice(-499), line]))
  }

  const onPreview = async () => {
    setPreview([])
    if (!datasetPath) return
    const res = await api.previewDataset(datasetPath)
    setPreview(res.preview || [])
  }

  const onRun = async () => {
    setLogs([]); setStatus(undefined)
    if (!datasetPath) return
    if (!sweepMode) {
      const res = await api.runExperiment({ dataset_path: datasetPath, model, batch_size: batchSize, max_count: maxCount, normalize })
      if (res.ok && res.id) {
        setRunningId(res.id)
        startStream(res.id)
        pollStatus(res.id)
      }
    } else {
      const models = modelsInput.split(',').map(s => s.trim()).filter(Boolean)
      const batches = batchesInput.split(',').map(s => parseInt(s.trim(), 10)).filter(n => Number.isFinite(n))
      const res = await api.runExperimentSweep({ dataset_path: datasetPath, models, batch_sizes: batches, max_count: maxCount, normalize })
      if (res.ok && res.id) {
        setRunningId(res.id)
        startStream(res.id)
        pollStatus(res.id)
      }
    }
  }

  const pollStatus = async (id: string) => {
    // simple poll every 1s while running
    const tick = async () => {
      try {
        const s = await api.getExperimentStatus(id)
        setStatus(s)
        if (s?.status && ['completed','error'].includes(s.status)) return
        setTimeout(tick, 1000)
      } catch {
        setTimeout(tick, 2000)
      }
    }
    tick()
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h2 className="text-2xl font-semibold text-white mb-4">Experiments</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass p-4 rounded-lg border border-white/10">
          <Field label="Dataset Path (JSONL with {text} or TXT)">
            <div className="flex gap-2">
              <input className="input" placeholder="data/my_texts.jsonl" value={datasetPath} onChange={(e) => setDatasetPath(e.target.value)} />
              <button className="btn" onClick={onPreview}>Preview</button>
            </div>
            {!!preview?.length && (
              <ul className="text-slate-300 text-xs mt-2 list-disc pl-4">
                {preview.map((p, i) => <li key={i}>{p.slice(0, 120)}</li>)}
              </ul>
            )}
          </Field>
          <div className="flex items-center gap-4 mb-3">
            <label className="inline-flex items-center gap-2 text-slate-300 text-sm">
              <input type="checkbox" checked={sweepMode} onChange={(e) => setSweepMode(e.target.checked)} /> Sweep Mode
            </label>
          </div>
          {!sweepMode ? (
            <>
              <Field label="Model">
                <input className="input" value={model} onChange={(e) => setModel(e.target.value)} />
              </Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Batch Size">
                  <input className="input" type="number" value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value,10)||32)} />
                </Field>
                <Field label="Max Count (optional)">
                  <input className="input" type="number" value={maxCount ?? ''} onChange={(e) => setMaxCount(e.target.value ? parseInt(e.target.value,10) : undefined)} />
                </Field>
              </div>
              <label className="inline-flex items-center gap-2 text-slate-300 text-sm mb-3">
                <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} /> Normalize vectors
              </label>
            </>
          ) : (
            <>
              <Field label="Models (comma-separated)">
                <input className="input" value={modelsInput} onChange={(e) => setModelsInput(e.target.value)} />
              </Field>
              <Field label="Batch Sizes (comma-separated)">
                <input className="input" value={batchesInput} onChange={(e) => setBatchesInput(e.target.value)} />
              </Field>
              <Field label="Max Count (optional)">
                <input className="input" type="number" value={maxCount ?? ''} onChange={(e) => setMaxCount(e.target.value ? parseInt(e.target.value,10) : undefined)} />
              </Field>
            </>
          )}
          <div className="mt-4 flex gap-2">
            <button className="btn-primary" onClick={onRun} disabled={!datasetPath}>Start</button>
            {runningId && <span className="text-slate-400 text-sm">Run ID: {runningId}</span>}
          </div>
        </div>
        <div className="glass p-4 rounded-lg border border-white/10">
          <div className="text-slate-300 text-sm mb-2">Status</div>
          <pre className="bg-black/30 text-slate-200 text-xs p-3 rounded min-h-[100px]">
{JSON.stringify(status ?? {}, null, 2)}
          </pre>
          <div className="text-slate-300 text-sm mt-4 mb-2">Live Logs</div>
          <div className="bg-black/30 text-green-300 text-xs p-3 rounded h-64 overflow-auto">
            {logs.map((ln, i) => (<div key={i} className="whitespace-pre-wrap">{ln}</div>))}
          </div>
        </div>
      </div>
    </div>
  )
}

