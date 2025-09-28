import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import Card from './Card'
import { api } from '@/api/client'
import { useToast } from './ToastProvider'

export default function CleanupPanel() {
  const [keepLast, setKeepLast] = useState(3)
  const [olderDays, setOlderDays] = useState(30)
  const [dryRun, setDryRun] = useState(true)
  const [output, setOutput] = useState('')
  const { notify } = useToast()

  // Kafka prune (advanced)
  const [enableKafka, setEnableKafka] = useState(false)
  const [topics, setTopics] = useState('agent.results, embeddings')
  const [retentionMs, setRetentionMs] = useState(60_000)
  const [composeFile, setComposeFile] = useState('')
  const [composeService, setComposeService] = useState('kafka')
  // Docker prune (danger zone)
  const [enableDocker, setEnableDocker] = useState(false)
  const [pruneImages, setPruneImages] = useState(false)
  const [pruneVolumes, setPruneVolumes] = useState(false)
  const [ack, setAck] = useState(false)

  const run = useMutation({
    mutationFn: async () => {
      const actions: any = {
        grpo_checkpoints: { keep_last: keepLast },
        embeddings_prune: { older_than_days: olderDays }
      }
      if (enableKafka) {
        actions.kafka_prune = {
          topics: topics.split(',').map(s => s.trim()).filter(Boolean),
          retention_ms: retentionMs,
          ...(composeFile ? { compose_file: composeFile } : {}),
          ...(composeService ? { service: composeService } : {})
        }
      }
      if (enableDocker) {
        if (!ack) throw new Error('Acknowledge the danger to proceed')
        actions.docker_prune = { images: pruneImages, volumes: pruneVolumes, force: true }
      }
      const res = await api.runCleanup({ dry_run: dryRun, actions })
      return res
    },
    onSuccess: (res) => { setOutput(JSON.stringify(res, null, 2)); notify(dryRun ? 'Cleanup preview complete' : 'Cleanup completed', 'ok') },
    onError: (e: any) => { setOutput(String(e?.message || e)); notify('Cleanup failed', 'err') }
  })

  return (
    <Card title="Cleanup Helpers" subtitle="Recover space: prune old checkpoints/embeddings; optional Kafka prune">
      <div style={{ position: 'relative', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {/* Toasts displayed by ToastProvider */}
        <div>
          <Row label="GRPO: keep last N">
            <input type="number" min={1} value={keepLast} onChange={(e) => setKeepLast(parseInt(e.target.value || '3', 10))} className="input" />
          </Row>
          <Row label="Embeddings: older than (days)">
            <input type="number" min={1} value={olderDays} onChange={(e) => setOlderDays(parseInt(e.target.value || '30', 10))} className="input" />
          </Row>
          <div style={{ margin: '10px 0 6px', color: '#9ca3af' }}>Kafka Prune (advanced)</div>
          <Row label="Enable">
            <input type="checkbox" checked={enableKafka} onChange={(e) => setEnableKafka(e.target.checked)} />
          </Row>
          <Row label="Topics (comma)">
            <input value={topics} onChange={(e) => setTopics(e.target.value)} className="input" placeholder="agent.results, embeddings" />
          </Row>
          <Row label="Retention (ms)">
            <input type="number" value={retentionMs} onChange={(e) => setRetentionMs(parseInt(e.target.value || '60000', 10))} className="input" />
          </Row>
          <Row label="Compose file (opt)">
            <input value={composeFile} onChange={(e) => setComposeFile(e.target.value)} className="input" placeholder="docker/lightweight/docker-compose.yml" />
          </Row>
          <Row label="Service">
            <input value={composeService} onChange={(e) => setComposeService(e.target.value)} className="input" placeholder="kafka" />
          </Row>
          <Row label="Dry run">
            <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} />
          </Row>
          <div style={{ margin: '10px 0 6px', color: '#f87171' }}>Docker Prune (danger zone)</div>
          <Row label="Enable">
            <input type="checkbox" checked={enableDocker} onChange={(e) => setEnableDocker(e.target.checked)} />
          </Row>
          <Row label="Prune images">
            <input type="checkbox" checked={pruneImages} onChange={(e) => setPruneImages(e.target.checked)} />
          </Row>
          <Row label="Prune volumes">
            <input type="checkbox" checked={pruneVolumes} onChange={(e) => setPruneVolumes(e.target.checked)} />
          </Row>
          <Row label="Acknowledge">
            <label style={{ color: '#fecaca' }}><input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} /> I understand this can delete data</label>
          </Row>
          <button onClick={() => run.mutate()} disabled={run.isPending} className="btn">{run.isPending ? 'Running…' : (dryRun ? 'Preview Cleanup' : 'Run Cleanup')}</button>
        </div>
        <pre style={{ maxHeight: 220, overflow: 'auto', background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af' }}>{output || 'No output yet.'}</pre>
      </div>
    </Card>
  )
}

function Row({ label, children }: { label: string; children: any }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: 10, alignItems: 'center', marginBottom: 8 }}>
      <div style={{ color: '#9ca3af' }}>{label}</div>
      <div>{children}</div>
    </div>
  )
}

const inputStyle: any = {}
const btnStyle: any = {}
