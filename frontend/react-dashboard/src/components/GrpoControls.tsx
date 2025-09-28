import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import Card from './Card';
import { api } from '@/api/client';
import BlockingOverlay from '@/components/BlockingOverlay'
import { useToast } from './ToastProvider'
import GrpoCharts from './GrpoCharts'

type StartPayload = {
  dataset_path: string;
  model_name?: string;
  reference_model_name?: string;
  device?: string;
  batch_groups?: number;
  lr?: number;
  max_steps?: number;
  log_interval?: number;
  ckpt_interval?: number;
  adv_clip?: number;
  kl_coeff?: number;
}

const Row = ({ label, children }: { label: string; children: any }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: 12, alignItems: 'center', margin: '6px 0' }}>
    <label style={{ color: '#ddd' }}>{label}</label>
    <div>{children}</div>
  </div>
);

const Input = (props: any) => (
  <input {...props} className={`input ${props.className || ''}`} />
);

const NumberInput = (props: any) => (
  <input type="number" {...props} className={`input ${props.className || ''}`} />
);

const Button = ({ children, className = '', ...rest }: any) => (
  <button {...rest} className={`btn ${className}`}>{children}</button>
);

export default function GrpoControls() {
  const { notify } = useToast()
  const [payload, setPayload] = useState<StartPayload>({ dataset_path: '/app/grpo.jsonl', max_steps: 1000, batch_groups: 8, lr: 1e-5, log_interval: 20, ckpt_interval: 200, adv_clip: 5.0, kl_coeff: 0.02 });
  const [seededPath, setSeededPath] = useState<string>('')
  useEffect(() => {
    try {
      const p = localStorage.getItem('grpo_dataset_path') || ''
      if (p) { setSeededPath(p); setPayload((prev) => ({ ...prev, dataset_path: p })) }
    } catch {}
  }, [])

  const status = useQuery({ queryKey: ['grpo-status'], queryFn: api.getGrpoStatus, refetchInterval: 3000 });
  const metrics = useQuery({ queryKey: ['grpo-metrics'], queryFn: () => api.getGrpoMetrics(200), refetchInterval: 5000 });
  const auto = useQuery({ queryKey: ['grpo-auto-status'], queryFn: api.getGrpoAutoStatus, refetchInterval: 5000 });
  const resources = useQuery({ queryKey: ['system-resources'], queryFn: api.getSystemResources, refetchInterval: 5000 });
  const guard = useQuery({ queryKey: ['system-guard'], queryFn: () => api.guardSystem({ min_free_gb: 5, min_ram_gb: 2, min_vram_mb: 0 }), refetchInterval: 5000 });
  const [streamSeries, setStreamSeries] = useState<any[]>([])
  const [tab, setTab] = useState<'stream'|'global'|'signature'|'patch'>('stream')
  const [levelSeries, setLevelSeries] = useState<Record<string, any[]>>({})
  useEffect(() => {
    let es: EventSource | null = null
    try {
      es = new EventSource('/api/grpo/metrics/stream')
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          const delta = Array.isArray(msg?.delta) ? msg.delta : []
          if (delta.length) setStreamSeries((prev) => [...prev, ...delta].slice(-500))
        } catch {}
      }
      es.onerror = () => {}
    } catch {}
    return () => { try { es?.close() } catch {} }
  }, [])

  const start = useMutation({ mutationFn: () => api.startGrpo(payload), onSuccess: () => { status.refetch(); metrics.refetch(); } });
  const stop = useMutation({ mutationFn: () => api.stopGrpo(), onSuccess: () => status.refetch() });
  const startAuto = useMutation({ mutationFn: () => api.startGrpoAuto({ mode: 'hierarchical', period_sec: 300, hours: 24, min_groups: 20, steps: 400, out_dir: '/app/.grpo/auto', apply_policy: true }), onSuccess: () => auto.refetch() });
  const stopAuto = useMutation({ mutationFn: () => api.stopGrpoAuto(), onSuccess: () => auto.refetch() });

  const running = !!status.data?.running;
  const err = status.data?.error;
  const baseSeries = useMemo(() => (streamSeries.length ? streamSeries : (metrics.data?.metrics || [])), [streamSeries, metrics.data]);
  const last = useMemo(() => (baseSeries.length ? baseSeries[baseSeries.length - 1] : null), [baseSeries]);
  const metricSeries = useMemo(() => baseSeries.slice(-200), [baseSeries]);
  const levels = useMemo(() => {
    const ls: ('global'|'signature'|'patch')[] = []
    const m = auto.data?.last_datasets || {}
    ;(['global','signature','patch'] as const).forEach((k) => { if (m[k]) ls.push(k) })
    return ls
  }, [auto.data])
  const levelMetrics = useQuery({
    queryKey: ['grpo-level-metrics', tab],
    queryFn: async () => {
      if (tab === 'stream') return { metrics: metricSeries }
      return await api.getGrpoLevelMetrics(tab as any)
    },
    enabled: tab !== 'stream',
    refetchInterval: tab === 'stream' ? false as any : 5000
  })

  // SSE for selected level (when not in 'stream' tab)
  useEffect(() => {
    if (tab === 'stream') return
    let es: EventSource | null = null
    try {
      es = new EventSource(`/api/grpo/level-metrics/stream?level=${encodeURIComponent(tab)}`)
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          const delta = Array.isArray(msg?.delta) ? msg.delta : []
          if (!delta.length) return
          setLevelSeries((prev) => ({ ...prev, [tab]: [ ...(prev[tab] || []), ...delta ].slice(-500) }))
        } catch {}
      }
      es.onerror = () => {}
    } catch {}
    return () => { try { es?.close() } catch {} }
  }, [tab])
  const [dsStats, setDsStats] = useState<any | null>(null);

  const blocked = !!guard.data && !guard.data.ok
  return (
    <Card title="GRPO Training" subtitle="Group Relative Preference Optimization (Torch)">
      <div style={{ position: 'relative' }}>
        {blocked && (
          <BlockingOverlay reason={`Insufficient capacity: ${!guard.data?.disk_ok ? 'disk ' : ''}${!guard.data?.ram_ok ? 'ram ' : ''}${!guard.data?.gpu_ok ? 'gpu ' : ''}below threshold.`} />
        )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div>
          <Row label="Dataset Path">
            <Input value={payload.dataset_path} onChange={(e: any) => setPayload(p => ({ ...p, dataset_path: e.target.value }))} />
          </Row>
          {!!seededPath && (
            <div style={{ color: '#9ca3af', fontSize: 12, marginTop: -6, marginBottom: 6 }}>
              Using last seeded dataset: {seededPath}
            </div>
          )}
          <Row label="Model Name">
            <Input placeholder="e.g., gpt2 or leave empty for fallback" value={payload.model_name || ''} onChange={(e: any) => setPayload(p => ({ ...p, model_name: e.target.value || undefined }))} />
          </Row>
          <Row label="Reference Model">
            <Input placeholder="defaults to model_name" value={payload.reference_model_name || ''} onChange={(e: any) => setPayload(p => ({ ...p, reference_model_name: e.target.value || undefined }))} />
          </Row>
          <Row label="Device">
            <Input placeholder="cuda or cpu (auto if empty)" value={payload.device || ''} onChange={(e: any) => setPayload(p => ({ ...p, device: e.target.value || undefined }))} />
          </Row>
          <Row label="Batch (groups)">
            <NumberInput value={payload.batch_groups} min={1} step={1} onChange={(e: any) => setPayload(p => ({ ...p, batch_groups: parseInt(e.target.value || '8', 10) }))} />
          </Row>
          <Row label="Max Steps">
            <NumberInput value={payload.max_steps} min={1} step={100} onChange={(e: any) => setPayload(p => ({ ...p, max_steps: parseInt(e.target.value || '1000', 10) }))} />
          </Row>
          <Row label="LR">
            <NumberInput value={payload.lr} step={1e-5} onChange={(e: any) => setPayload(p => ({ ...p, lr: parseFloat(e.target.value || '1e-5') }))} />
          </Row>
          <Row label="Log Interval">
            <NumberInput value={payload.log_interval} min={1} step={1} onChange={(e: any) => setPayload(p => ({ ...p, log_interval: parseInt(e.target.value || '20', 10) }))} />
          </Row>
          <Row label="CKPT Interval">
            <NumberInput value={payload.ckpt_interval} min={10} step={10} onChange={(e: any) => setPayload(p => ({ ...p, ckpt_interval: parseInt(e.target.value || '200', 10) }))} />
          </Row>
          <Row label="Adv Clip">
            <NumberInput value={payload.adv_clip} step={0.5} onChange={(e: any) => setPayload(p => ({ ...p, adv_clip: parseFloat(e.target.value || '5') }))} />
          </Row>
          <Row label="KL Coeff">
            <NumberInput value={payload.kl_coeff} step={0.01} onChange={(e: any) => setPayload(p => ({ ...p, kl_coeff: parseFloat(e.target.value || '0.02') }))} />
          </Row>
          <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
            <Button className="btn-primary" onClick={() => start.mutate()} disabled={running || start.isPending || (guard.data && !guard.data.ok)}>{start.isPending ? 'Starting…' : 'Start'}</Button>
            <Button onClick={() => stop.mutate()} disabled={!running || stop.isPending}>{stop.isPending ? 'Stopping…' : 'Stop'}</Button>
            <Button onClick={() => startAuto.mutate()} disabled={auto.data?.running || startAuto.isPending || (guard.data && !guard.data.ok)}>{startAuto.isPending ? 'Auto…' : 'Start Auto'}</Button>
            <Button onClick={() => stopAuto.mutate()} disabled={!auto.data?.running || stopAuto.isPending}>{stopAuto.isPending ? 'Stopping…' : 'Stop Auto'}</Button>
          </div>
          {err && <div style={{ color: 'crimson', marginTop: 8 }}>Error: {String(err)}</div>}
          <div style={{ marginTop: 8, color: '#9ca3af' }}>Auto: {auto.data?.running ? 'running' : 'stopped'} {auto.data?.mode ? `(mode=${auto.data?.mode})` : ''}</div>
          {!!auto.data?.last_datasets && (
            <pre style={{ marginTop: 6, background: '#0b0f17', color: '#9ca3af', border: '1px solid #233', borderRadius: 6, padding: 8, maxHeight: 140, overflow: 'auto' }}>{JSON.stringify(auto.data.last_datasets, null, 2)}</pre>
          )}
          <div style={{ marginTop: 8, color: guard.data?.ok ? '#10b981' : '#f59e0b' }}>
            Guard: {guard.data?.ok ? 'OK' : 'Insufficient capacity'}
            {!guard.data?.ok && (
              <span style={{ marginLeft: 8 }}>
                {!guard.data?.disk_ok ? 'disk ' : ''}
                {!guard.data?.ram_ok ? 'ram ' : ''}
                {!guard.data?.gpu_ok ? 'gpu ' : ''}
              </span>
            )}
          </div>
        </div>
        <div>
          {(!metrics.data || !(metrics.data.metrics||[]).length) ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div className="skeleton skeleton-block" />
              <div className="skeleton skeleton-block" />
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <Metric label="Running" value={running ? 'yes' : 'no'} />
              <Metric label="Step" value={String(status.data?.step ?? 0)} />
              <Metric label="Loss" value={last ? last.loss?.toFixed(4) : '--'} />
              <Metric label="KL" value={last ? last.kl?.toFixed(4) : '--'} />
              <Metric label="Adv μ±σ" value={last ? `${(last.adv_mean ?? 0).toFixed(3)} ± ${(last.adv_std ?? 0).toFixed(3)}` : '--'} />
              <Metric label="LR" value={last ? (last.lr ?? 0).toExponential(2) : '--'} />
            </div>
          )}
          <div style={{ marginTop: 12 }}>
            <div className="tabbar" style={{ marginBottom: 8 }}>
              <TabButton active={tab==='stream'} onClick={() => setTab('stream')}>Stream</TabButton>
              {levels.map((lv) => <TabButton key={lv} active={tab===lv} onClick={() => setTab(lv)}>{lv}</TabButton>)}
            </div>
            <GrpoCharts metrics={(tab==='stream' ? metricSeries : (levelSeries[tab] && levelSeries[tab]?.length ? levelSeries[tab] : ((levelMetrics.data as any)?.metrics || [])))} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
            <div style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8 }}>
              <div style={{ color: '#9ca3af', marginBottom: 6 }}>Dataset Stats</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 8 }}>
                <input value={payload.dataset_path} onChange={(e: any) => setPayload(p => ({ ...p, dataset_path: e.target.value }))} style={{ width: '100%', background: '#0b0f17', color: '#e6edf3', border: '1px solid #233', borderRadius: 6, padding: '6px 8px' }} />
                <button onClick={async () => { try { const s = await api.getGrpoDatasetStats(payload.dataset_path); setDsStats(s); notify('Loaded dataset stats', 'ok') } catch (e: any) { setDsStats({ error: e?.message || String(e) }); notify('Failed to load stats', 'err') } }} style={{ background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px' }}>Load</button>
              </div>
              {!!dsStats && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8, marginTop: 8 }}>
                  <Metric label="Groups" value={String(dsStats.scanned ?? '--')} />
                  <Metric label="Cand μ/min/max" value={`${dsStats?.candidates?.avg ?? '--'}/${dsStats?.candidates?.min ?? '--'}/${dsStats?.candidates?.max ?? '--'}`} />
                  <Metric label="Reward μ" value={String((dsStats?.rewards?.mean ?? 0).toFixed?.(3))} />
                </div>
              )}
              {!!dsStats?.quality && (
                <div style={{ marginTop: 8, color: '#9ca3af' }}>
                  <div style={{ marginBottom: 4 }}>Quality</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
                    <Metric label="Empty prompts" value={String(dsStats.quality.empty_prompts)} />
                    <Metric label="Short prompts" value={String(dsStats.quality.short_prompts)} />
                    <Metric label="Missing text" value={String(dsStats.quality.missing_text)} />
                    <Metric label="Reward outliers" value={String(dsStats.quality.reward_outliers_std3)} />
                  </div>
                  {(() => {
                    const q = dsStats.quality || {}
                    const n = (dsStats.rewards?.count || 0) as number
                    const outlierRate = n ? (q.reward_outliers_std3 || 0) / n : 0
                    const issues: string[] = []
                    if ((q.empty_prompts || 0) > 0) issues.push('Empty prompts present')
                    if ((q.missing_text || 0) > 0) issues.push('Missing candidate text')
                    if (outlierRate > 0.02) issues.push(`Reward outliers >2% (${(outlierRate*100).toFixed(1)}%)`)
                    if ((q.short_prompts || 0) > 0) issues.push('Short prompts detected')
                    if ((q.short_text || 0) > 0) issues.push('Short candidate texts detected')
                    return issues.length ? (
                      <div style={{ marginTop: 8, background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.35)', color: '#fbbf24', padding: 8, borderRadius: 6 }}>
                        <strong>Dataset Warnings:</strong> {issues.join('; ')}
                      </div>
                    ) : null
                  })()}
                </div>
              )}
              {!!dsStats?.mesh && (
                <div style={{ marginTop: 8, color: '#9ca3af' }}>
                  <div style={{ marginBottom: 4 }}>Mesh Tail</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8 }}>
                    <Metric label="Tail groups" value={String(dsStats.mesh.tail_groups)} />
                    <Metric label="Tail cands (total)" value={String(dsStats.mesh.tail_candidates_total)} />
                    <Metric label="Tail cands (avg/group)" value={String((dsStats.mesh.tail_avg_per_group ?? 0).toFixed?.(2))} />
                  </div>
                </div>
              )}
              <pre style={{ marginTop: 8, background: '#0b0f17', color: '#9ca3af', border: '1px solid #233', borderRadius: 6, padding: 8, maxHeight: 160, overflow: 'auto' }}>{dsStats ? JSON.stringify({ path: dsStats.path, top_keywords: dsStats.top_keywords, sample_prompts: dsStats.sample_prompts, quality: dsStats.quality }, null, 2) : 'No stats loaded.'}</pre>
            </div>
            <div style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8 }}>
              <div style={{ color: '#9ca3af', marginBottom: 6 }}>Policy Nudges</div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <button onClick={async () => { try { const res = await api.applyGrpoPolicy({ hours: 24 }); if (res.ok) notify('Policy nudges applied', 'ok'); else notify('Policy apply failed', 'err') } catch (e: any) { notify('Policy apply error', 'err') } }} style={{ background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px' }}>Apply Now</button>
              </div>
              {!!(auto.data?.policy_updates?.length) ? (
                <pre style={{ background: '#0b0f17', color: '#9ca3af', border: '1px solid #233', borderRadius: 6, padding: 8, maxHeight: 160, overflow: 'auto' }}>{JSON.stringify(auto.data?.policy_updates?.slice(-10), null, 2)}</pre>
              ) : (
                <div style={{ color: '#6b7280' }}>No recent updates.</div>
              )}
            </div>
          </div>
          <div style={{ marginTop: 12, background: '#0b0f17', color: '#9ca3af', border: '1px solid #233', borderRadius: 6, padding: 8 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <Metric label="CPU" value={formatCpu(resources.data)} />
              <Metric label="RAM" value={formatRam(resources.data)} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
              <Metric label="Disk Free" value={formatDisk(resources.data)} />
              <Metric label="GPU" value={formatGpu(resources.data)} />
            </div>
          <pre style={{ marginTop: 8, background: '#0b0f17', color: '#9ca3af', border: '1px solid #233', borderRadius: 6, padding: 8, maxHeight: 120, overflow: 'auto' }}>{formatContainers(resources.data)}</pre>
          </div>
        </div>
      </div>
      </div>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ background: '#0b0f17', padding: 10, border: '1px solid #233', borderRadius: 6 }}>
      <div style={{ color: '#93c5fd', fontSize: 12 }}>{label}</div>
      <div style={{ color: '#e5e7eb', fontSize: 18, marginTop: 4 }}>{value}</div>
    </div>
  );
}

function TabButton({ active, children, onClick }: { active?: boolean; children: any; onClick?: () => void }) {
  return (
    <button onClick={onClick} className={`tab ${active ? 'tab-active' : ''}`}>{children}</button>
  )
}

function formatCpu(res?: any): string {
  const cpu = res?.host?.cpu || {};
  const pct = Number.isFinite(cpu.cpu_pct) ? `${cpu.cpu_pct.toFixed(1)}%` : '';
  const load = Number.isFinite(cpu.load1) ? `load1=${cpu.load1}` : (Number.isFinite(cpu.load_avg_1m) ? `load1=${cpu.load_avg_1m}` : '');
  return pct || load || '--';
}

function formatRam(res?: any): string {
  const m = res?.host?.memory || res?.host?.mem || {};
  const total = m.total_gb ?? m.mem_total_gb;
  const free = m.free_gb ?? m.mem_free_gb;
  const pct = m.pct_used ?? m.mem_pct;
  if (Number.isFinite(total) && Number.isFinite(free)) {
    return `${free?.toFixed?.(2)} / ${total?.toFixed?.(2)} GB${Number.isFinite(pct) ? ` (${pct}%)` : ''}`;
  }
  return '--';
}

function formatDisk(res?: any): string {
  const d = res?.host?.disk || {};
  if (Number.isFinite(d.free_gb) && Number.isFinite(d.total_gb)) {
    return `${d.free_gb.toFixed(2)} / ${d.total_gb.toFixed(2)} GB free`;
  }
  return '--';
}

function formatGpu(res?: any): string {
  const g = Array.isArray(res?.host?.gpu) ? res.host.gpu : [];
  if (!g.length) return 'none';
  const first = g[0];
  const used = first.mem_used_mb ?? 0;
  const tot = first.mem_total_mb ?? 0;
  const util = first.util_pct;
  return `${first.name || 'GPU'} ${util != null ? `${util}% util ` : ''}${used}/${tot} MB`;
}

function formatContainers(res?: any): string {
  const list = Array.isArray(res?.containers) ? res.containers : [];
  if (!list.length) return 'No container stats';
  const top = list.slice(0, 5).map((c: any) => `${c.name}: cpu ${c.cpu_pct?.toFixed?.(1) ?? c.cpu_pct}% mem ${c.mem_used_mb?.toFixed?.(1)}/${c.mem_limit_mb?.toFixed?.(1)} MB (${c.mem_pct}%)`).join('\n');
  return top;
}
