import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import Card from '../components/Card';
import StatusPill from '../components/StatusPill';
import { api } from '../api/client';
import { Line, Bar } from 'react-chartjs-2';
import { ensureChartsRegistered } from '../lib/registerCharts';
import styles from './MonitoringPage.module.css';

const MonitoringPage = () => {
  ensureChartsRegistered();
  const [isPaused, setIsPaused] = useState(false);
  const [logs, setLogs] = useState('');
  const [command, setCommand] = useState('');
  const queryClient = useQueryClient();

  // Overview SSE for status + metrics (no polling)
  const [monOverview, setMonOverview] = useState<any | null>(null);
  useEffect(() => {
    try {
      const es = new EventSource('/api/overview/stream-diff');
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          setMonOverview((prev: any) => ({ ...(prev || {}), ...(msg || {}) }));
        } catch {}
      };
      es.onerror = () => {};
      return () => { es.close(); };
    } catch {}
  }, []);

  // SSE stream for logs + actions
  const [liveActions, setLiveActions] = useState<any | null>(null);
  const [showVectorizer, setShowVectorizer] = useState<boolean>(false);
  const [vecStats, setVecStats] = useState<any | null>(null);
  const [vecRate, setVecRate] = useState<{ rows?: number; bytes?: number }>({});
  const [spark, setSpark] = useState<any | null>(null);
  const [imesh, setImesh] = useState<any | null>(null);
  const [embedWorker, setEmbedWorker] = useState<any | null>(null);
  const [vecRowsHist, setVecRowsHist] = useState<number[]>([]);
  const [vecBytesHist, setVecBytesHist] = useState<number[]>([]);
  const [ewRowsHist, setEwRowsHist] = useState<number[]>([]);
  const [ewLast, setEwLast] = useState<{ t: number; out: number } | null>(null);
  const [sparkInHist, setSparkInHist] = useState<number[]>([]);
  const [sparkProcHist, setSparkProcHist] = useState<number[]>([]);
  const [knnDocId, setKnnDocId] = useState('');
  const [knnVector, setKnnVector] = useState('');
  const [knnK, setKnnK] = useState(5);
  const [knnShards, setKnnShards] = useState('');
  const [knnResult, setKnnResult] = useState<any | null>(null);
  const [shardStats, setShardStats] = useState<any | null>(null);
  const [dimsTrend, setDimsTrend] = useState<Record<string, number[]>>({});
  useEffect(() => {
    try {
      const es = new EventSource('/api/monitor/stream');
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (!isPaused && msg?.logs?.logs) setLogs(msg.logs.logs);
          if (msg?.actions) setLiveActions(msg.actions);
        } catch {}
      };
      es.onerror = () => { /* auto-retry by browser */ };
      return () => { es.close(); };
    } catch { /* ignore in non-SSE envs */ }
  }, [isPaused]);

  // Load runtime config to toggle vectorizer panel
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch('/config.json', { cache: 'no-cache' });
        if (resp.ok) {
          const cfg = await resp.json();
          setShowVectorizer(!!cfg?.vectorizerEnabled);
        }
      } catch {}
    })();
  }, []);

  // Persist kNN form defaults in localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem('knn_form');
      if (raw) {
        const v = JSON.parse(raw);
        if (typeof v.doc_id === 'string') setKnnDocId(v.doc_id);
        if (typeof v.k === 'number') setKnnK(v.k);
        if (typeof v.shards === 'string') setKnnShards(v.shards);
        if (typeof v.vector === 'string') setKnnVector(v.vector);
      }
    } catch {}
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem('knn_form', JSON.stringify({ doc_id: knnDocId, k: knnK, shards: knnShards, vector: knnVector }));
    } catch {}
  }, [knnDocId, knnK, knnShards, knnVector]);

  // Vectorizer SSE stream
  useEffect(() => {
    if (!showVectorizer) return;
    try {
      const es = new EventSource('/api/vectorizer/stream');
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          setVecStats(msg?.stats || null);
          setVecRate({ rows: msg?.rate_rows_per_sec || 0, bytes: msg?.rate_bytes_per_sec || 0 });
          // push rolling history (keep last 60)
          if (typeof msg?.rate_rows_per_sec === 'number') {
            setVecRowsHist((h) => [...h.slice(-59), Math.max(0, msg.rate_rows_per_sec)]);
          }
          if (typeof msg?.rate_bytes_per_sec === 'number') {
            setVecBytesHist((h) => [...h.slice(-59), Math.max(0, msg.rate_bytes_per_sec)]);
          }
        } catch {}
      };
      es.onerror = () => {};
      return () => { es.close(); };
    } catch {}
  }, [showVectorizer]);

  // Spark cluster SSE stream
  useEffect(() => {
    try {
      const es = new EventSource('/api/spark/stream');
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          setSpark(msg);
          if (typeof msg?.streaming?.inputRate === 'number') {
            setSparkInHist((h) => [...h.slice(-59), Math.max(0, msg.streaming.inputRate)]);
          }
          if (typeof msg?.streaming?.processingRate === 'number') {
            setSparkProcHist((h) => [...h.slice(-59), Math.max(0, msg.streaming.processingRate)]);
          }
        } catch {}
      };
      es.onerror = () => {};
      return () => { es.close(); };
    } catch {}
  }, []);

  // InferMesh SSE stream
  useEffect(() => {
    try {
      const es = new EventSource('/api/infermesh/stream');
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          setImesh(msg);
        } catch {}
      };
      es.onerror = () => {};
      return () => { es.close(); };
    } catch {}
  }, []);

  // Embed-worker metrics SSE stream
  useEffect(() => {
    try {
      const es = new EventSource('/api/embed-worker/stream');
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          setEmbedWorker(msg);
          const m = msg?.metrics;
          if (m && typeof m.records_out === 'number') {
            const now = Number(msg?.timestamp || Date.now()/1000);
            const last = ewLast;
            if (last && now > last.t) {
              const rate = Math.max(0, (m.records_out - last.out) / (now - last.t));
              setEwRowsHist((h) => [...h.slice(-59), rate]);
            }
            setEwLast({ t: now, out: m.records_out });
          }
        } catch {}
      };
      es.onerror = () => {};
      return () => { es.close(); };
    } catch {}
  }, []);

  const commandMutation = useMutation({
    mutationFn: (cmd: string) => api.sendCommand(cmd),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['logs'] });
      setCommand('');
    }
  });

  const restartMutation = useMutation({
    mutationFn: api.restartAgent,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['status'] });
    }
  });

  // Optional fallback if SSE unavailable
  // (kept minimal to avoid double-polling when SSE works)

  const statusItems = useMemo(() => {
    const s = monOverview?.status;
    if (!s) return [];
    const { agent, ollama, kafka, containers } = s;
    return [
      { label: 'Agent', status: agent },
      { label: 'Ollama', status: ollama },
      { label: 'Kafka', status: kafka },
      { label: 'Containers', status: containers }
    ];
  }, [monOverview]);

  const handleExport = () => {
    const blob = new Blob([logs], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `dspy-agent-logs-${new Date().toISOString().slice(0, 19)}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const latestMetrics = monOverview?.metrics;

  // Chat state
  const [chatInput, setChatInput] = useState('');
  const [chatLog, setChatLog] = useState<{ role: 'user' | 'agent'; text: string }[]>([]);
  const sendChat = async () => {
    const text = chatInput.trim();
    if (!text) return;
    setChatLog((l) => [...l, { role: 'user', text }]);
    setChatInput('');
    try {
      const res = await api.sendChat({ message: text });
      setChatLog((l) => [...l, { role: 'agent', text: res.response }]);
    } catch (e: any) {
      setChatLog((l) => [...l, { role: 'agent', text: `Error: ${e?.message || e}` }]);
    }
  };

  // Live actions stream (recent)
  const { data: actionsData, refetch: refetchActions } = useQuery({
    queryKey: ['actions-analytics', 'live'],
    queryFn: () => api.getActionsAnalytics(50, '1h'),
    refetchInterval: false,
    enabled: false
  });
  const onFeedback = async (action_id: string, decision: 'approve' | 'reject' | 'suggest', comment?: string) => {
    await api.sendActionFeedback({ action_id, decision, comment });
    // SSE should push fresh actions; no explicit refetch required
  };

  // Guardrails
  const { data: guardState, refetch: refetchGuard } = useQuery({ queryKey: ['guardrails'], queryFn: api.getGuardrailsState, refetchInterval: 8000 });
  const toggleGuardrails = async (enabled: boolean) => {
    await api.setGuardrails(enabled);
    refetchGuard();
  };
  const { data: proposedActions, refetch: refetchProposed } = useQuery({ queryKey: ['proposed-actions'], queryFn: api.getProposedActions, refetchInterval: 5000 });

  return (
    <div className={styles.wrapper}>
      <div className={styles.headerRow}>
        <div>
          <h1>Runtime Monitoring</h1>
          <p>Stream logs, inspect metrics, and issue agent commands in real time.</p>
        </div>
        <button className={styles.restartButton} onClick={() => restartMutation.mutate()} disabled={restartMutation.isPending}>
          {restartMutation.isPending ? 'Restarting…' : 'Restart Agent'}
        </button>
      </div>

      <div className={styles.layoutGrid}>
        <Card title="Service Health" dense>
          <div className={styles.statusList}>
            {statusItems.map(({ label, status }) => (
              <div key={label} className={styles.statusRow}>
                <span>{label}</span>
                <StatusPill status={status.status} text={status.details ?? status.status} />
              </div>
            ))}
            {!monOverview && <span className={styles.muted}>Waiting for status…</span>}
          </div>
        </Card>

        {showVectorizer && (
          <Card title="Vectorizer" subtitle="Kafka → vectors throughput" dense>
            <div className={styles.metricsGrid}>
              <Metric label="Rows/sec" value={typeof vecRate.rows === 'number' ? vecRate.rows.toFixed(1) : '--'} />
              <Metric label="Bytes/sec" value={typeof vecRate.bytes === 'number' ? `${(vecRate.bytes/1024).toFixed(1)} KB/s` : '--'} />
              <Metric label="Files" value={vecStats ? String(vecStats.files) : '--'} />
              <Metric label="Total Rows (est)" value={vecStats ? String(vecStats.rows_est) : '--'} />
              <Metric label="Output" value={vecStats?.path || '/workspace/vectorized/embeddings'} />
              <Metric label="Updated" value={vecStats?.latest_ts ? new Date(vecStats.latest_ts * 1000).toLocaleTimeString() : '--'} />
            </div>
          </Card>
        )}

        <Card title="Spark Health" subtitle="Cluster + job status" dense>
          <div className={styles.metricsGrid}>
            <Metric label="CPU Utilization" value={spark ? `${spark.cluster_metrics.cpu_utilization.toFixed(1)}%` : '--'} />
            <Metric label="Workers" value={spark ? String(spark.master.workers) : '--'} />
            <Metric label="Cores" value={spark ? `${spark.cluster_metrics.used_cores}/${spark.cluster_metrics.total_cores}` : '--'} />
            <Metric label="Apps Running" value={spark ? String(spark.master.applications_running) : '--'} />
            <Metric label="Memory" value={spark ? `${spark.cluster_metrics.used_memory} / ${spark.cluster_metrics.total_memory}` : '--'} />
            <Metric label="Updated" value={spark ? new Date(spark.timestamp * 1000).toLocaleTimeString() : '--'} />
          </div>
        </Card>

        <Card title="Spark Streaming" subtitle="Batch stats (REST)" dense>
          <div className={styles.metricsGrid}>
            <Metric label="Input Rate" value={spark?.streaming?.inputRate != null ? `${spark.streaming.inputRate.toFixed?.(1) ?? spark.streaming.inputRate} rec/s` : '--'} />
            <Metric label="Processing Rate" value={spark?.streaming?.processingRate != null ? `${spark.streaming.processingRate.toFixed?.(1) ?? spark.streaming.processingRate} rec/s` : '--'} />
            <Metric label="Avg Input/Batch" value={spark?.streaming?.avgInputPerBatch != null ? String(spark.streaming.avgInputPerBatch) : '--'} />
            <Metric label="Proc Time (ms)" value={spark?.streaming?.avgProcessingTimeMs != null ? String(spark.streaming.avgProcessingTimeMs) : '--'} />
            <Metric label="Sched Delay (ms)" value={spark?.streaming?.avgSchedulingDelayMs != null ? String(spark.streaming.avgSchedulingDelayMs) : '--'} />
            <Metric label="Total Delay (ms)" value={spark?.streaming?.avgTotalDelayMs != null ? String(spark.streaming.avgTotalDelayMs) : '--'} />
          </div>
        </Card>

        <Card title="Spark Rates" subtitle="Rolling input/processing rates" dense>
          <Line
            data={{
              labels: sparkInHist.map((_, i) => `${i}`),
              datasets: [
                { label: 'inputRate', data: sparkInHist, borderColor: '#f472b6', backgroundColor: 'rgba(244,114,182,0.15)', tension: 0.35, fill: true },
                { label: 'processingRate', data: sparkProcHist, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.15)', tension: 0.35, fill: true }
              ]
            }}
            options={{ responsive: true, plugins: { legend: { position: 'bottom' } }, scales: { x: { display: false }, y: { beginAtZero: true } } }}
          />
        </Card>

        <Card title="InferMesh" subtitle="Embedding service health & throughput" dense>
          <div className={styles.metricsGrid}>
            <Metric label="Status" value={imesh ? imesh.infermesh.status : '--'} />
            <Metric label="RTT" value={imesh && typeof imesh.infermesh.rtt_ms === 'number' ? `${imesh.infermesh.rtt_ms} ms` : '--'} />
            <Metric label="Rows/sec" value={imesh && typeof imesh.rows_per_sec === 'number' ? imesh.rows_per_sec.toFixed(1) : '--'} />
            <Metric label="Rows(est)" value={imesh ? String(imesh.rows_est) : '--'} />
            <Metric label="Updated" value={imesh ? new Date(imesh.timestamp * 1000).toLocaleTimeString() : '--'} />
          </div>
        </Card>

        <Card title="Embed Worker" subtitle="Worker metrics (batches, throughput, errors)" dense>
          <div className={styles.metricsGrid}>
            <Metric label="Batches" value={embedWorker?.metrics?.batches != null ? String(embedWorker.metrics.batches) : '--'} />
            <Metric label="Records In" value={embedWorker?.metrics?.records_in != null ? String(embedWorker.metrics.records_in) : '--'} />
            <Metric label="Records Out" value={embedWorker?.metrics?.records_out != null ? String(embedWorker.metrics.records_out) : '--'} />
            <Metric label="InferMesh RTT" value={typeof embedWorker?.metrics?.last_infermesh_rtt_ms === 'number' ? `${embedWorker.metrics.last_infermesh_rtt_ms} ms` : '--'} />
            <Metric label="Failures" value={embedWorker?.metrics?.infermesh_failures != null ? String(embedWorker.metrics.infermesh_failures) : '--'} />
            <Metric label="DLQ" value={embedWorker?.metrics?.dlq_records != null ? String(embedWorker.metrics.dlq_records) : '0'} />
            <Metric label="Started" value={embedWorker?.metrics?.started_ts ? new Date(embedWorker.metrics.started_ts * 1000).toLocaleTimeString() : '--'} />
            <Metric label="Last Flush" value={embedWorker?.metrics?.last_flush_ts ? new Date(embedWorker.metrics.last_flush_ts * 1000).toLocaleTimeString() : '--'} />
            <Metric label="Updated" value={embedWorker ? new Date(embedWorker.timestamp * 1000).toLocaleTimeString() : '--'} />
          </div>
        </Card>

        <Card title="Throughput (Vectorizer)" subtitle="Rows/sec (rolling)" dense>
          <Line
            data={{
              labels: vecRowsHist.map((_, i) => `${i}`),
              datasets: [{ label: 'rows/sec', data: vecRowsHist, borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.15)', tension: 0.35, fill: true }]
            }}
            options={{ responsive: true, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { beginAtZero: true } } }}
          />
        </Card>

        <Card title="Throughput (Embed)" subtitle="Records/sec (rolling)" dense>
          <Line
            data={{
              labels: ewRowsHist.map((_, i) => `${i}`),
              datasets: [{ label: 'records/sec', data: ewRowsHist, borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.15)', tension: 0.35, fill: true }]
            }}
            options={{ responsive: true, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { beginAtZero: true } } }}
          />
        </Card>

        <Card title="kNN Query" subtitle="Search neighbors by doc_id or vector" dense>
          <form
            className={styles.commandForm}
            onSubmit={async (e) => {
              e.preventDefault();
              try {
                const payload: any = { k: knnK };
                if (knnDocId.trim()) payload.doc_id = knnDocId.trim();
                if (knnVector.trim()) payload.vector = JSON.parse(knnVector);
                if (knnShards.trim()) payload.shards = knnShards.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isFinite(n));
                const resp = await fetch('/api/knn/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                setKnnResult(await resp.json());
              } catch (e: any) {
                setKnnResult({ error: e?.message || String(e) });
              }
            }}
          >
            <input className={styles.commandInput} placeholder="doc_id" value={knnDocId} onChange={(e) => setKnnDocId(e.target.value)} />
            <input className={styles.commandInput} placeholder="k (e.g., 5)" type="number" min={1} value={knnK} onChange={(e) => setKnnK(parseInt(e.target.value || '5', 10))} />
            <input className={styles.commandInput} placeholder="shards (e.g., 0,1,2)" value={knnShards} onChange={(e) => setKnnShards(e.target.value)} />
            <textarea className={styles.commandInput} placeholder='vector JSON (e.g., [0.1,0.2,...])' value={knnVector} onChange={(e) => setKnnVector(e.target.value)} />
            <button className={styles.primaryButton} type="submit">Query</button>
            {knnResult && (
              <button className={styles.secondaryButton} type="button" onClick={() => {
                const blob = new Blob([JSON.stringify(knnResult, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = `knn-${Date.now()}.json`; a.click(); URL.revokeObjectURL(url);
              }}>Download</button>
            )}
            <button className={styles.secondaryButton} type="button" onClick={() => {
              // Auto-fill from top shard sample (if any)
              if (Array.isArray(shardStats?.shards) && shardStats.shards.length > 0) {
                const top = [...shardStats.shards].sort((a: any, b: any) => (b.count||0)-(a.count||0))[0];
                const sample = (top?.samples || [])[0];
                if (sample) setKnnDocId(sample);
              }
            }}>Use sample from top shard</button>
          </form>
          <div className={`${styles.logViewer} scrollbar`} style={{ maxHeight: 200 }}>
            {Array.isArray(knnResult?.neighbors) && knnResult.neighbors.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <strong>Neighbors</strong>
                <div className={styles.statusList}>
                  {knnResult.neighbors.slice(0, 10).map((n: any, i: number) => (
                    <div className={styles.statusRow} key={i}>
                      <div>
                        <div>{n.doc_id}</div>
                        <div className={styles.muted}>score {typeof n.score === 'number' ? n.score.toFixed(4) : String(n.score)}</div>
                      </div>
                      <div className={styles.actionBar}>
                        <button className={styles.secondaryButton} onClick={() => setKnnDocId(n.doc_id)}>Use as query</button>
                        <button className={styles.secondaryButton} onClick={async () => {
                          const payload: any = { doc_id: n.doc_id, k: knnK };
                          if (knnShards.trim()) payload.shards = knnShards.split(',').map((s) => parseInt(s.trim(), 10)).filter((x) => Number.isFinite(x));
                          const resp = await fetch('/api/knn/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                          setKnnResult(await resp.json());
                        }}>Run kNN on this</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <pre>{JSON.stringify(knnResult || { info: 'Enter doc_id or vector to query.' }, null, 2)}</pre>
          </div>
          <div className={styles.actionBar}>
            <button className={styles.secondaryButton} onClick={async () => {
              const doc_id = knnDocId.trim(); if (!doc_id) { alert('Set doc_id first'); return; }
              const signature_name = prompt('Signature name for this example?'); if (!signature_name) return;
              const rewardStr = prompt('Reward value (e.g., 1.0 for success, 0.0 default)?', '1.0') || '0.0';
              const env = prompt('Environment (development/testing/staging/production/local)?', 'development') || 'development';
              let reward = parseFloat(rewardStr); if (!Number.isFinite(reward)) reward = 0.0;
              try {
                await api.recordActionResult({ signature_name, verifier_scores: {}, reward, environment: env, doc_id, action_type: 'VERIFICATION' });
                alert('Persisted example for finetuning');
              } catch (e: any) {
                alert(`Error: ${e?.message || e}`);
              }
            }}>Persist as example</button>
          </div>
        </Card>

        <Card title="Shard Stats" subtitle="Coverage and sizes" dense>
          <div className={styles.actionBar}>
            <button className={styles.secondaryButton} onClick={async () => {
              try {
                const resp = await fetch('/api/knn/shards');
                const data = await resp.json();
                setShardStats(data);
                const dims = (data?.dims_by_model ?? {}) as Record<string, number>;
                setDimsTrend((prev) => {
                  const next: Record<string, number[]> = { ...prev };
                  Object.keys(dims).forEach((m) => {
                    const arr = next[m] || [];
                    next[m] = [...arr.slice(-59), Number(dims[m] || 0)];
                  });
                  return next;
                });
              } catch (e: any) {
                setShardStats({ error: e?.message || String(e) });
              }
            }}>Refresh</button>
            {Array.isArray(shardStats?.shards) && (
              <span className={styles.muted}>Total: {shardStats.total_ids}, Non-empty: {shardStats.nonempty}</span>
            )}
          </div>
          <div className={`${styles.logViewer} scrollbar`} style={{ maxHeight: 220 }}>
            <pre>{JSON.stringify(shardStats || { info: 'Click Refresh to load shard sizes.' }, null, 2)}</pre>
          </div>
        </Card>

        {Array.isArray(shardStats?.shards) && shardStats.shards.length > 0 && (
          <Card title="Shard Sizes" subtitle="Top 8 shards by count" dense>
            <Bar
              data={() => {
                const sorted = [...(shardStats?.shards || [])].sort((a: any, b: any) => (b.count || 0) - (a.count || 0)).slice(0, 8);
                return {
                  labels: sorted.map((s: any) => `#${s.id}`),
                  datasets: [{ label: 'count', data: sorted.map((s: any) => s.count || 0), backgroundColor: 'rgba(99,102,241,0.45)' }]
                };
              }}
              options={{ responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }}
            />
          </Card>
        )}

        {Object.keys(dimsTrend).length > 0 && (
          <Card title="Dims by Model" subtitle="Smoothed trend (recent samples)" dense>
            <Line
              data={{
                labels: Array.from({ length: Math.max(1, Math.max(...Object.values(dimsTrend).map((a) => a.length))) }, (_, i) => `${i}`),
                datasets: Object.keys(dimsTrend).map((m, idx) => ({
                  label: m,
                  data: dimsTrend[m],
                  borderColor: ['#60a5fa','#34d399','#f59e0b','#f472b6','#22c55e','#ef4444'][idx % 6],
                  backgroundColor: 'transparent',
                  tension: 0.25,
                  pointRadius: 0
                }))
              }}
              options={{ responsive: true, plugins: { legend: { position: 'bottom' } }, scales: { x: { display: false }, y: { beginAtZero: true } } }}
            />
          </Card>
        )}

        <Card title="Embed DLQ" subtitle="Failed embeddings (download)" dense>
          <div className={styles.actionBar}>
            <button className={styles.secondaryButton} onClick={async () => {
              const limit = prompt('How many DLQ items to fetch? (e.g., 500)', '500') || '500';
              try {
                const resp = await fetch(`/api/embed-worker/dlq?limit=${encodeURIComponent(limit)}`);
                const data = await resp.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = `embed-dlq-${Date.now()}.json`; a.click(); URL.revokeObjectURL(url);
              } catch {}
            }}>Download DLQ JSON</button>
            <span className={styles.muted}>Total DLQ (session): {embedWorker?.metrics?.dlq_records ?? 0}</span>
          </div>
        </Card>

        <Card title="Performance Snapshot" subtitle="Rolling averages from /api/metrics" dense>
          <div className={styles.metricsGrid}>
            <Metric label="Response Time" value={latestMetrics ? `${latestMetrics.response_time.toFixed(2)}s` : '--'} />
            <Metric label="Containers" value={latestMetrics ? latestMetrics.containers.toString() : '--'} />
            <Metric label="Memory" value={latestMetrics?.memory_usage ?? '--'} />
            <Metric label="Updated" value={latestMetrics ? new Date(latestMetrics.timestamp * 1000).toLocaleTimeString() : '--'} />
          </div>
        </Card>

        <Card
          title="Live Agent Logs"
          subtitle="Tail of docker-compose logs"
          actions={
            <div className={styles.actionBar}>
              <button className={styles.secondaryButton} onClick={() => setLogs('')}>Clear</button>
              <button className={styles.secondaryButton} onClick={() => setIsPaused(!isPaused)}>
                {isPaused ? 'Resume' : 'Pause'}
              </button>
              <button className={styles.secondaryButton} onClick={handleExport}>Export</button>
            </div>
          }
        >
          <pre className={`${styles.logViewer} scrollbar`}>{logs || '[No logs received]'}</pre>
        </Card>

        <Card title="Agent Controls" subtitle="Send commands directly into the agent container (tail args only; prefix is auto-added)" dense>
          <form
            className={styles.commandForm}
            onSubmit={(event) => {
              event.preventDefault();
              if (!command.trim()) return;
              commandMutation.mutate(command.trim());
            }}
          >
            <input
              className={styles.commandInput}
              placeholder="rl train --steps 200"
              value={command}
              onChange={(event) => setCommand(event.target.value)}
            />
            <button className={styles.primaryButton} type="submit" disabled={commandMutation.isPending}>
              {commandMutation.isPending ? 'Running…' : 'Execute'}
            </button>
          </form>
          {commandMutation.data && (
            <div className={styles.commandResult}>
              <strong>{commandMutation.data.success ? 'Success' : 'Error'}</strong>
              <pre className={styles.commandOutput}>{(commandMutation.data.output || commandMutation.data.error || '').trim() || '[No output]'}</pre>
            </div>
          )}
        </Card>

        <Card title="Guardrails" subtitle="Require approval before running CLI commands" dense>
          <div className={styles.statusList}>
            <div className={styles.statusRow}>
              <span>Enabled</span>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <input type="checkbox" checked={!!guardState?.enabled} onChange={(e) => toggleGuardrails(e.target.checked)} />
                <span className={styles.muted}>{guardState?.enabled ? 'On' : 'Off'}</span>
              </label>
            </div>
          </div>
          <div className={styles.statusList}>
            {(guardState?.pending ?? []).map((p) => (
              <div key={p.id} className={styles.statusRow}>
                <div>
                  <strong>{p.command}</strong>
                  <div className={styles.muted}>ws={p.workspace || '/app/test_project'} · {new Date(p.ts * 1000).toLocaleTimeString()}</div>
                </div>
                <div className={styles.actionBar}>
                  <button className={styles.secondaryButton} onClick={async () => { await api.approvePending(p.id); refetchGuard(); }}>Approve</button>
                  <button className={styles.secondaryButton} onClick={async () => { await api.rejectPending(p.id); refetchGuard(); }}>Reject</button>
                </div>
              </div>
            ))}
            {(guardState?.pending ?? []).length === 0 && <div className={styles.muted}>No pending commands</div>}
          </div>
        </Card>

        <Card title="Proposed Actions" subtitle="Agent actions requiring approval (e.g., patch apply)" dense>
          <div className={styles.statusList}>
            {(proposedActions?.pending ?? []).map((a) => (
              <div key={a.id} className={styles.statusRow}>
                <div>
                  <strong>{a.type}</strong>
                  <div className={styles.muted}>{new Date(a.ts * 1000).toLocaleTimeString()}</div>
                  {a.payload?.summary && (
                    <div className={styles.muted}>
                      files={a.payload.summary.files} +{a.payload.summary.added_lines} -{a.payload.summary.removed_lines}
                    </div>
                  )}
                  {a.payload?.patch_preview && (
                    <pre className={styles.logViewer} style={{ maxHeight: 120, marginTop: 6 }}>{String(a.payload.patch_preview).slice(0, 2000)}</pre>
                  )}
                </div>
                <div className={styles.actionBar}>
                  <button className={styles.secondaryButton} onClick={async () => { await api.approveAction(a.id); refetchProposed(); }}>Approve</button>
                  <button className={styles.secondaryButton} onClick={async () => { const c = prompt('Reason'); await api.rejectAction(a.id, c || undefined); refetchProposed(); }}>Reject</button>
                </div>
              </div>
            ))}
            {(proposedActions?.pending ?? []).length === 0 && <div className={styles.muted}>No proposed actions</div>}
          </div>
        </Card>

        <Card title="Chat" subtitle="Talk to the agent" dense>
          <div className={styles.commandForm}>
            <input
              className={styles.commandInput}
              placeholder="Ask the agent…"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') sendChat(); }}
            />
            <button className={styles.primaryButton} onClick={sendChat}>Send</button>
          </div>
          <div className={`${styles.logViewer} scrollbar`} style={{ height: 180 }}>
            {chatLog.length === 0 ? (
              <div className={styles.muted}>No messages yet</div>
            ) : (
              chatLog.map((m, i) => (
                <div key={i} style={{ marginBottom: 8 }}>
                  <strong style={{ color: m.role === 'user' ? '#93c5fd' : '#fbbf24' }}>{m.role === 'user' ? 'You' : 'Agent'}:</strong>
                  <div>{m.text}</div>
                </div>
              ))
            )}
          </div>
        </Card>

        <Card title="Live Actions" subtitle="Recent agent actions with guardrails" dense>
          <div className={styles.statusList}>
            {(actionsData?.recent ?? []).map((a) => (
              <div key={a.id} className={styles.statusRow}>
                <div>
                  <strong>{a.type}</strong>
                  <div className={styles.muted}>
                    {new Date((a.timestamp || 0) * 1000).toLocaleTimeString()} · reward {a.reward?.toFixed(2) ?? '0.00'} · {a.execution_time?.toFixed(2) ?? '0.00'}s
                  </div>
                </div>
                <div className={styles.actionBar}>
                  <button className={styles.secondaryButton} onClick={() => onFeedback(a.id, 'approve')}>Approve</button>
                  <button className={styles.secondaryButton} onClick={() => onFeedback(a.id, 'reject')}>Reject</button>
                  <button className={styles.secondaryButton} onClick={() => {
                    const c = prompt('Suggest alternative');
                    if (c) onFeedback(a.id, 'suggest', c);
                  }}>Suggest</button>
                </div>
              </div>
            ))}
            {(actionsData?.recent ?? []).length === 0 && <div className={styles.muted}>No recent actions</div>}
          </div>
        </Card>

        <Card title="Presets" subtitle="One-click CLI flows" dense>
          <div className={styles.metricsGrid}>
            <PresetButton label="RL Train (200)" onClick={() => commandMutation.mutate('rl train --steps 200')} />
            <PresetButton label="RL Neural (500)" onClick={() => commandMutation.mutate('rl neural --steps 500')} />
            <PresetButton label="RL PPO (10k)" onClick={() => commandMutation.mutate('rl ppo --total-steps 10000')} />
            <PresetButton label="GEPA Orchestrator" onClick={() => commandMutation.mutate('gepa_orchestrator --dataset-dir /app/test_project/.dspy_data --auto light')} />
            <PresetButton label="GEPA Codegen" onClick={() => commandMutation.mutate('gepa_codegen --dataset-dir /app/test_project/.dspy_data --auto light')} />
            <PresetButton label="Live Training" onClick={() => commandMutation.mutate('live')} />
          </div>
        </Card>
      </div>
    </div>
  );
};

const Metric = ({ label, value }: { label: string; value: string }) => (
  <div className={styles.metricCard}>
    <span className={styles.metricLabel}>{label}</span>
    <strong className={styles.metricValue}>{value}</strong>
  </div>
);

export default MonitoringPage;

const PresetButton = ({ label, onClick }: { label: string; onClick: () => void }) => (
  <button className={styles.secondaryButton} onClick={onClick} style={{ justifySelf: 'flex-start' }}>
    {label}
  </button>
);
