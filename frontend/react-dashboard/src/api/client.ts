import type {
  StatusResponse,
  LogsResponse,
  MetricsResponse,
  ContainersResponse,
  SignaturesResponse,
  VerifiersResponse,
  LearningMetricsResponse,
  PerformanceHistoryResponse,
  KafkaTopicsResponse,
  SparkWorkersResponse,
  RlMetricsResponse,
  SystemTopologyResponse,
  StreamMetricsResponse,
  CommandResponse,
  ChatRequest,
  ChatResponse,
  UpdateConfigRequest,
  UpdateConfigResponse,
  OptimizeSignatureRequest,
  OptimizeSignatureResponse,
  OverviewResponse,
  GraphSnapshotsResponse,
  GraphDiffResponse,
  GraphMctsTopResponse,
  GraphPatternsResponse,
  GraphMemoryReportResponse,
  WorkflowDefinition,
  WorkflowListResponse,
  WorkflowSummary,
  WorkflowRunListResponse,
  WorkflowRunRecord,
  RunnerHardwareSnapshot
} from './types';

const computeDefaultBaseUrl = () => {
  if (typeof window === 'undefined') {
    return 'http://localhost:8083'; // Rust metrics backend
  }

  const { hostname, protocol, port } = window.location;
  // Connect to RedDB backend server
  if (port === '5173' || port === '4173' || port === '3000' || port === '5175' || port === '5176') {
    return `${protocol}//${hostname}:8083`; // Rust metrics backend
  }
  if (!port && hostname === 'localhost') {
    return `${protocol}//${hostname}:8083`; // Rust metrics backend
  }
  return `${protocol}//${hostname}${port ? `:${port}` : ''}`;
};

let RUNTIME_CFG: { apiBaseUrl?: string; wsBaseUrl?: string } | null = null;
let RUNTIME_CFG_LOADED = false;

async function loadRuntimeConfig(): Promise<void> {
  if (RUNTIME_CFG_LOADED) return;
  try {
    const resp = await fetch('/config.json', { headers: { 'Cache-Control': 'no-cache' } });
    if (resp.ok) {
      RUNTIME_CFG = (await resp.json()) as any;
    }
  } catch {}
  RUNTIME_CFG_LOADED = true;
}

async function getApiBaseUrl(): Promise<string> {
  if (!RUNTIME_CFG_LOADED) await loadRuntimeConfig();
  const envBase = (import.meta as any)?.env?.VITE_API_BASE_URL as string | undefined;
  return RUNTIME_CFG?.apiBaseUrl || envBase || computeDefaultBaseUrl();
}

async function apiRequest<T>(path: string, init?: RequestInit, timeoutMs = 8000): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), Math.max(1000, timeoutMs));
  const base = await getApiBaseUrl();
  const response = await fetch(`${base}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    signal: ctrl.signal,
    ...init
  });
  clearTimeout(t);

  if (!response.ok) {
    const message = await response.text().catch(() => response.statusText);
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

function adminHeaders(): HeadersInit {
  try {
    const k = typeof localStorage !== 'undefined' ? (localStorage.getItem('ADMIN_KEY') || '') : ''
    return k ? { 'X-Admin-Key': k } : {}
  } catch {
    return {}
  }
}

export const api = {
  getStatus: async () => normalizeStatus(await apiRequest<any>('/api/status').catch(() => ({}))),
  getLogs: () => apiRequest<LogsResponse>('/api/logs'),
  getMetrics: async () => normalizeMetrics(await apiRequest<any>('/api/metrics').catch(() => ({}))),
  getContainers: () => apiRequest<ContainersResponse>('/api/containers'),
  sendCommand: (command: string, opts?: { workspace?: string; logs?: string }) =>
    apiRequest<CommandResponse>('/api/command', {
      method: 'POST',
      body: JSON.stringify({ command, ...(opts?.workspace ? { workspace: opts.workspace } : {}), ...(opts?.logs ? { logs: opts.logs } : {}) })
    }),
  restartAgent: () =>
    apiRequest<CommandResponse>('/api/restart', {
      method: 'POST'
    }),
  sendChat: (payload: ChatRequest) =>
    apiRequest<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify(payload)
    }),
  getSignatures: async () => normalizeSignatures(await apiRequest<any>('/api/signatures').catch(() => ({}))),
  createSignature: async (payload: { name: string; type?: string; description?: string; tools?: string[]; active?: boolean }) => {
    const res = await apiRequest<any>('/api/signatures', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
    return (res?.updated || res) as import('./types').SignatureSummary;
  },
  getSignatureDetail: (name: string) =>
    apiRequest<import('./types').SignatureDetailResponse>(`/api/signatures/${encodeURIComponent(name)}`),
  getSignatureSchema: (name: string) =>
    apiRequest<{ name: string; inputs: { name: string; desc?: string; default?: any }[]; outputs: { name: string; desc?: string; default?: any }[] }>(`/api/signatures/${encodeURIComponent(name)}/schema`),
  getSignatureAnalytics: (name: string, timeframe?: '1h'|'24h'|'7d'|'30d', env?: string, verifier?: string) =>
    apiRequest<{ signature: string; metrics?: any; related_verifiers: { name: string; avg_score: number; count: number }[]; reward_summary: { avg: number; min: number; max: number; count: number; hist: { bins: number[]; counts: number[] } }; context_keywords: Record<string, number>; actions_sample: any[]; top_embeddings?: { doc_id: string; avg_reward: number; count: number; last_ts: number; model?: string; dim?: number }[] }>(`/api/signatures/${encodeURIComponent(name)}/analytics?${timeframe ? `timeframe=${encodeURIComponent(timeframe)}` : ''}${env ? `&env=${encodeURIComponent(env)}` : ''}${verifier ? `&verifier=${encodeURIComponent(verifier)}` : ''}`),
  getSignatureFeatureAnalysis: (name: string, timeframe?: '1h'|'24h'|'7d'|'30d', env?: string, limit?: number) =>
    apiRequest<{ signature: string; n_dims: number; direction: number[]; top_positive: { idx: number; weight: number }[]; top_negative: { idx: number; weight: number }[]; explanations: string[] }>(`/api/signature/feature-analysis?name=${encodeURIComponent(name)}${timeframe ? `&timeframe=${encodeURIComponent(timeframe)}` : ''}${env ? `&env=${encodeURIComponent(env)}` : ''}${Number.isFinite(limit as any) ? `&limit=${limit}` : ''}`),
  updateSignature: (payload: import('./types').UpdateSignatureRequest) =>
    apiRequest<import('./types').UpdateSignatureResponse>(`/api/signatures/${encodeURIComponent(payload.name)}`, {
      method: 'PUT',
      body: JSON.stringify(payload)
    }),
  recordActionResult: (payload: { signature_name: string; verifier_scores: Record<string, number>; reward: number; environment?: string; execution_time?: number; query?: string; doc_id?: string; action_type?: string }) =>
    apiRequest<{ success: boolean; id?: string; timestamp?: number; error?: string }>('/api/action/record-result', {
      method: 'POST',
      body: JSON.stringify(payload)
    }),
  getSignatureOptHistory: (name: string) =>
    apiRequest<{ history: any[]; metrics: any; timestamp: number }>(`/api/signature/optimization-history?name=${encodeURIComponent(name)}`),
  getSignatureGraph: (timeframe?: '1h'|'24h'|'7d'|'30d', env?: string) =>
    apiRequest<{ nodes: { id: string; type: 'signature'|'verifier' }[]; edges: { source: string; target: string; avg: number; count: number }[] }>(`/api/signature/graph${timeframe || env ? `?${[timeframe?`timeframe=${encodeURIComponent(timeframe)}`:'', env?`env=${encodeURIComponent(env)}`:''].filter(Boolean).join('&')}` : ''}`),
  deleteSignature: (name: string) =>
    apiRequest<{ success: boolean }>(`/api/signatures/${encodeURIComponent(name)}`, {
      method: 'DELETE'
    }),
  
  getVerifiers: async () => normalizeVerifiers(await apiRequest<any>('/api/verifiers').catch(() => ({}))),
  createVerifier: async (payload: { name: string; description?: string; tool?: string; status?: string }) => {
    const res = await apiRequest<any>('/api/verifiers', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
    return (res?.updated || res) as import('./types').VerifierSummary;
  },
  updateVerifier: (payload: import('./types').UpdateVerifierRequest) =>
    apiRequest<import('./types').UpdateVerifierResponse>(`/api/verifiers/${encodeURIComponent(payload.name)}`, {
      method: 'PUT',
      body: JSON.stringify(payload)
    }),
  deleteVerifier: (name: string) =>
    apiRequest<{ success: boolean }>(`/api/verifiers/${encodeURIComponent(name)}`, {
      method: 'DELETE'
    }),
  getLearningMetrics: async () => normalizeLearningMetrics(await apiRequest<any>('/api/learning-metrics').catch(() => ({}))),
  optimizeSignature: (payload: OptimizeSignatureRequest) =>
    apiRequest<OptimizeSignatureResponse>('/api/signature/optimize', {
      method: 'POST',
      body: JSON.stringify(payload)
    }),
  getPerformanceHistory: async (timeframe: string) =>
    normalizePerformanceHistory(await apiRequest<any>(`/api/performance-history?timeframe=${encodeURIComponent(timeframe)}`).catch(() => ({})), timeframe),
  getKafkaTopics: () => apiRequest<KafkaTopicsResponse>('/api/kafka-topics'),
  getSparkWorkers: async () => normalizeSparkWorkers(await apiRequest<any>('/api/spark-workers').catch(() => ({}))),
  // Normalize RL metrics responses coming from different backends
  getRlMetrics: async () => {
    const raw = await apiRequest<any>('/api/rl-metrics').catch(() => ({}));
    return normalizeRlMetrics(raw);
  },
  getSystemTopology: () => apiRequest<SystemTopologyResponse>('/api/system-topology'),
  getSystemResources: async () => {
    try {
      return await apiRequest<{ host: { disk: { path: string; total_gb: number; used_gb: number; free_gb: number; pct_used: number }; gpu: { name: string; mem_used_mb: number; mem_total_mb: number; util_pct: number }[]; threshold_free_gb: number; ok: boolean; timestamp: number }; containers: { name: string; cpu_pct: number; mem_used_mb: number; mem_limit_mb: number; mem_pct: number; net_io?: string; block_io?: string; pids?: any }[] }>('/api/system/resources');
    } catch {
      return { host: { disk: { path: '', total_gb: 0, used_gb: 0, free_gb: 0, pct_used: 0 }, gpu: [], threshold_free_gb: 0, ok: true, timestamp: Date.now()/1000 }, containers: [] } as any;
    }
  },
  getSparkApps: async (namespace?: string) => {
    const qs = namespace ? `?namespace=${encodeURIComponent(namespace)}` : ''
    try { return await apiRequest<any>(`/api/spark/apps${qs}`) } catch { return { sparkapplications: {}, scheduled: {}, timestamp: Date.now()/1000 } }
  },
  getSparkAppList: async (namespace?: string) => {
    const qs = namespace ? `?namespace=${encodeURIComponent(namespace)}` : ''
    try { return await apiRequest<{ items: { kind: string; name: string; namespace: string; state?: string; submissionTime?: string }[]; scheduled: { kind: string; name: string; namespace: string; schedule?: string; lastRun?: string }[] }>(`/api/spark/app-list${qs}`) } catch { return { items: [], scheduled: [] } as any }
  },
  getSparkAppLogs: async (name: string, namespace?: string, tail?: number) => {
    const params = new URLSearchParams()
    params.set('name', name)
    if (namespace) params.set('namespace', namespace)
    if (tail) params.set('tail', String(tail))
    try { return await apiRequest<{ logs: Record<string,string> }>(`/api/spark/app-logs?${params.toString()}`) } catch { return { logs: {} } as any }
  },
  getIngestPending: async () => {
    try { return await apiRequest<{ pending: { source_file: string; rows: number; status: string; dir: string }[]; timestamp: number }>(`/api/ingest/pending-files`) } catch { return { pending: [], timestamp: Date.now()/1000 } }
  },
  approveIngest: (pattern: string) => apiRequest<{ ok: boolean }>(`/api/ingest/approve`, { method: 'POST', body: JSON.stringify({ pattern }) }),
  rejectIngest: (pattern: string) => apiRequest<{ ok: boolean }>(`/api/ingest/reject`, { method: 'POST', body: JSON.stringify({ pattern }) }),
  triggerToolTrain: (payload: { trainer?: 'tiny'|'hf'; args?: { model?: string; epochs?: number; batch_size?: number; max_len?: number; lr?: number; manifest?: string } }) =>
    apiRequest<{ ok: boolean }>(`/api/train/tool`, { method: 'POST', body: JSON.stringify(payload) }),
  triggerCodeLogTrain: (payload: { trainer?: 'tiny'|'hf'; args?: { model?: string; epochs?: number; batch?: number; max_code?: number; max_log?: number; lr?: number; dataset_dir?: string } }) =>
    apiRequest<{ ok: boolean }>(`/api/train/code-log`, { method: 'POST', body: JSON.stringify(payload) }),
  evalCodeLog: (payload: { code: string; model?: string; max_new_tokens?: number }) =>
    apiRequest<{ text: string }>(`/api/eval/code-log`, { method: 'POST', body: JSON.stringify(payload) }),
  evalCodeLogScore: (payload: { code: string; topic?: string; limit?: number; since?: number; until?: number; max_new_tokens?: number }) =>
    apiRequest<{ generated: string; best?: { score: number; bleu1: number; rougeL: number; log: any }; count: number }>(`/api/eval/code-log/score`, { method: 'POST', body: JSON.stringify(payload) }),
  getModelsInfo: () => apiRequest<{ code_log: { model_dir: string; size_bytes: number; updated_at?: number }; grpo: { manifest: string; manifest_mtime?: number; model_dir?: string }; timestamp: number }>(`/api/models`),
  logEvent: (topic: 'ui.action'|'backend.api_call'|'agent.action'|'ingest.decision'|'ingest.file'|'training.trigger'|'training.result'|'spark.app'|'spark.log', event: any, meta?: any) =>
    apiRequest<{ ok: boolean }>(`/api/events`, { method: 'POST', body: JSON.stringify({ topic, event, meta }) }),
  getEventsTail: (topic: string, limit = 50) =>
    apiRequest<{ topic: string; limit: number; items: any[]; count: number; timestamp: number }>(`/api/events/tail?topic=${encodeURIComponent(topic)}&limit=${encodeURIComponent(String(limit))}`),
  getEventsTailEx: (topic: string, opts?: { limit?: number; q?: string; keys?: string[]; values?: string[]; since?: number; until?: number }) => {
    const qs = new URLSearchParams()
    qs.set('topic', topic)
    qs.set('limit', String(opts?.limit ?? 50))
    if (opts?.q) qs.set('q', opts.q)
    if (opts?.keys?.length) for (const k of opts.keys) qs.append('key', k)
    if (opts?.values?.length) for (const v of opts.values) qs.append('value', v)
    if (typeof opts?.since === 'number') qs.set('since', String(opts.since))
    if (typeof opts?.until === 'number') qs.set('until', String(opts.until))
    return apiRequest<{ topic: string; limit: number; items: any[]; count: number; timestamp: number }>(`/api/events/tail?${qs.toString()}`)
  },
  streamEvents: (opts: { topics?: string[]; topic?: string; limit?: number; follow?: boolean; q?: string; keys?: string[]; values?: string[]; since?: number; until?: number }, onData: (data: any) => void): EventSource => {
    const params = new URLSearchParams()
    if (opts?.topics?.length) params.set('topics', opts.topics.join(','))
    if (opts?.topic) params.set('topic', opts.topic)
    if (opts?.limit) params.set('limit', String(opts.limit))
    if (opts?.follow) params.set('follow', '1')
    if (opts?.q) params.set('q', opts.q)
    if (opts?.keys?.length) for (const k of opts.keys) params.append('key', k)
    if (opts?.values?.length) for (const v of opts.values) params.append('value', v)
    if (typeof opts?.since === 'number') params.set('since', String(opts.since))
    if (typeof opts?.until === 'number') params.set('until', String(opts.until))
    const es = new EventSource(`/api/events/stream?${params.toString()}`)
    es.onmessage = (ev) => {
      try { onData(JSON.parse(ev.data)) } catch { onData(ev.data) }
    }
    return es
  },
  getKafkaConfigs: async (topics: string[], composeFile?: string, service?: string) => {
    const qs = new URLSearchParams()
    if (topics?.length) qs.set('topics', topics.join(','))
    if (composeFile) qs.set('compose_file', composeFile)
    if (service) qs.set('service', service)
    try {
      return await apiRequest<{ topics: Record<string, { retention_ms?: number; ok: boolean; raw: string }>; timestamp: number }>(`/api/kafka/configs?${qs.toString()}`)
    } catch {
      return { topics: {}, timestamp: Date.now()/1000 } as any
    }
  },
  getKafkaSettings: async () => {
    try { return await apiRequest<{ settings: { compose_file?: string; service?: string } }>('/api/kafka/settings') } catch { return { settings: {} } as any }
  },
  // Guardrails
  getGuardrailsPending: async () => apiRequest<{ pending: any[]; timestamp: number }>(`/api/guardrails/pending-actions`).catch(()=>({ pending: [], timestamp: Date.now()/1000 } as any)),
  approveGuardrailsAction: (id: string) => apiRequest<{ success: boolean }>(`/api/guardrails/approve-action`, { method: 'POST', body: JSON.stringify({ id }) }),
  rejectGuardrailsAction: (id: string, comment?: string) => apiRequest<{ success: boolean }>(`/api/guardrails/reject-action`, { method: 'POST', body: JSON.stringify({ id, comment }) }),
  setKafkaSettings: async (settings: { compose_file?: string; service?: string }) =>
    apiRequest<{ ok: boolean; settings: any }>('/api/kafka/settings', { method: 'POST', body: JSON.stringify(settings) }),
  guardSystem: async (thresholds: { min_free_gb?: number; min_ram_gb?: number; min_vram_mb?: number }) => {
    const payload = thresholds ?? {};
    try {
      const res = await apiRequest<{ ok: boolean; disk_ok: boolean; ram_ok: boolean; gpu_ok: boolean; thresholds?: typeof payload; snapshot?: any; timestamp?: number }>(
        '/api/system/guard',
        { method: 'POST', body: JSON.stringify(payload) }
      );
      return {
        ok: !!res?.ok,
        disk_ok: !!res?.disk_ok,
        ram_ok: !!res?.ram_ok,
        gpu_ok: !!res?.gpu_ok,
        thresholds: res?.thresholds ?? payload,
        snapshot: res?.snapshot,
        timestamp: typeof res?.timestamp === 'number' ? res.timestamp : Date.now() / 1000,
      };
    } catch {
      const res: any = await apiRequest<any>('/api/system/resources').catch(() => ({}));
      const minFree = typeof payload?.min_free_gb === 'number' ? payload.min_free_gb : 2;
      const minRam = typeof payload?.min_ram_gb === 'number' ? payload.min_ram_gb : 1;
      const minVram = typeof payload?.min_vram_mb === 'number' ? payload.min_vram_mb : 0;
      const diskFree = res?.host?.disk?.free_gb ?? Infinity;
      const ramFree = res?.host?.memory?.free_gb ?? Infinity;
      const vramAvail = (() => {
        const g = Array.isArray(res?.host?.gpu) ? res.host.gpu : [];
        if (!g.length) return Infinity;
        return Math.max(...g.map((x: any) => (x.mem_total_mb ?? 0) - (x.mem_used_mb ?? 0)));
      })();
      const disk_ok = diskFree >= minFree;
      const ram_ok = ramFree >= minRam;
      const gpu_ok = vramAvail >= minVram;
      return {
        ok: disk_ok && ram_ok && gpu_ok,
        disk_ok,
        ram_ok,
        gpu_ok,
        disk_free_gb: diskFree,
        ram_free_gb: ramFree,
        vram_free_mb: vramAvail,
        thresholds: payload,
        timestamp: Date.now() / 1000,
      };
    }
  },
  runCleanup: (payload: { dry_run?: boolean; actions: { grpo_checkpoints?: { base_dir?: string; keep_last?: number }; embeddings_prune?: { dir?: string; older_than_days?: number }; kafka_prune?: { topics: string[]; retention_ms?: number; compose_file?: string; service?: string }; docker_prune?: { images?: boolean; volumes?: boolean; force?: boolean } } }) =>
    apiRequest<{ deleted: string[]; would_delete: string[]; errors: string[]; timestamp: number }>('/api/system/cleanup', { method: 'POST', headers: { ...adminHeaders() }, body: JSON.stringify(payload) }),
  getWorkspace: async () => apiRequest<{ path: string }>('/api/system/workspace'),
  setWorkspace: async (path: string) => apiRequest<{ ok: boolean; path: string }>('/api/system/workspace', { method: 'POST', body: JSON.stringify({ path }) }),
  // Capacity (admin-only; requires X-Admin-Key)
  getCapacityStatus: async (adminKey: string) =>
    apiRequest<any>('/api/capacity/status', { headers: { 'X-Admin-Key': adminKey } }).catch(() => ({ error: 'forbidden' })),
  getCapacityConfig: async (adminKey: string) =>
    apiRequest<any>('/api/capacity/config', { headers: { 'X-Admin-Key': adminKey } }).catch(() => ({ error: 'forbidden' })),
  capacityApprove: (payload: { kind: 'storage_budget_increase' | 'gpu_hours_increase'; value: number }, adminKey: string) =>
    apiRequest<any>('/api/capacity/approve', { method: 'POST', headers: { 'X-Admin-Key': adminKey }, body: JSON.stringify(payload) }),
  capacityDeny: (payload: { kind: string }, adminKey: string) =>
    apiRequest<any>('/api/capacity/deny', { method: 'POST', headers: { 'X-Admin-Key': adminKey }, body: JSON.stringify(payload) }),
  getStreamMetrics: async () => normalizeStreamMetrics(await apiRequest<any>('/api/stream-metrics').catch(() => ({}))),
  getBusMetrics: async () => normalizeBusMetrics(await apiRequest<any>('/api/bus-metrics').catch(() => ({}))),
  getOverview: async () => normalizeOverview(await apiRequest<any>('/api/overview').catch(() => ({}))),
  getProfile: async () => {
    try {
      return await apiRequest<{ profile?: string; updated_at?: number }>('/api/profile');
    } catch {
      return { profile: undefined } as any;
    }
  },
  // RL Sweep controls
  getRlSweepState: async () => {
    try {
      return await apiRequest<{ exists: boolean; state?: any; pareto?: { output: number; cost: number }[] }>('/api/rl/sweep/state');
    } catch {
      return { exists: false } as any;
    }
  },
  getRlSweepHistory: async () => {
    try {
      return await apiRequest<{ best?: any; experiments: any[] }>('/api/rl/sweep/history');
    } catch {
      return { best: null, experiments: [] } as any;
    }
  },
  runRlSweep: (payload: { method?: string; iterations?: number; trainer_steps?: number; puffer?: boolean; workspace?: string }) =>
    apiRequest<{ started: boolean; method: string; iterations: number }>('/api/rl/sweep/run', {
      method: 'POST',
      body: JSON.stringify(payload || {})
    }),
  getReddbSummary: async () => {
    try {
      return await apiRequest<{ signatures: number; recent_actions: number; recent_training: number; timestamp: number }>('/api/reddb/summary');
    } catch {
      return { signatures: 0, recent_actions: 0, recent_training: 0, timestamp: Date.now()/1000 } as any;
    }
  },
  getReddbHealth: async () => {
    try {
      return await apiRequest<{ status: string; alerts: string[]; signatures: number; recent_actions: number; recent_training: number; timestamp: number }>('/api/reddb/health');
    } catch {
      return { status: 'unknown', alerts: [], signatures: 0, recent_actions: 0, recent_training: 0, timestamp: Date.now()/1000 } as any;
    }
  },
  // Experiments
  runExperiment: (payload: { model?: string; dataset_path: string; max_count?: number; batch_size?: number; normalize?: boolean; infermesh_url?: string }) =>
    apiRequest<{ ok: boolean; id?: string; error?: string }>('/api/experiments/run', { method: 'POST', body: JSON.stringify(payload) }),
  getExperimentStatus: (id: string) =>
    apiRequest<{ id: string; status?: string; done?: number; elapsed?: number; rate?: number; logs?: string[]; [k: string]: any }>(`/api/experiments/status?id=${encodeURIComponent(id)}`),
  getExperimentHistory: () => apiRequest<{ items: any[] }>(`/api/experiments/history`),
  streamExperimentLogs: (id: string, onLine: (line: string) => void): EventSource => {
    const es = new EventSource(`/api/experiments/stream?id=${encodeURIComponent(id)}`);
    es.onmessage = (ev) => {
      try { const line = JSON.parse(ev.data); onLine(typeof line === 'string' ? line : JSON.stringify(line)); }
      catch { onLine(ev.data); }
    };
    return es;
  },
  previewDataset: (path: string) => apiRequest<{ path: string; preview: string[] }>(`/api/datasets/preview?path=${encodeURIComponent(path)}`),
  runExperimentSweep: (payload: { models?: string[]; batch_sizes?: number[]; dataset_path: string; max_count?: number; normalize?: boolean; infermesh_url?: string }) =>
    apiRequest<{ ok: boolean; id?: string; error?: string }>('/api/experiments/sweep', { method: 'POST', body: JSON.stringify(payload) }),
  getRewardsConfig: async (workspace?: string) =>
    normalizeRewardsConfig(await apiRequest<any>(`/api/rewards-config${workspace ? `?workspace=${encodeURIComponent(workspace)}` : ''}`).catch(() => ({}))),
  updateRewardsConfig: (payload: import('./types').UpdateRewardsConfigRequest, workspace?: string) =>
    apiRequest<import('./types').UpdateRewardsConfigResponse>(`/api/rewards-config${workspace ? `?workspace=${encodeURIComponent(workspace)}` : ''}`, {
      method: 'POST',
      body: JSON.stringify({ ...payload, ...(workspace ? { workspace } : {}) })
    }),
  getActionsAnalytics: async (limit = 500, timeframe: string = '24h') =>
    normalizeActionsAnalytics(await apiRequest<any>(`/api/actions-analytics?limit=${limit}&timeframe=${encodeURIComponent(timeframe)}`).catch(() => ({}))),
  getGraphSnapshots: async (limit = 10) =>
    apiRequest<GraphSnapshotsResponse>(`/api/graph/snapshots?limit=${limit}`),
  getGraphDiff: async (a: string | number, b: string | number) =>
    apiRequest<GraphDiffResponse>(`/api/graph/diff?a=${encodeURIComponent(String(a))}&b=${encodeURIComponent(String(b))}`),
  getGraphPatterns: async (mode: 'cycles' | 'mixed-language', opts?: { start?: string; maxLength?: number }) => {
    const params = new URLSearchParams();
    params.set('mode', mode);
    if (opts?.start) params.set('start', opts.start);
    if (opts?.maxLength) params.set('max_length', String(opts.maxLength));
    return apiRequest<GraphPatternsResponse>(`/api/graph/patterns?${params.toString()}`);
  },
  getGraphMctsTop: async (limit = 10) =>
    apiRequest<GraphMctsTopResponse>(`/api/graph/mcts-top?limit=${limit}`),
  getGraphMemoryReport: async (query = 'graph memory review', limit = 8) =>
    apiRequest<GraphMemoryReportResponse>(`/api/graph/memory-report?query=${encodeURIComponent(query)}&limit=${limit}`),
  updateConfig: (payload: UpdateConfigRequest) =>
    apiRequest<UpdateConfigResponse>('/api/config', {
      method: 'POST',
      body: JSON.stringify(payload)
    }),
  sendActionFeedback: (payload: { action_id: string; decision: 'approve' | 'reject' | 'suggest'; comment?: string }) =>
    apiRequest<{ success: boolean; error?: string }>('/api/action/feedback', {
      method: 'POST',
      body: JSON.stringify(payload)
    }),
  getGuardrailsState: async () => {
    try {
      return await apiRequest<{ enabled: boolean; pending: { id: string; command: string; workspace?: string; logs?: string; ts: number }[]; timestamp: number }>('/api/guardrails/state');
    } catch {
      return { enabled: false, pending: [], timestamp: Date.now() / 1000 };
    }
  },
  setGuardrails: async (enabled: boolean) => {
    try {
      return await apiRequest<{ success: boolean; enabled: boolean }>('/api/guardrails', {
        method: 'POST',
        body: JSON.stringify({ enabled })
      });
    } catch {
      return { success: false, enabled };
    }
  },
  approvePending: async (id: string) => {
    try {
      return await apiRequest<{ success: boolean; output?: string; error?: string }>('/api/guardrails/approve', {
        method: 'POST',
        body: JSON.stringify({ id })
      });
    } catch {
      return { success: false };
    }
  },
  rejectPending: async (id: string) => {
    try {
      return await apiRequest<{ success: boolean }>('/api/guardrails/reject', {
        method: 'POST',
        body: JSON.stringify({ id })
      });
    } catch {
      return { success: false };
    }
  },
  getProposedActions: async () => {
    try {
      return await apiRequest<{ pending: { id: string; type: string; ts: number; payload?: any }[]; timestamp: number }>('/api/guardrails/pending-actions');
    } catch {
      return { pending: [], timestamp: Date.now() / 1000 };
    }
  },
  approveAction: async (id: string) => {
    try {
      return await apiRequest<{ success: boolean }>('/api/guardrails/approve-action', { method: 'POST', body: JSON.stringify({ id }) });
    } catch {
      return { success: false };
    }
  },
  rejectAction: async (id: string, comment?: string) => {
    try {
      return await apiRequest<{ success: boolean }>('/api/guardrails/reject-action', { method: 'POST', body: JSON.stringify({ id, comment }) });
    } catch {
      return { success: false };
    }
  },
  // GRPO controls
  getGrpoStatus: async () => {
    try {
      return await apiRequest<{ running: boolean; error?: string; step?: number; started_at?: number; finished_at?: number; config?: any; timestamp: number }>('/api/grpo/status');
    } catch {
      return { running: false, error: 'unavailable', timestamp: Date.now() / 1000 } as any;
    }
  },
  getGrpoMetrics: async (limit = 200) => {
    try {
      return await apiRequest<{ metrics: any[]; count: number; timestamp: number }>(`/api/grpo/metrics?limit=${limit}`);
    } catch {
      return { metrics: [], count: 0, timestamp: Date.now() / 1000 } as any;
    }
  },
  getGrpoDatasetStats: async (path: string, limit = 20000) =>
    apiRequest<{ path: string; scanned: number; candidates: { avg: number; min: number; max: number }; rewards: { hist_bins: number[]; hist_counts: number[]; count: number; mean: number }; top_keywords: string[]; sample_prompts: string[]; quality?: { empty_prompts: number; short_prompts: number; missing_text: number; short_text: number; reward_min: number; reward_max: number; reward_mean: number; reward_std: number; reward_outliers_std3: number; reward_out_of_range_0_1: number }; timestamp: number }>(`/api/grpo/dataset-stats?path=${encodeURIComponent(path)}&limit=${limit}`),
  applyGrpoPolicy: (payload: { hours?: number; top_k?: number; bottom_k?: number; min_count?: number; min_avg_for_deny?: number; workspace?: string }) =>
    apiRequest<{ ok: boolean; applied?: { prefer?: string; deny?: string; regex?: string }[]; error?: string }>(`/api/grpo/apply-policy`, { method: 'POST', body: JSON.stringify(payload) }),
  getGrpoLevelMetrics: (level: 'global'|'signature'|'patch', root?: string, limit = 500) =>
    apiRequest<{ metrics: any[]; count: number; level: string; path: string; timestamp: number }>(`/api/grpo/level-metrics?level=${encodeURIComponent(level)}${root ? `&root=${encodeURIComponent(root)}` : ''}&limit=${limit}`),
  getPolicySummary: async () => {
    try {
      return await apiRequest<{ tools: Record<string, { last24h: { mean: number; count: number }; last7d: { mean: number; count: number }; delta?: number }>; rules: any[]; policy: { prefer_tools: string[]; deny_tools: string[] } }>('/api/policy/summary');
    } catch {
      return { tools: {}, rules: [], policy: { prefer_tools: [], deny_tools: [] } } as any;
    }
  },
  startGrpo: (payload: { dataset_path: string; model_name?: string; reference_model_name?: string; device?: string; batch_groups?: number; lr?: number; max_steps?: number; log_interval?: number; ckpt_interval?: number; adv_clip?: number; kl_coeff?: number }) =>
    apiRequest<{ ok: boolean; error?: string }>('/api/grpo/start', { method: 'POST', body: JSON.stringify(payload) }),
  stopGrpo: () => apiRequest<{ ok: boolean; error?: string }>('/api/grpo/stop', { method: 'POST' }),
  listWorkflows: async () => {
    try {
      const res = await apiRequest<WorkflowListResponse>('/workflows');
      return Array.isArray(res?.items) ? res.items : [];
    } catch {
      return [] as WorkflowSummary[];
    }
  },
  getWorkflow: (id: string) => apiRequest<WorkflowDefinition>(`/workflows/${encodeURIComponent(id)}`),
  getWorkflowRuns: async (workflowId: string, limit = 25) => {
    if (!workflowId) return [] as WorkflowRunRecord[];
    const res = await apiRequest<WorkflowRunListResponse>(
      `/workflows/${encodeURIComponent(workflowId)}/runs${limit ? `?limit=${limit}` : ''}`
    );
    return Array.isArray(res?.items) ? res.items : [];
  },
  getWorkflowRun: (runId: string) => apiRequest<WorkflowRunRecord>(`/workflow-runs/${encodeURIComponent(runId)}`),
  getRunnerHardware: () => apiRequest<RunnerHardwareSnapshot>('/hardware'),
  saveWorkflow: (workflow: WorkflowDefinition) => {
    const hasId = Boolean(workflow.id);
    const path = hasId ? `/workflows/${encodeURIComponent(String(workflow.id))}` : '/workflows';
    const method = hasId ? 'PUT' : 'POST';
    return apiRequest<WorkflowDefinition>(path, {
      method,
      body: JSON.stringify(workflow)
    });
  },
  // Auto mode
  getGrpoAutoStatus: async () => {
    try {
      return await apiRequest<{ running: boolean; iterations?: number; mode?: string; last_cycle?: number; last_datasets?: Record<string, string>; error?: string; config?: any; timestamp: number }>('/api/grpo/auto/status');
    } catch {
      return { running: false, timestamp: Date.now() / 1000 } as any;
    }
  },
  // Env Queue
  getEnvQueueStats: async () => {
    try { return await apiRequest<{ pending: number; done: number; items: any[]; timestamp: number }>(`/api/env-queue/stats`) } catch { return { pending: 0, done: 0, items: [], timestamp: Date.now()/1000 } as any }
  },
  submitEnvTask: (payload: { id?: string; class?: 'cpu_short'|'cpu_long'|'gpu'; payload?: string }) =>
    apiRequest<{ ok: boolean; file: string; base: string }>(`/api/env-queue/submit`, { method: 'POST', body: JSON.stringify(payload) }),
  startGrpoAuto: (payload: { period_sec?: number; hours?: number; min_groups?: number; mode?: 'single'|'hierarchical'; steps?: number; out_dir?: string; model_name?: string; device?: string; levels?: string[]; apply_policy?: boolean; workspace?: string; policy_top_k?: number; policy_bottom_k?: number; policy_min_avg_for_deny?: number; lr_step_size?: number; lr_gamma?: number; kl_warmup_steps?: number; kl_target?: number }) =>
    apiRequest<{ ok: boolean; error?: string }>('/api/grpo/auto/start', { method: 'POST', body: JSON.stringify(payload) }),
  stopGrpoAuto: () => apiRequest<{ ok: boolean; error?: string }>('/api/grpo/auto/stop', { method: 'POST' })
};

// ---- helpers ----
function normalizeStatus(data: any): StatusResponse {
  const mk = (x: any): { status: any; details?: string } => {
    if (!x || typeof x !== 'object') return { status: 'unknown', details: 'No data' };
    const status = (x.status ?? x.state ?? 'unknown') as any;
    const details = typeof x.details === 'string' ? x.details : (x.message as string | undefined);
    return { status, ...(details ? { details } : {}) };
  };
  return {
    agent: mk(data?.agent),
    ollama: mk(data?.ollama),
    kafka: mk(data?.kafka),
    containers: mk(data?.containers),
    reddb: mk(data?.reddb),
    spark: mk(data?.spark),
    embeddings: mk(data?.embeddings),
    pipeline: mk(data?.pipeline),
    learning_active: !!data?.learning_active,
    auto_training: !!data?.auto_training,
    timestamp: typeof data?.timestamp === 'number' ? data.timestamp : Date.now() / 1000,
  };
}

function normalizeMetrics(data: any): MetricsResponse {
  return {
    timestamp: typeof data?.timestamp === 'number' ? data.timestamp : Date.now() / 1000,
    containers: Number.isFinite(data?.containers) ? Number(data.containers) : 0,
    memory_usage: typeof data?.memory_usage === 'string' ? data.memory_usage : '--',
    response_time: Number.isFinite(data?.response_time) ? Number(data.response_time) : 0,
  };
}

function normalizeBusMetrics(data: any): import('./types').BusMetricsResponse {
  const empty = { bins: [], counts: [] } as any;
  return {
    bus: {
      dlq_total: Number.isFinite(data?.bus?.dlq_total) ? Number(data.bus.dlq_total) : 0,
      topics: (data?.bus?.topics && typeof data.bus.topics === 'object') ? data.bus.topics : {},
      groups: (data?.bus?.groups && typeof data.bus.groups === 'object') ? data.bus.groups : {},
    },
    dlq: {
      total: Number.isFinite(data?.dlq?.total) ? Number(data.dlq.total) : 0,
      by_topic: (data?.dlq?.by_topic && typeof data.dlq.by_topic === 'object') ? data.dlq.by_topic : {},
      last_ts: typeof data?.dlq?.last_ts === 'number' ? data.dlq.last_ts : null,
    },
    alerts: Array.isArray(data?.alerts) ? data.alerts : [],
    history: (data?.history && typeof data.history === 'object') ? {
      timestamps: Array.isArray(data.history.timestamps) ? data.history.timestamps : [],
      queue_max_depth: Array.isArray(data.history.queue_max_depth) ? data.history.queue_max_depth : [],
      dlq_total: Array.isArray(data.history.dlq_total) ? data.history.dlq_total : [],
    } : undefined,
    thresholds: (data?.thresholds && typeof data.thresholds === 'object') ? data.thresholds : undefined,
    timestamp: Number.isFinite(data?.timestamp) ? Number(data.timestamp) : Date.now() / 1000,
  };
}

function normalizePerformanceHistory(data: any, timeframe: string): PerformanceHistoryResponse {
  return {
    timestamps: Array.isArray(data?.timestamps) ? data.timestamps : [],
    metrics: {
      response_times: Array.isArray(data?.metrics?.response_times) ? data.metrics.response_times : [],
      success_rates: Array.isArray(data?.metrics?.success_rates) ? data.metrics.success_rates : [],
      throughput: Array.isArray(data?.metrics?.throughput) ? data.metrics.throughput : [],
      error_rates: Array.isArray(data?.metrics?.error_rates) ? data.metrics.error_rates : [],
    },
    timeframe: typeof data?.timeframe === 'string' ? data.timeframe : timeframe,
    interval: typeof data?.interval === 'string' ? data.interval : 'auto',
    timestamp: Number.isFinite(data?.timestamp) ? Number(data.timestamp) : Date.now() / 1000,
  };
}

function normalizeSparkWorkers(data: any): SparkWorkersResponse {
  return {
    master: {
      status: data?.master?.status ?? 'unknown',
      workers: Number.isFinite(data?.master?.workers) ? Number(data.master.workers) : 0,
      cores_total: Number.isFinite(data?.master?.cores_total) ? Number(data.master.cores_total) : 0,
      cores_used: Number.isFinite(data?.master?.cores_used) ? Number(data.master.cores_used) : 0,
      memory_total: data?.master?.memory_total ?? '0GB',
      memory_used: data?.master?.memory_used ?? '0GB',
      applications_running: Number.isFinite(data?.master?.applications_running) ? Number(data.master.applications_running) : 0,
      applications_completed: Number.isFinite(data?.master?.applications_completed) ? Number(data.master.applications_completed) : 0,
    },
    workers: Array.isArray(data?.workers) ? data.workers : [],
    applications: Array.isArray(data?.applications) ? data.applications : [],
    cluster_metrics: {
      total_cores: Number.isFinite(data?.cluster_metrics?.total_cores) ? Number(data.cluster_metrics.total_cores) : 0,
      used_cores: Number.isFinite(data?.cluster_metrics?.used_cores) ? Number(data.cluster_metrics.used_cores) : 0,
      total_memory: data?.cluster_metrics?.total_memory ?? '0GB',
      used_memory: data?.cluster_metrics?.used_memory ?? '0GB',
      cpu_utilization: Number.isFinite(data?.cluster_metrics?.cpu_utilization) ? Number(data.cluster_metrics.cpu_utilization) : 0,
    },
    timestamp: Number.isFinite(data?.timestamp) ? Number(data.timestamp) : Date.now() / 1000,
  };
}

function normalizeStreamMetrics(data: any): StreamMetricsResponse {
  const cm = data?.current_metrics ?? {};
  return {
    current_metrics: {
      kafka_throughput: {
        messages_per_second: Number.isFinite(cm?.kafka_throughput?.messages_per_second) ? Number(cm.kafka_throughput.messages_per_second) : 0,
        bytes_per_second: Number.isFinite(cm?.kafka_throughput?.bytes_per_second) ? Number(cm.kafka_throughput.bytes_per_second) : 0,
        producer_rate: Number.isFinite(cm?.kafka_throughput?.producer_rate) ? Number(cm.kafka_throughput.producer_rate) : 0,
        consumer_rate: Number.isFinite(cm?.kafka_throughput?.consumer_rate) ? Number(cm.kafka_throughput.consumer_rate) : 0,
      },
      spark_streaming: {
        batch_duration: cm?.spark_streaming?.batch_duration ?? '0s',
        processing_time: Number.isFinite(cm?.spark_streaming?.processing_time) ? Number(cm.spark_streaming.processing_time) : 0,
        scheduling_delay: Number.isFinite(cm?.spark_streaming?.scheduling_delay) ? Number(cm.spark_streaming.scheduling_delay) : 0,
        total_delay: Number.isFinite(cm?.spark_streaming?.total_delay) ? Number(cm.spark_streaming.total_delay) : 0,
        records_per_batch: Number.isFinite(cm?.spark_streaming?.records_per_batch) ? Number(cm.spark_streaming.records_per_batch) : 0,
        batches_completed: Number.isFinite(cm?.spark_streaming?.batches_completed) ? Number(cm.spark_streaming.batches_completed) : 0,
      },
      data_pipeline: {
        input_rate: Number.isFinite(cm?.data_pipeline?.input_rate) ? Number(cm.data_pipeline.input_rate) : 0,
        output_rate: Number.isFinite(cm?.data_pipeline?.output_rate) ? Number(cm.data_pipeline.output_rate) : 0,
        error_rate: Number.isFinite(cm?.data_pipeline?.error_rate) ? Number(cm.data_pipeline.error_rate) : 0,
        backpressure: !!cm?.data_pipeline?.backpressure,
        queue_depth: Number.isFinite(cm?.data_pipeline?.queue_depth) ? Number(cm.data_pipeline.queue_depth) : 0,
      },
      network_io: {
        bytes_in_per_sec: Number.isFinite(cm?.network_io?.bytes_in_per_sec) ? Number(cm.network_io.bytes_in_per_sec) : 0,
        bytes_out_per_sec: Number.isFinite(cm?.network_io?.bytes_out_per_sec) ? Number(cm.network_io.bytes_out_per_sec) : 0,
        packets_in_per_sec: Number.isFinite(cm?.network_io?.packets_in_per_sec) ? Number(cm.network_io.packets_in_per_sec) : 0,
        packets_out_per_sec: Number.isFinite(cm?.network_io?.packets_out_per_sec) ? Number(cm.network_io.packets_out_per_sec) : 0,
        connections_active: Number.isFinite(cm?.network_io?.connections_active) ? Number(cm.network_io.connections_active) : 0,
      },
    },
    time_series: Array.isArray(data?.time_series) ? data.time_series : [],
    alerts: Array.isArray(data?.alerts) ? data.alerts : [],
    timestamp: Number.isFinite(data?.timestamp) ? Number(data.timestamp) : Date.now() / 1000,
  };
}

function normalizeOverview(data: any): OverviewResponse {
  return {
    status: normalizeStatus(data?.status || {} as any),
    metrics: normalizeMetrics(data?.metrics || {} as any),
    rl: normalizeRlMetrics(data?.rl || {} as any),
    bus: normalizeBusMetrics(data?.bus || {} as any),
    timestamp: Number.isFinite(data?.timestamp) ? Number(data.timestamp) : Date.now() / 1000,
  };
}

function normalizeSignatures(data: any): SignaturesResponse {
  const signatures = Array.isArray(data?.signatures) ? data.signatures : [];
  return {
    signatures,
    total_active: Number.isFinite(data?.total_active) ? Number(data.total_active) : signatures.filter((s: any) => !!s?.active).length,
    avg_performance: Number.isFinite(data?.avg_performance) ? Number(data.avg_performance) : (signatures.length ? (signatures.reduce((a: number, s: any) => a + (s?.performance ?? 0), 0) / signatures.length) : 0),
    timestamp: Number.isFinite(data?.timestamp) ? Number(data.timestamp) : Date.now() / 1000,
  };
}

function normalizeVerifiers(data: any): VerifiersResponse {
  const verifiers = Array.isArray(data?.verifiers) ? data.verifiers : [];
  return {
    verifiers,
    total_active: Number.isFinite(data?.total_active) ? Number(data.total_active) : verifiers.length,
    avg_accuracy: Number.isFinite(data?.avg_accuracy) ? Number(data.avg_accuracy) : (verifiers.length ? (verifiers.reduce((a: number, v: any) => a + (v?.accuracy ?? 0), 0) / verifiers.length) : 0),
    total_checks: Number.isFinite(data?.total_checks) ? Number(data.total_checks) : 0,
    total_issues: Number.isFinite(data?.total_issues) ? Number(data.total_issues) : 0,
    timestamp: Number.isFinite(data?.timestamp) ? Number(data.timestamp) : Date.now() / 1000,
  };
}

function normalizeLearningMetrics(data: any): LearningMetricsResponse {
  const pot = data?.performance_over_time ?? {};
  const ls = data?.learning_stats ?? {};
  const ru = data?.resource_usage ?? {};
  return {
    performance_over_time: {
      timestamps: Array.isArray(pot.timestamps) ? pot.timestamps : [],
      overall_performance: Array.isArray(pot.overall_performance) ? pot.overall_performance : [],
      training_accuracy: Array.isArray(pot.training_accuracy) ? pot.training_accuracy : [],
      validation_accuracy: Array.isArray(pot.validation_accuracy) ? pot.validation_accuracy : [],
    },
    signature_performance: typeof data?.signature_performance === 'object' ? data.signature_performance : {},
    learning_stats: {
      total_training_examples: Number.isFinite(ls.total_training_examples) ? Number(ls.total_training_examples) : 0,
      successful_optimizations: Number.isFinite(ls.successful_optimizations) ? Number(ls.successful_optimizations) : 0,
      failed_optimizations: Number.isFinite(ls.failed_optimizations) ? Number(ls.failed_optimizations) : 0,
      avg_improvement_per_iteration: Number.isFinite(ls.avg_improvement_per_iteration) ? Number(ls.avg_improvement_per_iteration) : 0,
      current_learning_rate: Number.isFinite(ls.current_learning_rate) ? Number(ls.current_learning_rate) : 0,
    },
    resource_usage: {
      memory_usage: Array.isArray(ru.memory_usage) ? ru.memory_usage : [],
      cpu_usage: Array.isArray(ru.cpu_usage) ? ru.cpu_usage : [],
      gpu_usage: Array.isArray(ru.gpu_usage) ? ru.gpu_usage : [],
    },
    timestamp: Number.isFinite(data?.timestamp) ? Number(data.timestamp) : Date.now() / 1000,
  };
}

function normalizeRewardsConfig(data: any): import('./types').RewardsConfigResponse {
  const def = {
    policy: 'bandit',
    epsilon: 0.1,
    ucb_c: 1.0,
    n_envs: 1,
    puffer: false,
    weights: { pass_rate: 1.0, blast_radius: 1.0 },
    penalty_kinds: ['blast_radius'],
    clamp01_kinds: ['pass_rate'],
    scales: { blast_radius: [0, 1] as [number, number] },
    actions: [],
    test_cmd: null as string | null,
    lint_cmd: null as string | null,
    build_cmd: null as string | null,
    timeout_sec: null as number | null,
    path: undefined as string | undefined,
  };
  return { ...def, ...(typeof data === 'object' ? data : {}) };
}

function normalizeActionsAnalytics(data: any): import('./types').ActionsAnalyticsResponse {
  return {
    counts_by_type: (data?.counts_by_type && typeof data.counts_by_type === 'object') ? data.counts_by_type : {},
    reward_hist: (data?.reward_hist && typeof data.reward_hist === 'object') ? data.reward_hist : { bins: [], counts: [] },
    duration_hist: (data?.duration_hist && typeof data.duration_hist === 'object') ? data.duration_hist : { bins: [], counts: [] },
    top_actions: Array.isArray(data?.top_actions) ? data.top_actions : [],
    worst_actions: Array.isArray(data?.worst_actions) ? data.worst_actions : [],
    recent: Array.isArray(data?.recent) ? data.recent : [],
    timestamp: Number.isFinite(data?.timestamp) ? Number(data.timestamp) : Date.now() / 1000,
  };
}
function normalizeRlMetrics(data: any): RlMetricsResponse {
  const m = (data && typeof data === 'object' && data.metrics) ? data.metrics : data || {};

  const toNum = (v: any, def = 0) => (typeof v === 'number' && Number.isFinite(v) ? v : def);
  const nowIso = new Date().toISOString();

  const metrics: RlMetricsResponse['metrics'] = {
    training_status: (m.training_status || m.status || 'UNKNOWN') as string,
    current_episode: toNum(m.current_episode, toNum(data?.current_episode, 0)),
    total_episodes: toNum(m.total_episodes, toNum(data?.episodes, toNum(m.current_episode, 0))),
    avg_reward: toNum(m.avg_reward, toNum(m.average_reward, toNum(data?.avg_reward, 0))),
    best_reward: toNum(m.best_reward, 0),
    worst_reward: toNum(m.worst_reward, 0),
    epsilon: toNum(m.epsilon, toNum(m.exploration_rate, toNum(data?.epsilon, 0))),
    learning_rate: toNum(m.learning_rate, 0),
    loss: toNum(m.loss, 0),
    q_value_mean: toNum(m.q_value_mean, 0),
    exploration_rate: toNum(m.exploration_rate, toNum(m.epsilon, 0)),
    replay_buffer_size: toNum(m.replay_buffer_size, 0),
    replay_buffer_used: toNum(m.replay_buffer_used, 0)
  };

  // Reward history normalization
  let reward_history: RlMetricsResponse['reward_history'] = [];
  if (Array.isArray(data?.reward_history)) {
    reward_history = data.reward_history.map((r: any) => ({
      episode: toNum(r?.episode, 0),
      reward: toNum(r?.reward, 0),
      timestamp: typeof r?.timestamp === 'string' ? r.timestamp : nowIso
    }));
  } else if (Array.isArray(data?.recent_rewards)) {
    const baseEp = metrics.total_episodes - data.recent_rewards.length + 1;
    reward_history = data.recent_rewards.map((rw: any, i: number) => ({
      episode: baseEp + i,
      reward: toNum(rw, 0),
      timestamp: nowIso
    }));
  }

  const action_stats = (data?.action_stats && typeof data.action_stats === 'object') ? data.action_stats : {};
  const environment_info = (data?.environment_info && typeof data.environment_info === 'object') ? data.environment_info : {
    state_space_size: 0,
    action_space_size: 0,
    observation_type: 'unknown',
    reward_range: [0, 0] as [number, number]
  };

  return {
    metrics,
    reward_history,
    action_stats,
    environment_info,
    timestamp: toNum(data?.timestamp, Date.now() / 1000)
  };
}
