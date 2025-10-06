import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Bar, Line } from 'react-chartjs-2';
import Card from '../components/Card';
import StatusPill from '../components/StatusPill';
import { api } from '../api/client';
import type { ActionsAnalyticsResponse, RlMetricsResponse } from '../api/types';
import { ensureChartsRegistered } from '../lib/registerCharts';

ensureChartsRegistered();

type TimelineMode = 'training' | 'inference' | 'verification' | 'system' | 'unknown';

interface TimelineEvent {
  id: string;
  timestamp: number;
  topic: string;
  mode: TimelineMode;
  title: string;
  summary: string;
  reward?: number;
  score?: number;
  environment?: string;
  payload: any;
}

const MAX_EVENTS = 200;

const AgentPulsePage = () => {
  const [timeframe, setTimeframe] = useState<'1h' | '24h' | '7d'>('24h');
  const [limit, setLimit] = useState(400);
  const [live, setLive] = useState(true);
  const [modeFilter, setModeFilter] = useState<'all' | TimelineMode>('all');
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const seen = useRef<Set<string>>(new Set());
  const streamRef = useRef<EventSource | null>(null);

  const { data: actionsData, isFetching: actionsLoading } = useQuery({
    queryKey: ['agent-pulse', timeframe, limit],
    queryFn: () => api.getActionsAnalytics(limit, timeframe),
    refetchInterval: 20000
  });

  const { data: rlMetrics } = useQuery({
    queryKey: ['rl-metrics'],
    queryFn: api.getRlMetrics,
    refetchInterval: 15000
  });

  const { data: status } = useQuery({
    queryKey: ['status'],
    queryFn: api.getStatus,
    refetchInterval: 10000
  });

  const pushEvents = (events: TimelineEvent[]) => {
    if (!events.length) return;
    setTimeline((prev) => {
      const next: TimelineEvent[] = [];
      const localSeen = new Set(seen.current);
      for (const item of [...events, ...prev]) {
        const key = item.id || `${item.topic}-${item.timestamp}-${item.title}`;
        if (localSeen.has(key)) continue;
        localSeen.add(key);
        next.push(item);
        if (next.length >= MAX_EVENTS) break;
      }
      seen.current = localSeen;
      return next;
    });
  };

  useEffect(() => {
    if (!actionsData) return;
    const bootstrap = (actionsData.recent || []).map(convertActionToEvent);
    pushEvents(bootstrap);
  }, [actionsData?.timestamp]);

  useEffect(() => {
    if (!live) {
      if (streamRef.current) {
        streamRef.current.close();
        streamRef.current = null;
      }
      return;
    }

    const es = api.streamEvents(
      {
        topics: ['agent.action', 'training.result', 'training.episode', 'training.grpo'],
        follow: true,
        limit: 200
      },
      (payload) => {
        const items = extractStreamEvents(payload);
        const mapped = items
          .map(({ topic, record }) => normalizeTimelineEvent(topic, record))
          .filter((item): item is TimelineEvent => !!item);
        pushEvents(mapped);
      }
    );
    streamRef.current = es;
    return () => {
      es.close();
      streamRef.current = null;
    };
  }, [live]);

  const filteredTimeline = useMemo(() => {
    if (modeFilter === 'all') return timeline;
    return timeline.filter((item) => item.mode === modeFilter);
  }, [timeline, modeFilter]);

  const rewardSeries = useMemo(() => {
    const series = (actionsData?.recent || []).slice(0, 50).reverse();
    return {
      labels: series.map((item) => new Date(item.timestamp * 1000).toLocaleTimeString()),
      datasets: [
        {
          label: 'Reward',
          data: series.map((item) => item.reward ?? 0),
          borderColor: '#22d3ee',
          backgroundColor: 'rgba(34, 211, 238, 0.18)',
          tension: 0.35,
          fill: true,
          pointRadius: 0
        }
      ]
    };
  }, [actionsData?.recent]);

  const actionTypeChart = useMemo(() => {
    const counts = actionsData?.counts_by_type || {};
    const labels = Object.keys(counts);
    const values = labels.map((label) => counts[label] ?? 0);
    return {
      labels,
      datasets: [
        {
          label: 'Actions',
          data: values,
          backgroundColor: labels.map((label) => colorForType(label)),
          borderWidth: 0
        }
      ]
    };
  }, [actionsData?.counts_by_type]);

  const durationChart = useMemo(() => {
    const hist = actionsData?.duration_hist;
    const bins = Array.isArray(hist?.bins) ? hist?.bins : [];
    const counts = Array.isArray(hist?.counts) ? hist?.counts : [];
    const labels = bins.slice(0, -1).map((val, idx) => {
      const next = bins[idx + 1] ?? val;
      return `${val.toFixed(2)}s – ${next.toFixed(2)}s`;
    });
    return {
      labels,
      datasets: [
        {
          label: 'Executions',
          data: counts,
          backgroundColor: '#a855f7',
          borderWidth: 0
        }
      ]
    };
  }, [actionsData?.duration_hist]);

  const rewardDistribution = useMemo(() => {
    const hist = actionsData?.reward_hist;
    const bins = Array.isArray(hist?.bins) ? hist?.bins : [];
    const counts = Array.isArray(hist?.counts) ? hist?.counts : [];
    const labels = bins.slice(0, -1).map((val, idx) => {
      const next = bins[idx + 1] ?? val;
      return `${val.toFixed(2)} – ${next.toFixed(2)}`;
    });
    return {
      labels,
      datasets: [
        {
          label: 'Frequency',
          data: counts,
          backgroundColor: '#34d399',
          borderWidth: 0
        }
      ]
    };
  }, [actionsData?.reward_hist]);

  const pulseSummary = useMemo(() => buildSummary(actionsData, rlMetrics), [actionsData, rlMetrics]);
  const rlSummary = rlMetrics?.metrics;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <h1 className="text-3xl font-semibold text-white">Agent Pulse</h1>
          <p className="text-sm text-slate-300">
            Observe live activity across training runs, inference calls, verifier feedback, and reward signals.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="rounded-lg border border-slate-600 bg-slate-900/60 px-3 py-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Agent</span>
              <StatusPill status={status?.agent?.status} />
              <span className="text-slate-200">{status?.agent?.details || 'Operational'}</span>
            </div>
          </div>
          <div className="rounded-lg border border-slate-600 bg-slate-900/60 px-3 py-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Training</span>
              <StatusPill status={status?.pipeline?.status} />
              <span className="text-slate-200">{rlSummary?.training_status ?? 'Unknown'}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {pulseSummary.map((item) => (
          <Card key={item.label} className="p-4" variant="outlined">
            <div className="text-sm text-slate-400">{item.label}</div>
            <div className="mt-2 text-2xl font-semibold text-white">{item.value}</div>
            {item.detail && <div className="mt-1 text-xs text-slate-500">{item.detail}</div>}
          </Card>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2" title="Live Timeline" subtitle="Streaming events across training and inference">
          <div className="flex flex-wrap items-center gap-3 pb-4">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <label htmlFor="timeframe-select">Window</label>
              <select
                id="timeframe-select"
                className="rounded border border-slate-600 bg-slate-900 px-2 py-1 text-xs"
                value={timeframe}
                onChange={(event) => setTimeframe(event.target.value as typeof timeframe)}
              >
                <option value="1h">Last hour</option>
                <option value="24h">Last 24h</option>
                <option value="7d">Last 7d</option>
              </select>
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <label htmlFor="limit-input">History</label>
              <input
                id="limit-input"
                type="number"
                min={100}
                max={2000}
                step={100}
                value={limit}
                onChange={(event) => setLimit(Math.max(100, Number(event.target.value) || 100))}
                className="w-24 rounded border border-slate-600 bg-slate-900 px-2 py-1 text-xs"
              />
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <span>Filter</span>
              {(['all', 'training', 'inference', 'verification', 'system'] as const).map((entry) => (
                <button
                  key={entry}
                  className={`rounded-full px-3 py-1 text-xs transition ${
                    modeFilter === entry
                      ? 'bg-emerald-500 text-slate-900'
                      : 'border border-slate-600 bg-slate-900 text-slate-300 hover:border-emerald-500 hover:text-white'
                  }`}
                  onClick={() => setModeFilter(entry)}
                >
                  {entry.charAt(0).toUpperCase() + entry.slice(1)}
                </button>
              ))}
            </div>
            <button
              className={`ml-auto rounded-full px-3 py-1 text-xs font-medium ${
                live ? 'bg-rose-500 text-white' : 'bg-slate-800 text-slate-300 border border-slate-600'
              }`}
              onClick={() => setLive((prev) => !prev)}
            >
              {live ? 'Streaming…' : 'Resume stream'}
            </button>
          </div>

          <div className="space-y-3 max-h-[420px] overflow-auto pr-1">
            {actionsLoading && timeline.length === 0 && (
              <div className="text-sm text-slate-400">Loading activity…</div>
            )}
            {!actionsLoading && filteredTimeline.length === 0 && (
              <div className="text-sm text-slate-400">No activity yet for this filter.</div>
            )}
            {filteredTimeline.map((item) => (
              <TimelineRow key={item.id} event={item} />
            ))}
          </div>
        </Card>

        <div className="space-y-6">
          <Card title="Reward Trend" subtitle="Recent rewards across actions">
            <div className="h-48">
              <Line
                data={rewardSeries}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } },
                  scales: {
                    x: {
                      ticks: { display: false },
                      grid: { display: false }
                    },
                    y: {
                      ticks: { color: '#94a3b8', precision: 2 },
                      grid: { color: 'rgba(148, 163, 184, 0.15)' }
                    }
                  }
                }}
              />
            </div>
          </Card>

          <Card title="RL Snapshot" subtitle="Live trainer health">
            <RLSnapshot rl={rlMetrics} />
          </Card>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Action Mix" subtitle="Distribution of agent actions">
          <div className="h-52">
            <Bar
              data={actionTypeChart}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  x: {
                    ticks: { color: '#94a3b8', autoSkip: false },
                    grid: { display: false }
                  },
                  y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.15)' }
                  }
                }
              }}
            />
          </div>
        </Card>

        <Card title="Reward Distribution" subtitle="Histogram of rewards">
          <div className="h-52">
            <Bar
              data={rewardDistribution}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  x: {
                    ticks: { color: '#94a3b8', maxRotation: 0, minRotation: 0 },
                    grid: { display: false }
                  },
                  y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.15)' }
                  }
                }
              }}
            />
          </div>
        </Card>

        <Card title="Execution Time" subtitle="Latency buckets">
          <div className="h-52">
            <Bar
              data={durationChart}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  x: {
                    ticks: { color: '#94a3b8', maxRotation: 0, minRotation: 0 },
                    grid: { display: false }
                  },
                  y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.15)' }
                  }
                }
              }}
            />
          </div>
        </Card>
      </div>
    </div>
  );
};

const TimelineRow = ({ event }: { event: TimelineEvent }) => {
  const ts = new Date(event.timestamp * 1000);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(event.payload, null, 2));
    } catch {
      /* ignore copy errors */
    }
  };

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/60 p-3">
      <div className="flex flex-wrap items-center gap-3">
        <span className={`text-xs font-medium uppercase tracking-wide ${modeBadgeClass(event.mode)}`}>
          {event.mode}
        </span>
        <span className="text-sm font-semibold text-white">{event.title}</span>
        <span className="text-xs text-slate-400">{event.topic}</span>
        <span className="text-xs text-slate-500">{ts.toLocaleTimeString()}</span>
        <div className="ml-auto flex items-center gap-2 text-xs text-slate-400">
          {typeof event.reward === 'number' && (
            <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-emerald-300">
              Reward {event.reward.toFixed(2)}
            </span>
          )}
          {typeof event.score === 'number' && (
            <span className="rounded bg-sky-500/10 px-2 py-0.5 text-sky-300">
              Score {event.score.toFixed(2)}
            </span>
          )}
          <button
            className="rounded border border-slate-600 px-2 py-0.5 text-xs text-slate-300 hover:border-emerald-500 hover:text-white"
            onClick={handleCopy}
          >
            Copy payload
          </button>
        </div>
      </div>
      {event.summary && <div className="mt-2 text-sm text-slate-300">{event.summary}</div>}
    </div>
  );
};

const RLSnapshot = ({ rl }: { rl?: RlMetricsResponse }) => {
  if (!rl) {
    return <div className="text-sm text-slate-400">No trainer metrics reported yet.</div>;
  }
  const metrics = rl.metrics;
  return (
    <div className="space-y-3 text-sm text-slate-200">
      <div className="flex items-center justify-between"><span className="text-slate-400">Status</span><span>{metrics.training_status}</span></div>
      <div className="flex items-center justify-between"><span className="text-slate-400">Episode</span><span>{metrics.current_episode} / {metrics.total_episodes}</span></div>
      <div className="flex items-center justify-between"><span className="text-slate-400">Avg reward</span><span>{metrics.avg_reward.toFixed(3)}</span></div>
      <div className="flex items-center justify-between"><span className="text-slate-400">Best reward</span><span>{metrics.best_reward.toFixed(3)}</span></div>
      <div className="flex items-center justify-between"><span className="text-slate-400">Exploration</span><span>ε {metrics.epsilon.toFixed(3)}</span></div>
      <div className="flex items-center justify-between"><span className="text-slate-400">Loss</span><span>{metrics.loss.toFixed(4)}</span></div>
      <div className="flex items-center justify-between"><span className="text-slate-400">Replay buffer</span><span>{metrics.replay_buffer_used} / {metrics.replay_buffer_size}</span></div>
    </div>
  );
};

function convertActionToEvent(action: ActionsAnalyticsResponse['recent'][number]): TimelineEvent {
  const timestamp = typeof action.timestamp === 'number' ? action.timestamp : Date.now() / 1000;
  const environment = action.environment || '';
  const mode = inferModeFromStrings(action.type, environment);
  const title = action.type.replace(/_/g, ' ');
  const summary = `Reward ${(action.reward ?? 0).toFixed(2)} • Confidence ${(action.confidence ?? 0).toFixed(2)} • ${(action.execution_time ?? 0).toFixed(2)}s`;
  return {
    id: String(action.id || `${action.type}-${timestamp}`),
    timestamp,
    topic: 'agent.action',
    mode,
    title,
    summary,
    reward: action.reward,
    score: action.confidence,
    environment,
    payload: action
  };
}

function extractStreamEvents(payload: any): { topic: string; record: any }[] {
  const out: { topic: string; record: any }[] = [];
  if (!payload) return out;
  if (Array.isArray(payload)) {
    for (const record of payload) {
      out.push({ topic: record?.topic || 'agent.action', record });
    }
    return out;
  }
  if (payload.delta && typeof payload.delta === 'object') {
    for (const [topic, records] of Object.entries(payload.delta)) {
      if (Array.isArray(records)) {
        for (const record of records) out.push({ topic, record });
      }
    }
    return out;
  }
  if (payload.topics && typeof payload.topics === 'object') {
    for (const [topic, wrapped] of Object.entries<any>(payload.topics)) {
      const items = Array.isArray(wrapped?.items) ? wrapped.items : [];
      for (const record of items) out.push({ topic, record });
    }
    return out;
  }
  if (payload.topic) {
    out.push({ topic: payload.topic, record: payload });
    return out;
  }
  return out;
}

function normalizeTimelineEvent(topic: string, record: any): TimelineEvent | null {
  if (!record || typeof record !== 'object') return null;
  const event = record.event && typeof record.event === 'object' ? record.event : record;
  const baseTs = firstNumber(record.ts, record.timestamp, event.ts, event.timestamp);
  const timestamp = baseTs ? baseTs / (baseTs > 1e12 ? 1000 : 1) : Date.now() / 1000;
  const rawId = record.id || event.id || `${topic}-${timestamp}-${Math.random().toString(16).slice(2)}`;
  const action = event.action || event.name || event.type || 'Agent Event';
  const reward = firstNumber(record.reward, event.reward);
  const score = firstNumber(record.score, event.score, event.verifier_score);
  const environment = event.environment || record.environment || '';
  const mode = inferModeFromStrings(action, environment, topic);
  const status = event.status ? `Status ${event.status}` : '';
  const signature = event.signature ? `Signature ${event.signature}` : '';
  const pieces = [status, signature].filter(Boolean);
  if (typeof reward === 'number') pieces.push(`Reward ${reward.toFixed(2)}`);
  if (typeof score === 'number') pieces.push(`Score ${Number(score).toFixed(2)}`);
  if (event.message && pieces.length < 2) pieces.push(event.message);
  const summary = pieces.join(' • ');

  return {
    id: String(rawId),
    timestamp,
    topic,
    mode,
    title: titleCase(action),
    summary,
    reward: typeof reward === 'number' ? reward : undefined,
    score: typeof score === 'number' ? Number(score) : undefined,
    environment,
    payload: record
  };
}

function inferModeFromStrings(...labels: (string | undefined)[]): TimelineMode {
  const haystack = labels
    .filter((item): item is string => typeof item === 'string')
    .map((item) => item.toLowerCase())
    .join(' ');
  if (!haystack) return 'unknown';
  if (haystack.includes('train') || haystack.includes('grpo') || haystack.includes('gepa') || haystack.includes('rl')) {
    return 'training';
  }
  if (haystack.includes('verify') || haystack.includes('verifier')) {
    return 'verification';
  }
  if (haystack.includes('inference') || haystack.includes('serve') || haystack.includes('production')) {
    return 'inference';
  }
  if (haystack.includes('system') || haystack.includes('pipeline')) {
    return 'system';
  }
  return 'unknown';
}

function titleCase(label: string): string {
  return label.replace(/[_\-]+/g, ' ').replace(/\b\w/g, (micro) => micro.toUpperCase());
}

function firstNumber(...values: any[]): number | undefined {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return undefined;
}

function colorForType(label: string): string {
  const palette = ['#38bdf8', '#f97316', '#8b5cf6', '#34d399', '#facc15', '#fb7185'];
  const idx = label.split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0) % palette.length;
  return palette[idx];
}

function buildSummary(actions?: ActionsAnalyticsResponse, rl?: RlMetricsResponse) {
  const rewards = (actions?.recent || []).map((action) => action.reward ?? 0);
  const avgReward = rewards.length ? rewards.reduce((acc, value) => acc + value, 0) / rewards.length : 0;
  const positive = rewards.filter((value) => value > 0).length;
  const throughput = (actions?.recent || []).filter((item) => Date.now() / 1000 - item.timestamp < 600).length;
  const rlMetrics = rl?.metrics;
  const trainingProgress = rlMetrics?.total_episodes
    ? `${rlMetrics.current_episode}/${rlMetrics.total_episodes}`
    : 'n/a';

  return [
    {
      label: 'Recent Reward',
      value: avgReward.toFixed(2),
      detail: `${positive}/${rewards.length || 1} positive`
    },
    {
      label: '10 min Actions',
      value: throughput.toString(),
      detail: 'Live throughput sample'
    },
    {
      label: 'Trainer Loss',
      value: rlMetrics ? rlMetrics.loss.toFixed(3) : '—',
      detail: rlMetrics ? `Avg reward ${rlMetrics.avg_reward.toFixed(2)}` : 'No trainer data'
    },
    {
      label: 'Episodes',
      value: trainingProgress,
      detail: rlMetrics ? `Exploration ε ${rlMetrics.epsilon.toFixed(3)}` : 'Not running'
    }
  ];
}

function modeBadgeClass(mode: TimelineMode) {
  switch (mode) {
    case 'training':
      return 'text-emerald-300 bg-emerald-500/10 border border-emerald-400/40 px-2 py-0.5 rounded';
    case 'inference':
      return 'text-sky-300 bg-sky-500/10 border border-sky-400/40 px-2 py-0.5 rounded';
    case 'verification':
      return 'text-violet-300 bg-violet-500/10 border border-violet-400/40 px-2 py-0.5 rounded';
    case 'system':
      return 'text-amber-300 bg-amber-500/10 border border-amber-400/40 px-2 py-0.5 rounded';
    default:
      return 'text-slate-300 bg-slate-700/40 border border-slate-500/50 px-2 py-0.5 rounded';
  }
}

export default AgentPulsePage;
