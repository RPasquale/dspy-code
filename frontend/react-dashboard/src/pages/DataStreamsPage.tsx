import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Card from '../components/Card';
import TopicMultiSelect from '../components/TopicMultiSelect';
import TimeRangePicker, { TimeRange } from '../components/TimeRangePicker';
import EventsTable from '../components/EventsTable';
import { api } from '../api/client';
import type { KafkaTopicsResponse, StreamMetricsResponse, BusMetricsResponse } from '../api/types';

const DEFAULT_RANGE: TimeRange = {
  label: 'Last 15m',
  since: Math.floor(Date.now() / 1000) - 15 * 60,
  until: Math.floor(Date.now() / 1000),
  live: true
};

const DataStreamsPage = () => {
  const [topics, setTopics] = useState<string[]>(['agent.action']);
  const [range, setRange] = useState<TimeRange>(DEFAULT_RANGE);
  const [search, setSearch] = useState('');
  const [fields, setFields] = useState('');
  const [rows, setRows] = useState<any[]>([]);
  const streamRef = useRef<EventSource | null>(null);

  const { data: streamMetrics } = useQuery({ queryKey: ['stream-metrics'], queryFn: api.getStreamMetrics, refetchInterval: 15000 });
  const { data: busMetrics } = useQuery({ queryKey: ['bus-metrics'], queryFn: api.getBusMetrics, refetchInterval: 20000 });
  const { data: kafkaTopics } = useQuery({ queryKey: ['kafka-topics'], queryFn: api.getKafkaTopics, refetchInterval: 30000 });

  const paramsKey = useMemo(
    () => [topics.join(','), range.since, range.until, range.live ? 1 : 0, search, fields].join('|'),
    [topics, range.since, range.until, range.live, search, fields]
  );

  useEffect(() => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }

    let cancelled = false;
    const fieldList = fields.split(',').map((token) => token.trim()).filter(Boolean);

    if (range.live) {
      const es = api.streamEvents(
        {
          topics,
          follow: true,
          q: search || undefined,
          keys: fieldList.length ? fieldList : undefined,
          limit: 500
        },
        (payload) => {
          if (cancelled) return;
          const events = flattenStreamPayload(payload);
          if (!events.length) return;
          setRows((prev) => {
            const next = [...prev, ...events].slice(-1200);
            return next;
          });
        }
      );
      streamRef.current = es;
      return () => {
        cancelled = true;
        es.close();
        streamRef.current = null;
      };
    }

    // Not live: fetch snapshots for each topic
    let active = true;
    (async () => {
      const collected: any[] = [];
      for (const topic of topics) {
        try {
          const res = await api.getEventsTailEx(topic, {
            limit: 400,
            q: search || undefined,
            keys: fieldList.length ? fieldList : undefined,
            since: range.since,
            until: range.until
          });
          collected.push(...(res.items || []));
        } catch {
          // ignore fetch errors per topic
        }
      }
      if (active) {
        setRows(collected);
      }
    })();

    return () => {
      active = false;
    };
  }, [paramsKey]);

  useEffect(() => () => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
  }, []);

  const cards = useMemo(() => buildSummaryCards(streamMetrics, busMetrics), [streamMetrics?.timestamp, busMetrics?.timestamp]);
  const topicRows = useMemo(() => buildTopicRows(kafkaTopics), [kafkaTopics?.timestamp]);
  const alerts = useMemo(() => buildAlerts(streamMetrics, busMetrics), [streamMetrics?.timestamp, busMetrics?.timestamp]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3">
        <h1 className="text-3xl font-semibold text-white">Data Sources</h1>
        <p className="text-sm text-slate-300 max-w-3xl">
          Monitor what your agent is learning from, data quality, and training progress. See the information sources that are improving your AI's performance.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => (
          <Card key={card.label} className="p-4" variant="outlined">
            <div className="text-sm text-slate-400">{card.label}</div>
            <div className="mt-2 text-2xl font-semibold text-white">{card.value}</div>
            {card.detail && <div className="mt-1 text-xs text-slate-500">{card.detail}</div>}
          </Card>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2" title="Stream Explorer" subtitle="Follow live topics or replay history">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-slate-300">
                <span>Topics</span>
                <TopicMultiSelect value={topics} onChange={setTopics} />
              </div>
              <div className="flex items-center gap-2 text-sm text-slate-300">
                <span>Window</span>
                <TimeRangePicker
                  value={range}
                  onChange={(value) => {
                    setRange(value);
                  }}
                />
              </div>
              <input
                className="min-w-[12rem] rounded border border-slate-600 bg-slate-900 px-3 py-1 text-sm text-slate-200"
                placeholder="Search (regex)"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
              <input
                className="min-w-[10rem] rounded border border-slate-600 bg-slate-900 px-3 py-1 text-sm text-slate-200"
                placeholder="Fields (dot paths)"
                value={fields}
                onChange={(event) => setFields(event.target.value)}
              />
              <button
                className={`ml-auto rounded-full px-3 py-1 text-xs font-medium ${
                  range.live ? 'bg-emerald-500 text-slate-900' : 'border border-slate-600 bg-slate-900 text-slate-200'
                }`}
                onClick={() => setRange((prev) => ({ ...prev, live: !prev.live }))}
              >
                {range.live ? 'Streaming…' : 'Follow stream'}
              </button>
              <a
                className="rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-200 hover:border-emerald-500 hover:text-white"
                href={buildExportUrl(topics, range, search, fieldList(fields))}
                target="_blank"
                rel="noreferrer"
              >
                Download JSON
              </a>
            </div>
            <EventsTable rows={rows} />
          </div>
        </Card>
        <Card title="Topic Health" subtitle="Kafka / stream backlog">
          <TopicTable rows={topicRows} />
        </Card>
      </div>

      <Card title="Alerts & Backpressure" subtitle="DLQ activity, queue depths, and pipeline alerts">
        <AlertList alerts={alerts} />
      </Card>
    </div>
  );
};

interface SummaryCard {
  label: string;
  value: string;
  detail?: string;
}

function buildSummaryCards(stream?: StreamMetricsResponse, bus?: BusMetricsResponse): SummaryCard[] {
  const kafka = stream?.current_metrics.kafka_throughput;
  const pipe = stream?.current_metrics.data_pipeline;
  const spark = stream?.current_metrics.spark_streaming;
  const network = stream?.current_metrics.network_io;
  return [
    {
      label: 'Learning Data Rate',
      value: kafka ? `${formatNumber(kafka.messages_per_second)} events/s` : '—',
      detail: kafka ? `Processing ${formatBytes(kafka.bytes_per_second)}/s of training data` : undefined
    },
    {
      label: 'Training Progress',
      value: pipe ? `${formatNumber(pipe.output_rate)} processed/s` : '—',
      detail: pipe ? `${formatNumber(pipe.input_rate)} new examples/s · ${formatNumber(pipe.error_rate)} issues` : undefined
    },
    {
      label: 'Processing Speed',
      value: spark ? `${spark.processing_time.toFixed(2)}s` : '—',
      detail: spark ? `Processing ${spark.records_per_batch} examples per batch` : undefined
    },
    {
      label: 'Data Sources',
      value: network ? `${formatNumber(network.connections_active)} active` : '—',
      detail: network ? `Receiving ${formatBytes(network.bytes_in_per_sec)}/s of new data` : undefined
    }
  ];
}

function buildTopicRows(topics?: KafkaTopicsResponse) {
  if (!topics) return [] as const;
  return (topics.topics || []).slice().sort((a, b) => b.messages_per_minute - a.messages_per_minute).map((topic) => ({
    name: topic.name,
    rate: topic.messages_per_minute,
    lag: topic.consumer_lag,
    size: topic.size_bytes,
    partitions: topic.partitions,
    retention: topic.retention_ms
  }));
}

function buildAlerts(stream?: StreamMetricsResponse, bus?: BusMetricsResponse) {
  const alerts = [] as { ts: string; level: string; message: string }[];
  if (stream?.alerts) {
    for (const alert of stream.alerts) {
      alerts.push({ ts: alert.timestamp, level: alert.level, message: alert.message });
    }
  }
  if (bus?.alerts) {
    for (const alert of bus.alerts) {
      alerts.push({ ts: new Date(alert.timestamp * 1000).toISOString(), level: alert.level, message: alert.message });
    }
  }
  if (bus?.dlq?.total) {
    alerts.push({ ts: new Date().toISOString(), level: 'warning', message: `DLQ contains ${formatNumber(bus.dlq.total)} messages` });
  }
  return alerts.sort((a, b) => (a.ts < b.ts ? 1 : -1));
}

const TopicTable = ({ rows }: { rows: ReturnType<typeof buildTopicRows> }) => (
  <div className="overflow-auto">
    <table className="min-w-full divide-y divide-slate-700 text-sm">
      <thead className="bg-slate-900/60 text-xs uppercase tracking-wide text-slate-400">
        <tr>
          <th className="px-4 py-3 text-left">Topic</th>
          <th className="px-4 py-3 text-left">Rate (msg/min)</th>
          <th className="px-4 py-3 text-left">Lag</th>
          <th className="px-4 py-3 text-left">Size</th>
          <th className="px-4 py-3 text-left">Partitions</th>
          <th className="px-4 py-3 text-left">Retention</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-800">
        {rows.map((row) => (
          <tr key={row.name} className="hover:bg-slate-900/30">
            <td className="px-4 py-3 text-slate-200">{row.name}</td>
            <td className="px-4 py-3 text-slate-200">{formatNumber(row.rate)}</td>
            <td className="px-4 py-3 text-slate-200">{formatNumber(row.lag)}</td>
            <td className="px-4 py-3 text-slate-200">{formatBytes(row.size)}</td>
            <td className="px-4 py-3 text-slate-200">{row.partitions}</td>
            <td className="px-4 py-3 text-slate-200">{formatDuration(row.retention)}</td>
          </tr>
        ))}
        {!rows.length && (
          <tr>
            <td colSpan={6} className="px-4 py-6 text-center text-sm text-slate-500">No topic metrics available.</td>
          </tr>
        )}
      </tbody>
    </table>
  </div>
);

const AlertList = ({ alerts }: { alerts: ReturnType<typeof buildAlerts> }) => (
  <div className="space-y-2 text-sm text-slate-200">
    {alerts.map((alert, index) => (
      <div
        key={`${alert.ts}-${index}`}
        className={`flex items-start gap-3 rounded border px-3 py-2 ${
          alert.level === 'error'
            ? 'border-rose-500/50 bg-rose-500/10'
            : alert.level === 'warning'
            ? 'border-amber-500/40 bg-amber-500/10'
            : 'border-emerald-500/30 bg-emerald-500/10'
        }`}
      >
        <span className="text-xs text-slate-300 w-32 shrink-0">{new Date(alert.ts).toLocaleString()}</span>
        <span className="text-xs uppercase tracking-wide text-slate-400">{alert.level}</span>
        <span className="text-sm text-slate-100">{alert.message}</span>
      </div>
    ))}
    {!alerts.length && <div className="text-sm text-slate-400">No alerts reported.</div>}
  </div>
);

function formatNumber(value?: number) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '0';
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return value.toFixed(1);
}

function formatBytes(value?: number) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let idx = 0;
  let current = value;
  while (current >= 1024 && idx < units.length - 1) {
    current /= 1024;
    idx += 1;
  }
  return `${current.toFixed(1)} ${units[idx]}`;
}

function formatDuration(ms?: number) {
  if (typeof ms !== 'number' || ms <= 0) return '—';
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const minutes = seconds / 60;
  if (minutes < 60) return `${minutes.toFixed(0)}m`;
  const hours = minutes / 60;
  if (hours < 48) return `${hours.toFixed(0)}h`;
  const days = hours / 24;
  return `${days.toFixed(0)}d`;
}

function flattenStreamPayload(payload: any): any[] {
  const out: any[] = [];
  if (!payload) return out;
  if (Array.isArray(payload)) return payload;
  if (payload.delta && typeof payload.delta === 'object') {
    for (const arr of Object.values<any>(payload.delta)) {
      if (Array.isArray(arr)) out.push(...arr);
    }
    return out;
  }
  if (payload.topics && typeof payload.topics === 'object') {
    for (const entry of Object.values<any>(payload.topics)) {
      const items = Array.isArray(entry?.items) ? entry.items : [];
      out.push(...items);
    }
    return out;
  }
  if (payload.items && Array.isArray(payload.items)) return payload.items;
  if (payload.event) return [payload.event];
  return [];
}

function fieldList(fields: string) {
  return fields
    .split(',')
    .map((token) => token.trim())
    .filter(Boolean)
    .map((field) => `&field=${encodeURIComponent(field)}`)
    .join('');
}

function buildExportUrl(topics: string[], range: TimeRange, search: string, fields: string) {
  const params = new URLSearchParams();
  if (topics.length) params.set('topics', topics.join(','));
  params.set('limit', '1000');
  if (range.since) params.set('since', String(range.since));
  if (range.until) params.set('until', String(range.until));
  if (search) params.set('q', search);
  const base = `/api/events/export?${params.toString()}${fields}`;
  return `${base}&download=1`;
}

export default DataStreamsPage;
