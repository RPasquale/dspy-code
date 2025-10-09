import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import Card from '../components/Card';
import StatusPill from '../components/StatusPill';
import SystemResourcesCard from '../components/SystemResourcesCard';
import MiniSystemCharts from '../components/MiniSystemCharts';
import MeshStatusCard from '../components/MeshStatusCard';
import GuardSettingsPanel from '../components/GuardSettingsPanel';
import CapacityPanel from '../components/CapacityPanel';
import PipelineStatus from '../components/PipelineStatus';
import AgentBrainPanel from '../components/AgentBrainPanel';
import DebugTracePanel from '../components/DebugTracePanel';
import { api } from '../api/client';
import type { StatusResponse, BusMetricsResponse } from '../api/types';

const SystemOverviewPage = () => {
  const { data: status } = useQuery({ queryKey: ['status'], queryFn: api.getStatus, refetchInterval: 15000 });
  const { data: busMetrics } = useQuery({ queryKey: ['bus-metrics'], queryFn: api.getBusMetrics, refetchInterval: 30000 });
  const { data: reddbHealth } = useQuery({ queryKey: ['reddb-health'], queryFn: api.getReddbHealth, refetchInterval: 30000 });
  const { data: overview } = useQuery({ queryKey: ['overview'], queryFn: api.getOverview, refetchInterval: 20000 });

  const cards = useMemo(() => buildSummary(status, busMetrics, reddbHealth), [status, busMetrics, reddbHealth]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3">
        <h1 className="text-3xl font-semibold text-white">Health Status</h1>
        <p className="text-sm text-slate-300 max-w-3xl">
          Monitor the health and performance of your AI training system. Ensure everything is running smoothly for optimal learning.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => (
          <Card key={card.name} className="p-4" variant="outlined">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-slate-400">{card.name}</div>
                <div className="mt-2 text-2xl font-semibold text-white">{card.status}</div>
                {card.detail && <div className="mt-1 text-xs text-slate-500">{card.detail}</div>}
              </div>
              <StatusPill status={card.state} />
            </div>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2" title="System Performance" subtitle="Computing resources and processing capacity">
          <SystemResourcesCard />
        </Card>
        <Card title="Learning Performance" subtitle="Training speed, accuracy, and progress trends">
          <MiniSystemCharts />
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Agent Brain" subtitle="Current policies, rewards, and tool weights">
          <AgentBrainPanel />
        </Card>
        <Card title="Pipeline Status" subtitle="Orchestration, ingest, and streaming">
          <PipelineStatus />
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2" title="Mesh & Capacity" subtitle="Distributed services and guardrails">
          <div className="grid gap-4 md:grid-cols-2">
            <MeshStatusCard />
            <CapacityPanel />
          </div>
        </Card>
        <Card title="Guardrails" subtitle="Access controls and pending approvals">
          <GuardSettingsPanel />
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Debug Trace" subtitle="Latest debug beacons">
          <DebugTracePanel />
        </Card>
        <Card title="Overview Snapshot" subtitle="Aggregated metrics across services">
          <OverviewSnapshot overview={overview} />
        </Card>
      </div>
    </div>
  );
};

function buildSummary(status?: StatusResponse, bus?: BusMetricsResponse, reddb?: { status: string; recent_actions: number; recent_training: number; signatures: number }) {
  return [
    {
      name: 'AI Learning Engine',
      status: status?.agent?.details || 'Ready',
      state: status?.agent?.status,
      detail: status?.learning_active ? 'Currently learning' : 'Ready to learn'
    },
    {
      name: 'Data Pipeline',
      status: bus ? `${bus.dlq.total === 0 ? 'Healthy' : `${bus.dlq.total} issues`}` : '—',
      state: bus && bus.dlq.total > 0 ? 'warning' : (status?.kafka?.status ?? 'unknown'),
      detail: bus ? `${Object.keys(bus.bus.topics || {}).length} data sources` : undefined
    },
    {
      name: 'Knowledge Base',
      status: reddb ? `${reddb.signatures} learned patterns` : '—',
      state: (reddb?.status as StatusResponse['agent']['status']) || status?.reddb?.status,
      detail: reddb ? `${reddb.recent_actions} recent actions · ${reddb.recent_training} training sessions` : undefined
    },
    {
      name: 'Processing Power',
      status: status?.spark?.details || 'Available',
      state: status?.spark?.status,
      detail: status?.containers?.details
    }
  ];
}

const OverviewSnapshot = ({ overview }: { overview?: { status: StatusResponse; bus: BusMetricsResponse } | undefined }) => {
  if (!overview) {
    return <div className="text-sm text-slate-400">Overview aggregation unavailable.</div>;
  }
  const statuses = overview.status;
  const bus = overview.bus;
  return (
    <div className="space-y-3 text-sm text-slate-200">
      <div className="flex items-center justify-between"><span className="text-slate-400">Agent</span><StatusPill status={statuses.agent.status} /></div>
      <div className="flex items-center justify-between"><span className="text-slate-400">Kafka</span><StatusPill status={statuses.kafka.status} /></div>
      <div className="flex items-center justify-between"><span className="text-slate-400">RedDB</span><StatusPill status={statuses.reddb?.status} /></div>
      <div className="flex items-center justify-between"><span className="text-slate-400">Spark</span><StatusPill status={statuses.spark?.status} /></div>
      <div className="rounded border border-slate-700 bg-slate-900/60 p-3 text-xs text-slate-300">
        <div className="text-slate-400 uppercase tracking-wide mb-2">Bus utilisation</div>
        <div>DLQ total: {bus.dlq.total}</div>
        <div>Topics: {Object.keys(bus.bus.topics || {}).length}</div>
        <div>Alerts: {bus.alerts.length}</div>
      </div>
    </div>
  );
};

export default SystemOverviewPage;
