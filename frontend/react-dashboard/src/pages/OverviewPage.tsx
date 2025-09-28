import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import { ensureChartsRegistered } from '../lib/registerCharts';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import Card from '../components/Card';
import SystemResourcesCard from '@/components/SystemResourcesCard';
import GuardSettingsPanel from '@/components/GuardSettingsPanel';
import AdminSettingsDrawer from '@/components/AdminSettingsDrawer';
import DebugTracePanel from '@/components/DebugTracePanel';
import DevCyclePanel from '@/components/DevCyclePanel';
import MiniSystemCharts from '@/components/MiniSystemCharts';
import CleanupPanel from '@/components/CleanupPanel';
import AgentBrainPanel from '@/components/AgentBrainPanel';
import CapacityPanel from '@/components/CapacityPanel';
import QuickCommands from '@/components/QuickCommands';
import MeshStatusCard from '@/components/MeshStatusCard';
import EmbeddingsStatus from '@/components/EmbeddingsStatus';
import PipelineStatus from '@/components/PipelineStatus';
import StatusPill from '../components/StatusPill';
import { api } from '../api/client';
import type { StatusResponse } from '../api/types';

const statusEntries = (status?: StatusResponse) => {
  if (!status) return [];
  return [
    { id: 'Agent', data: status.agent },
    { id: 'Ollama', data: status.ollama },
    { id: 'Kafka', data: status.kafka },
    { id: 'RedDB', data: status.reddb },
    { id: 'Spark', data: status.spark },
    { id: 'Embeddings', data: status.embeddings },
    { id: 'Pipeline', data: status.pipeline },
  ];
};

const OverviewPage = () => {
  const queryClient = useQueryClient();
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('24h');
  
  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['status'],
    queryFn: api.getStatus,
    refetchInterval: 10000,
  });

  const { data: busData } = useQuery({
    queryKey: ['bus-metrics'],
    queryFn: api.getBusMetrics,
    refetchInterval: 10000,
  });

  const { data: systemResources } = useQuery({
    queryKey: ['system-resources'],
    queryFn: api.getSystemResources,
    refetchInterval: 5000,
  });

  useEffect(() => {
    ensureChartsRegistered();
  }, []);

  const statusData = statusEntries(status);
  const overviewLoading = statusLoading;

  return (
    <>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Dashboard Overview</h1>
          <p className="text-slate-600">Monitor your DSPy training system status and performance</p>
        </div>

        {/* Status Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {statusData.map(({ id, data }) => (
            <Card key={id} variant="outlined" className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-medium text-slate-600">{id}</h3>
                  <div className="mt-2">
                    <StatusPill status={data?.status} />
                  </div>
                </div>
                <div className="text-2xl">
                  {id === 'Agent' && '🤖'}
                  {id === 'Ollama' && '🦙'}
                  {id === 'Kafka' && '📊'}
                  {id === 'RedDB' && '🗄️'}
                  {id === 'Spark' && '⚡'}
                  {id === 'Embeddings' && '🧠'}
                  {id === 'Pipeline' && '🔧'}
                </div>
              </div>
            </Card>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-8">
            {/* System Resources */}
            <Card title="System Resources" subtitle="CPU, Memory, and Storage utilization">
              <SystemResourcesCard />
            </Card>

            {/* Agent Brain */}
            <Card title="Agent Brain" subtitle="Policy rules and tool rewards">
              <AgentBrainPanel />
            </Card>

            {/* Quick Commands */}
            <Card title="Quick Actions" subtitle="Common system operations">
              <QuickCommands />
            </Card>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            {/* System Status */}
            <Card title="System Status" subtitle="Real-time system health">
              <div className="space-y-4">
                {overviewLoading ? (
                  <div className="space-y-3">
                    <div className="h-4 bg-slate-200 rounded animate-pulse"></div>
                    <div className="h-4 bg-slate-200 rounded animate-pulse"></div>
                    <div className="h-4 bg-slate-200 rounded animate-pulse"></div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-600">Message Bus</span>
                      <StatusPill status={busData ? 'ok' : 'error'} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-600">Database</span>
                      <StatusPill status={status?.reddb?.status} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-600">Agent</span>
                      <StatusPill status={status?.agent?.status} />
                    </div>
                  </div>
                )}
              </div>
            </Card>

            {/* System Charts */}
            <Card title="Performance Metrics" subtitle="System performance over time">
              <MiniSystemCharts />
            </Card>

            {/* Capacity Panel */}
            <Card title="Capacity" subtitle="Resource capacity and limits">
              <CapacityPanel />
            </Card>

            {/* Mesh Status */}
            <Card title="Mesh Status" subtitle="Distributed system status">
              <MeshStatusCard />
            </Card>
          </div>
        </div>

        {/* Advanced Panels */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card title="Guard Settings" subtitle="Security and access controls">
            <GuardSettingsPanel />
          </Card>

          <Card title="Debug Trace" subtitle="System debugging and tracing">
            <DebugTracePanel />
          </Card>
        </div>

        {/* Development Tools */}
        <div className="mt-8">
          <Card title="Development Tools" subtitle="Development and testing utilities">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <DevCyclePanel />
              <CleanupPanel />
            </div>
          </Card>
        </div>
      </div>
    </>
  );
};

export default OverviewPage;