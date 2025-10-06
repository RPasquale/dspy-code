import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Line } from 'react-chartjs-2';
import Card from '../components/Card';
import { api } from '../api/client';
import type {
  LearningMetricsResponse,
  PerformanceHistoryResponse,
  RlMetricsResponse,
  SignaturesResponse,
  VerifiersResponse,
  ActionsAnalyticsResponse
} from '../api/types';
import { ensureChartsRegistered } from '../lib/registerCharts';

ensureChartsRegistered();

const PerformanceInsightsPage = () => {
  const { data: signatures } = useQuery({ queryKey: ['signatures'], queryFn: api.getSignatures, refetchInterval: 20000 });
  const { data: verifiers } = useQuery({ queryKey: ['verifiers'], queryFn: api.getVerifiers, refetchInterval: 20000 });
  const { data: learning } = useQuery({ queryKey: ['learning-metrics'], queryFn: api.getLearningMetrics, refetchInterval: 30000 });
  const { data: rlMetrics } = useQuery({ queryKey: ['rl-metrics'], queryFn: api.getRlMetrics, refetchInterval: 15000 });
  const { data: performanceHistory } = useQuery({ queryKey: ['performance-history', '7d'], queryFn: () => api.getPerformanceHistory('7d'), refetchInterval: 60000 });
  const { data: actionAnalytics } = useQuery({ queryKey: ['actions-analytics', 'performance'], queryFn: () => api.getActionsAnalytics(200, '24h'), refetchInterval: 20000 });

  const summaryCards = useMemo(() => buildSummary(signatures, verifiers, rlMetrics, learning, actionAnalytics), [signatures, verifiers, rlMetrics, learning, actionAnalytics]);
  const learningSeries = useMemo(() => buildLearningSeries(learning), [learning?.timestamp]);
  const rewardSeries = useMemo(() => buildRewardSeries(rlMetrics), [rlMetrics?.timestamp]);
  const systemSeries = useMemo(() => buildSystemSeries(performanceHistory), [performanceHistory?.timestamp]);
  const confidenceSeries = useMemo(() => buildConfidenceSeries(actionAnalytics), [actionAnalytics?.timestamp]);
  const topSignatures = useMemo(() => selectTopSignatures(signatures), [signatures?.timestamp]);
  const verifierRows = useMemo(() => selectVerifierRows(verifiers), [verifiers?.timestamp]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3">
        <h1 className="text-3xl font-semibold text-white">Performance Command Center</h1>
        <p className="text-sm text-slate-300 max-w-3xl">
          Trace how the agent learns, measure verification quality, and compare reward signals across DSPy signature GEPA and RL training cycles.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {summaryCards.map((card) => (
          <Card key={card.label} className="p-4" variant="outlined">
            <div className="text-sm text-slate-400">{card.label}</div>
            <div className="mt-2 text-2xl font-semibold text-white">{card.value}</div>
            {card.trend && <div className="mt-1 text-xs text-emerald-400">{card.trend}</div>}
            {card.detail && <div className="mt-1 text-xs text-slate-500">{card.detail}</div>}
          </Card>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2" title="DSPy Signature Learning" subtitle="Performance, training, and validation accuracy">
          <div className="h-64">
            <Line data={learningSeries.data} options={learningSeries.options} />
          </div>
        </Card>
        <Card title="RL Reward History" subtitle="Episode rewards from the trainer">
          <div className="h-64">
            <Line data={rewardSeries.data} options={rewardSeries.options} />
          </div>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2" title="Signature Leaderboard" subtitle="Top performing DSPy signatures (GEPA)">
          <SignatureTable rows={topSignatures} />
        </Card>
        <Card title="Verifier Quality" subtitle="Verifier accuracy and coverage">
          <VerifierTable rows={verifierRows} />
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2" title="Success & Error Rates" subtitle="System level throughput and quality">
          <div className="h-64">
            <Line data={systemSeries.data} options={systemSeries.options} />
          </div>
        </Card>
        <Card title="Action Confidence" subtitle="Recent confidence distribution">
          <div className="h-64">
            <Line data={confidenceSeries.data} options={confidenceSeries.options} />
          </div>
        </Card>
      </div>

      <Card title="RL Diagnostics" subtitle="Current trainer state">
        <RLDiagnostics rl={rlMetrics} />
      </Card>
    </div>
  );
};

interface SummaryCard {
  label: string;
  value: string;
  detail?: string;
  trend?: string;
}

function buildSummary(
  signatures?: SignaturesResponse,
  verifiers?: VerifiersResponse,
  rl?: RlMetricsResponse,
  learning?: LearningMetricsResponse,
  actions?: ActionsAnalyticsResponse
): SummaryCard[] {
  const sigs = signatures?.signatures || [];
  const ver = verifiers?.verifiers || [];
  const avgPerf = sigs.length
    ? sigs.reduce((acc, sig) => acc + (sig.performance ?? 0), 0) / sigs.length
    : 0;
  const bestSig = sigs.slice().sort((a, b) => (b.performance ?? 0) - (a.performance ?? 0))[0];
  const avgVerifier = ver.length
    ? ver.reduce((acc, item) => acc + (item.accuracy ?? 0), 0) / ver.length
    : 0;
  const rlMetrics = rl?.metrics;
  const learningStats = learning?.learning_stats;
  const rewards = (actions?.recent || []).map((item) => item.reward ?? 0);
  const avgReward = rewards.length ? rewards.reduce((acc, value) => acc + value, 0) / rewards.length : 0;
  const pctPositive = rewards.length ? (rewards.filter((value) => value > 0).length / rewards.length) * 100 : 0;

  return [
    {
      label: 'Avg Signature Performance',
      value: `${avgPerf.toFixed(1)}%`,
      trend: bestSig ? `Top: ${bestSig.name} · ${(bestSig.performance ?? 0).toFixed(1)}%` : undefined,
      detail: learningStats ? `${learningStats.successful_optimizations}/${learningStats.total_training_examples} successful updates` : undefined
    },
    {
      label: 'Verifier Accuracy',
      value: `${avgVerifier.toFixed(1)}%`,
      detail: `Across ${ver.length} verifiers`
    },
    {
      label: 'RL Avg Reward',
      value: rlMetrics ? rlMetrics.avg_reward.toFixed(3) : '—',
      trend: rlMetrics ? `Best ${rlMetrics.best_reward.toFixed(3)} / Worst ${rlMetrics.worst_reward.toFixed(3)}` : undefined,
      detail: rlMetrics ? `Exploration ε ${rlMetrics.epsilon.toFixed(3)}` : undefined
    },
    {
      label: 'Recent Action Reward',
      value: avgReward.toFixed(2),
      detail: `${pctPositive.toFixed(0)}% positive last 24h`
    }
  ];
}

function buildLearningSeries(learning?: LearningMetricsResponse) {
  const timestamps = learning?.performance_over_time?.timestamps || [];
  const labels = timestamps.map((ts) => formatTimestamp(ts));
  return {
    data: {
      labels,
      datasets: [
        {
          label: 'Overall Performance',
          data: learning?.performance_over_time?.overall_performance || [],
          borderColor: '#22d3ee',
          backgroundColor: 'rgba(34, 211, 238, 0.18)',
          tension: 0.35,
          fill: true,
          pointRadius: 0
        },
        {
          label: 'Training Accuracy',
          data: learning?.performance_over_time?.training_accuracy || [],
          borderColor: '#34d399',
          backgroundColor: 'rgba(52, 211, 153, 0.15)',
          tension: 0.35,
          fill: true,
          pointRadius: 0
        },
        {
          label: 'Validation Accuracy',
          data: learning?.performance_over_time?.validation_accuracy || [],
          borderColor: '#fbbf24',
          backgroundColor: 'rgba(251, 191, 36, 0.12)',
          tension: 0.35,
          fill: true,
          pointRadius: 0
        }
      ]
    },
    options: defaultLineOptions
  };
}

function buildRewardSeries(rl?: RlMetricsResponse) {
  const history = rl?.reward_history || [];
  const labels = history.map((item) => `Ep ${item.episode}`);
  return {
    data: {
      labels,
      datasets: [
        {
          label: 'Reward',
          data: history.map((item) => item.reward),
          borderColor: '#f472b6',
          backgroundColor: 'rgba(244, 114, 182, 0.15)',
          tension: 0.3,
          fill: true,
          pointRadius: 0
        }
      ]
    },
    options: defaultLineOptions
  };
}

function buildSystemSeries(history?: PerformanceHistoryResponse) {
  const labels = (history?.timestamps || []).map((ts) => formatTimestamp(ts));
  const successRates = history?.metrics.success_rates || [];
  const errorRates = history?.metrics.error_rates || [];
  const throughput = history?.metrics.throughput || [];
  return {
    data: {
      labels,
      datasets: [
        {
          label: 'Success Rate',
          data: successRates,
          borderColor: '#34d399',
          backgroundColor: 'rgba(52, 211, 153, 0.08)',
          tension: 0.3,
          yAxisID: 'y'
        },
        {
          label: 'Error Rate',
          data: errorRates,
          borderColor: '#f87171',
          backgroundColor: 'rgba(248, 113, 113, 0.08)',
          tension: 0.3,
          yAxisID: 'y'
        },
        {
          label: 'Throughput',
          data: throughput,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.08)',
          tension: 0.3,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      ...defaultLineOptions,
      scales: {
        y: {
          position: 'left',
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(148, 163, 184, 0.12)' }
        },
        y1: {
          position: 'right',
          ticks: { color: '#94a3b8' },
          grid: { drawOnChartArea: false }
        },
        x: defaultLineOptions.scales?.x
      }
    }
  };
}

function buildConfidenceSeries(actions?: ActionsAnalyticsResponse) {
  const recent = (actions?.recent || []).slice(-80);
  const labels = recent.map((item) => new Date(item.timestamp * 1000).toLocaleTimeString());
  return {
    data: {
      labels,
      datasets: [
        {
          label: 'Confidence',
          data: recent.map((item) => item.confidence ?? 0),
          borderColor: '#f97316',
          backgroundColor: 'rgba(249, 115, 22, 0.12)',
          tension: 0.35,
          fill: true,
          pointRadius: 0
        }
      ]
    },
    options: defaultLineOptions
  };
}

function selectTopSignatures(signatures?: SignaturesResponse) {
  const rows = (signatures?.signatures || []).slice().sort((a, b) => (b.performance ?? 0) - (a.performance ?? 0));
  return rows.slice(0, 10);
}

function selectVerifierRows(verifiers?: VerifiersResponse) {
  const rows = (verifiers?.verifiers || []).slice().sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0));
  return rows.slice(0, 10);
}

const defaultLineOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#cbd5f5' } } },
  scales: {
    x: {
      ticks: { color: '#94a3b8', maxRotation: 0, minRotation: 0 },
      grid: { display: false }
    },
    y: {
      ticks: { color: '#94a3b8' },
      grid: { color: 'rgba(148, 163, 184, 0.12)' }
    }
  }
};

function formatTimestamp(ts: string | number) {
  if (!ts) return '';
  const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  return `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:00`;
}

const SignatureTable = ({ rows }: { rows: ReturnType<typeof selectTopSignatures> }) => (
  <div className="overflow-auto">
    <table className="min-w-full divide-y divide-slate-700 text-sm">
      <thead className="bg-slate-900/60 text-xs uppercase tracking-wide text-slate-400">
        <tr>
          <th className="px-4 py-3 text-left">Signature</th>
          <th className="px-4 py-3 text-left">Performance</th>
          <th className="px-4 py-3 text-left">Success</th>
          <th className="px-4 py-3 text-left">Iterations</th>
          <th className="px-4 py-3 text-left">Type</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-800">
        {rows.map((row) => (
          <tr key={row.name} className="hover:bg-slate-900/30">
            <td className="px-4 py-3 text-slate-200">{row.name}</td>
            <td className="px-4 py-3 text-emerald-300">{row.performance.toFixed(1)}%</td>
            <td className="px-4 py-3 text-slate-200">{(row.success_rate ?? 0).toFixed(1)}%</td>
            <td className="px-4 py-3 text-slate-400">{row.iterations}</td>
            <td className="px-4 py-3 text-slate-300 capitalize">{row.type}</td>
          </tr>
        ))}
        {!rows.length && (
          <tr>
            <td colSpan={5} className="px-4 py-6 text-center text-sm text-slate-500">No signature data available.</td>
          </tr>
        )}
      </tbody>
    </table>
  </div>
);

const VerifierTable = ({ rows }: { rows: ReturnType<typeof selectVerifierRows> }) => (
  <div className="overflow-auto">
    <table className="min-w-full divide-y divide-slate-700 text-sm">
      <thead className="bg-slate-900/60 text-xs uppercase tracking-wide text-slate-400">
        <tr>
          <th className="px-4 py-3 text-left">Verifier</th>
          <th className="px-4 py-3 text-left">Accuracy</th>
          <th className="px-4 py-3 text-left">Checks</th>
          <th className="px-4 py-3 text-left">Issues</th>
          <th className="px-4 py-3 text-left">Last Run</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-800">
        {rows.map((row) => (
          <tr key={row.name} className="hover:bg-slate-900/30">
            <td className="px-4 py-3 text-slate-200">{row.name}</td>
            <td className="px-4 py-3 text-emerald-300">{row.accuracy.toFixed(1)}%</td>
            <td className="px-4 py-3 text-slate-200">{row.checks_performed}</td>
            <td className="px-4 py-3 text-slate-200">{row.issues_found}</td>
            <td className="px-4 py-3 text-slate-400">{row.last_run}</td>
          </tr>
        ))}
        {!rows.length && (
          <tr>
            <td colSpan={5} className="px-4 py-6 text-center text-sm text-slate-500">No verifier data available.</td>
          </tr>
        )}
      </tbody>
    </table>
  </div>
);

const RLDiagnostics = ({ rl }: { rl?: RlMetricsResponse }) => {
  if (!rl) {
    return <div className="text-sm text-slate-400">Trainer metrics unavailable.</div>;
  }
  const metrics = rl.metrics;
  const env = rl.environment_info;
  const actionStats = Object.entries(rl.action_stats || {})
    .sort(([, a], [, b]) => Number(b) - Number(a))
    .slice(0, 6);
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div className="space-y-2 text-sm text-slate-200">
        <div className="flex items-center justify-between"><span className="text-slate-400">Training status</span><span>{metrics.training_status}</span></div>
        <div className="flex items-center justify-between"><span className="text-slate-400">Episodes</span><span>{metrics.current_episode} / {metrics.total_episodes}</span></div>
        <div className="flex items-center justify-between"><span className="text-slate-400">Average reward</span><span>{metrics.avg_reward.toFixed(3)}</span></div>
        <div className="flex items-center justify-between"><span className="text-slate-400">Learning rate</span><span>{metrics.learning_rate.toExponential(2)}</span></div>
        <div className="flex items-center justify-between"><span className="text-slate-400">Q-value μ</span><span>{metrics.q_value_mean.toFixed(3)}</span></div>
        <div className="flex items-center justify-between"><span className="text-slate-400">Replay buffer</span><span>{metrics.replay_buffer_used} / {metrics.replay_buffer_size}</span></div>
      </div>
      <div className="space-y-3">
        <div className="rounded border border-slate-700 bg-slate-900/60 p-3 text-xs text-slate-300">
          <div className="mb-2 text-slate-400">Environment</div>
          <div>State space: {env.state_space_size}</div>
          <div>Action space: {env.action_space_size}</div>
          <div>Observation: {env.observation_type}</div>
          <div>Reward range: {env.reward_range[0]} – {env.reward_range[1]}</div>
        </div>
        <div className="rounded border border-slate-700 bg-slate-900/60 p-3 text-xs text-slate-300">
          <div className="mb-2 text-slate-400">Top actions</div>
          {actionStats.map(([name, count]) => (
            <div key={name} className="flex items-center justify-between">
              <span>{name}</span>
              <span>{count}</span>
            </div>
          ))}
          {!actionStats.length && <div>No action stats yet.</div>}
        </div>
      </div>
    </div>
  );
};

export default PerformanceInsightsPage;
