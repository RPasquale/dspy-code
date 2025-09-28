import { useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Line } from 'react-chartjs-2';
import Card from '../components/Card';
import { api } from '../api/client';
import type { RewardsConfigResponse } from '../api/types';
import { ensureChartsRegistered } from '../lib/registerCharts';
import styles from './RewardsPage.module.css';

const RewardsPage = () => {
  ensureChartsRegistered();
  const [workspace, setWorkspace] = useState<string>('');
  const { data: cfg, isFetching, refetch } = useQuery({
    queryKey: ['rewards-config', workspace],
    queryFn: () => api.getRewardsConfig(workspace || undefined),
    refetchInterval: 30000
  });
  const { data: rl } = useQuery({ queryKey: ['rl-metrics'], queryFn: api.getRlMetrics, refetchInterval: 30000 });
  const [weightsDraft, setWeightsDraft] = useState<Record<string, string>>({});
  const [actionsDraft, setActionsDraft] = useState<string | null>(null);
  const save = useMutation({ mutationFn: (payload: any) => api.updateRewardsConfig(payload, workspace || undefined), onSuccess: () => refetch() });

  const weights = useMemo(() => cfg?.weights ?? {}, [cfg]);
  const weightKeys = Object.keys(weights);

  const handleSave = () => {
    const nextWeights: Record<string, number> = {};
    for (const [k, v] of Object.entries(weightsDraft)) {
      const f = parseFloat(v);
      if (!Number.isNaN(f)) nextWeights[k] = f;
    }
    const payload: Partial<RewardsConfigResponse> = {};
    if (Object.keys(nextWeights).length > 0) (payload as any).weights = nextWeights;
    if (actionsDraft !== null) (payload as any).actions = actionsDraft.split(',').map((s) => s.trim()).filter(Boolean);
    if (Object.keys(payload).length > 0) save.mutate(payload);
  };

  return (
    <div className={styles.wrapper}>
      <h1>Rewards & RL Config</h1>
      <div className={styles.grid}>
        <Card title="Reward Weights" subtitle={isFetching ? 'Refreshing…' : (cfg as any)?.path ? `Config: ${(cfg as any).path}` : undefined}
          actions={<button className={styles.button} onClick={handleSave} disabled={save.isPending}>{save.isPending ? 'Saving…' : 'Save Changes'}</button>}
        >
          <div className={styles.row}>
            <label className={styles.muted} style={{ minWidth: 90 }}>Workspace</label>
            <input className={styles.input} placeholder="optional: /abs/path/to/workspace" value={workspace} onChange={(e) => setWorkspace(e.target.value)} />
            <button className={styles.button + ' ' + 'secondary'} onClick={() => refetch()}>Load</button>
          </div>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Key</th>
                <th>Current</th>
                <th>New</th>
              </tr>
            </thead>
            <tbody>
              {weightKeys.map((k) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{weights[k]}</td>
                  <td><input className={styles.input} placeholder={`${weights[k]}`} value={weightsDraft[k] ?? ''} onChange={(e) => setWeightsDraft((w) => ({ ...w, [k]: e.target.value }))} /></td>
                </tr>
              ))}
              {weightKeys.length === 0 && (
                <tr><td colSpan={3} className={styles.muted}>No weights defined</td></tr>
              )}
            </tbody>
          </table>
          <div style={{ marginTop: 12 }}>
            <div className={styles.row}>
              <input className={styles.input} placeholder="new_key" onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  const key = (e.target as HTMLInputElement).value.trim();
                  if (key) setWeightsDraft((w) => ({ ...w, [key]: '0.0' }));
                  (e.target as HTMLInputElement).value = '';
                }
              }} />
              <span className={styles.muted}>Press Enter to add new weight key with 0.0</span>
            </div>
          </div>
        </Card>

        <Card title="Policy & Actions">
          <div className={styles.row}><span className={styles.muted}>Policy</span><strong>{cfg?.policy ?? 'epsilon-greedy'}</strong></div>
          <div className={styles.row}><span className={styles.muted}>Epsilon</span><strong>{cfg?.epsilon ?? 0.1}</strong></div>
          <div className={styles.row}><span className={styles.muted}>UCB c</span><strong>{cfg?.ucb_c ?? 2.0}</strong></div>
          <div className={styles.row}><span className={styles.muted}>Timeout</span><strong>{cfg?.timeout_sec ?? 180}s</strong></div>
          <div style={{ marginTop: 8 }}>
            <label className={styles.muted}>Actions (comma-separated)</label>
            <input
              className={styles.input}
              placeholder={(cfg?.actions ?? []).join(', ')}
              value={actionsDraft ?? ''}
              onChange={(e) => setActionsDraft(e.target.value)}
            />
          </div>
        </Card>

        <Card title="Reward History" subtitle="From RL metrics">
          <div className={styles.chart}>
            <Line
              data={{
                labels: (rl?.reward_history ?? []).map((r) => new Date(r.timestamp).toLocaleTimeString()),
                datasets: [
                  {
                    label: 'Reward',
                    data: (rl?.reward_history ?? []).map((r) => r.reward),
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245,158,11,0.15)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                  }
                ]
              }}
              options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false } } }}
            />
          </div>
        </Card>
      </div>
    </div>
  );
};

export default RewardsPage;
