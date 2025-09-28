import { useMutation, useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import Card from '../components/Card';
import ParetoChart from '../components/ParetoChart';
import styles from '../styles/Forms.module.css';
import React from 'react';

const SweepsPage: React.FC = () => {
  const [method, setMethod] = React.useState('eprotein');
  const [iterations, setIterations] = React.useState(8);
  const [trainerSteps, setTrainerSteps] = React.useState<number | ''>('');
  const [puffer, setPuffer] = React.useState(false);
  const [workspace, setWorkspace] = React.useState('');

  const { data: stateData, refetch: refetchState } = useQuery({ queryKey: ['rl-sweep-state'], queryFn: api.getRlSweepState, refetchInterval: 10000 });
  const { data: histData, refetch: refetchHist } = useQuery({ queryKey: ['rl-sweep-history'], queryFn: api.getRlSweepHistory, refetchInterval: 15000 });
  const run = useMutation({
    mutationFn: () => api.runRlSweep({ method, iterations, puffer, ...(trainerSteps ? { trainer_steps: Number(trainerSteps) } : {}), ...(workspace ? { workspace } : {}) }),
    onSuccess: () => { setTimeout(() => { refetchState(); refetchHist(); }, 500); }
  });

  const pareto = stateData?.pareto || [];

  return (
    <div style={{ display: 'grid', gap: '16px' }}>
      <Card title="Run RL Sweep" subtitle="Native strategies: eprotein, ecarbs; also random, pareto, protein, carbs">
        <div className={styles.formRow}>
          <label>Method</label>
          <input value={method} onChange={(e) => setMethod(e.target.value)} placeholder="eprotein" />
          <label>Iterations</label>
          <input type="number" min={1} value={iterations} onChange={(e) => setIterations(Number(e.target.value))} />
          <label>Trainer Steps (optional)</label>
          <input value={trainerSteps} onChange={(e) => setTrainerSteps(e.target.value === '' ? '' : Number(e.target.value))} placeholder="e.g., 200" />
          <label>Workspace (optional)</label>
          <input value={workspace} onChange={(e) => setWorkspace(e.target.value)} placeholder="/path/to/ws" />
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <input type="checkbox" checked={puffer} onChange={(e) => setPuffer(e.target.checked)} /> Puffer backend
          </label>
          <button className={styles.button} onClick={() => run.mutate()} disabled={run.isPending}>Run Sweep</button>
        </div>
        {run.isError && <div style={{ color: 'crimson' }}>Failed to start sweep</div>}
        {run.isSuccess && <div style={{ color: 'green' }}>Sweep started</div>}
      </Card>

      <Card title="Current Sweep State" subtitle={stateData?.exists ? 'Active' : 'No state found'}>
        {!stateData?.exists && <div>No sweep has been persisted yet.</div>}
        {stateData?.exists && (
          <pre style={{ maxHeight: 320, overflow: 'auto' }}>{JSON.stringify(stateData?.state, null, 2)}</pre>
        )}
      </Card>

      <Card title="Pareto (score vs cost)" subtitle="Derived from state observations">
        {pareto.length === 0 ? (
          <div>No Pareto points yet.</div>
        ) : (
          <>
            <ParetoChart points={pareto as any} />
            <table className={styles.table} style={{ marginTop: 8 }}>
              <thead><tr><th>Score</th><th>Cost (s)</th></tr></thead>
              <tbody>
                {pareto.map((p: any, i: number) => (
                  <tr key={i}><td>{Number(p.output).toFixed(3)}</td><td>{Number(p.cost).toFixed(2)}</td></tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </Card>

      <Card title="History" subtitle="Recent sweep experiments">
        <pre style={{ maxHeight: 240, overflow: 'auto' }}>{JSON.stringify(histData || {}, null, 2)}</pre>
      </Card>
    </div>
  );
};

export default SweepsPage;
