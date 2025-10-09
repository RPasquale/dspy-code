import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import Card from '../components/Card';
import { api } from '../api/client';
import styles from './TrainingPage.module.css';
import GrpoControls from '@/components/GrpoControls';

const TrainingPage = () => {
  const [workspace, setWorkspace] = useState('/app/test_project');
  const [steps, setSteps] = useState(500);
  const [nEnv, setNEnv] = useState(2);
  const [lr, setLr] = useState(1e-3);
  const [entropy, setEntropy] = useState(0.01);
  const [skipGepa, setSkipGepa] = useState(false);
  const [logJsonl, setLogJsonl] = useState('');
  const [output, setOutput] = useState('');

  const sys = useQuery({
    queryKey: ['system-resources'],
    queryFn: api.getSystemResources,
    refetchInterval: 5000,
  });

  const run = useMutation({
    mutationFn: async () => {
      const skipFlag = skipGepa ? ' --skip-gepa' : '';
      const logFlag = logJsonl.trim() ? ` --log-jsonl ${logJsonl.trim()}` : '';
      const cmd = `rl train --steps ${steps} --n-envs ${nEnv} --lr ${lr} --entropy ${entropy}${skipFlag}${logFlag}`;
      const res = await api.sendCommand(cmd, { workspace });
      return res;
    },
    onSuccess: (res) => setOutput((res.output || res.error || '').trim()),
    onError: (e: any) => setOutput(String(e?.message || e)),
  });

  const storageBlocked = Boolean(sys.data && sys.data.host && sys.data.host.ok === false);

  return (
    <div className={styles.wrapper}>
      <h1>Training Setup</h1>
      {storageBlocked && (
        <div
          style={{
            background: 'rgba(239,68,68,0.1)',
            border: '1px solid rgba(239,68,68,0.3)',
            color: '#fecaca',
            padding: 8,
            borderRadius: 6,
            marginBottom: 12,
          }}
        >
          Insufficient storage. Free {sys.data?.host?.disk?.free_gb?.toFixed?.(2)} GB &lt; threshold {sys.data?.host?.threshold_free_gb}. Training actions are disabled until space is freed.
        </div>
      )}
      <GrpoControls />
      <Card title="Configure RL Training">
        <div className={styles.form}>
          <div className={styles.row}>
            <label className={styles.label}>Workspace (container path)</label>
            <input className={styles.input} value={workspace} onChange={(e) => setWorkspace(e.target.value)} />
          </div>
          <div className={styles.row}>
            <label className={styles.label}>Steps</label>
            <input
              className={styles.input}
              type="number"
              min={100}
              step={100}
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value || '500', 10))}
            />
          </div>
          <div className={styles.row}>
            <label className={styles.label}>n_envs</label>
            <input
              className={styles.input}
              type="number"
              min={1}
              step={1}
              value={nEnv}
              onChange={(e) => setNEnv(parseInt(e.target.value || '2', 10))}
            />
          </div>
          <div className={styles.row}>
            <label className={styles.label}>Learning Rate</label>
            <input
              className={styles.input}
              type="number"
              step={0.0001}
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value || '0.001'))}
            />
          </div>
          <div className={styles.row}>
            <label className={styles.label}>Entropy Coef</label>
            <input
              className={styles.input}
              type="number"
              step={0.001}
              value={entropy}
              onChange={(e) => setEntropy(parseFloat(e.target.value || '0.01'))}
            />
          </div>
          <div className={styles.row}>
            <label className={styles.label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="checkbox" checked={skipGepa} onChange={(e) => setSkipGepa(e.target.checked)} />
              Skip GEPA warmup (useful for quick smoke tests)
            </label>
          </div>
          <div className={styles.row}>
            <label className={styles.label}>Log JSONL (optional)</label>
            <input
              className={styles.input}
              placeholder="logs/rl_trace.jsonl"
              value={logJsonl}
              onChange={(e) => setLogJsonl(e.target.value)}
            />
          </div>
        </div>
        <div className={styles.actions}>
          <button
            className={styles.button}
            onClick={() => run.mutate()}
            disabled={run.isPending || storageBlocked}
          >
            {run.isPending ? 'Starting…' : 'Start Training'}
          </button>
        </div>
      </Card>
      <Card title="Output">
        <pre className={styles.log}>{output || 'No output'}</pre>
      </Card>
    </div>
  );
};

export default TrainingPage;
