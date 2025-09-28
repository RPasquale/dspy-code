import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import Card from '../components/Card';
import { api } from '../api/client';
import styles from './TrainingPage.module.css';
import GrpoControls from '@/components/GrpoControls';

const TrainingPage = () => {
  const [workspace, setWorkspace] = useState('/app/test_project');
  const [steps, setSteps] = useState(200);
  const [nEnv, setNEnv] = useState(2);
  const [policy, setPolicy] = useState<'epsilon-greedy' | 'ucb1' | 'thompson'>('epsilon-greedy');
  const [epsilon, setEpsilon] = useState(0.1);
  const [ucbC, setUcbC] = useState(2.0);
  const [trainer, setTrainer] = useState<'bandit' | 'neural' | 'ppo'>('bandit');
  const [output, setOutput] = useState('');

  const sys = useQuery({ queryKey: ['system-resources'], queryFn: api.getSystemResources, refetchInterval: 5000 });

  const run = useMutation({
    mutationFn: async () => {
      let cmd = '';
      if (trainer === 'bandit') {
        cmd = `rl train --steps ${steps} --n-envs ${nEnv} --policy ${policy} --epsilon ${epsilon} --ucb-c ${ucbC}`;
      } else if (trainer === 'neural') {
        cmd = `rl neural --steps ${steps} --n-envs ${nEnv}`;
      } else {
        cmd = `rl ppo --n-envs ${nEnv} --total-steps ${Math.max(steps, 10000)}`;
      }
      const res = await api.sendCommand(cmd, { workspace });
      return res;
    },
    onSuccess: (res) => setOutput((res.output || res.error || '').trim()),
    onError: (e: any) => setOutput(String(e?.message || e))
  });

  return (
    <div className={styles.wrapper}>
      <h1>Training Setup</h1>
      {(sys.data && sys.data.host && sys.data.host.ok === false) && (
        <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fecaca', padding: 8, borderRadius: 6, marginBottom: 12 }}>
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
            <label className={styles.label}>Trainer</label>
            <select className={styles.select} value={trainer} onChange={(e) => setTrainer(e.target.value as any)}>
              <option value="bandit">Bandit</option>
              <option value="neural">Neural</option>
              <option value="ppo">PPO</option>
            </select>
          </div>
          <div className={styles.row}>
            <label className={styles.label}>Steps</label>
            <input className={styles.input} type="number" min={100} step={100} value={steps} onChange={(e) => setSteps(parseInt(e.target.value || '200', 10))} />
          </div>
          <div className={styles.row}>
            <label className={styles.label}>n_envs</label>
            <input className={styles.input} type="number" min={1} step={1} value={nEnv} onChange={(e) => setNEnv(parseInt(e.target.value || '2', 10))} />
          </div>
          {trainer === 'bandit' && (
            <>
              <div className={styles.row}>
                <label className={styles.label}>Policy</label>
                <select className={styles.select} value={policy} onChange={(e) => setPolicy(e.target.value as any)}>
                  <option value="epsilon-greedy">epsilon-greedy</option>
                  <option value="ucb1">ucb1</option>
                  <option value="thompson">thompson</option>
                </select>
              </div>
              <div className={styles.row}>
                <label className={styles.label}>Epsilon</label>
                <input className={styles.input} type="number" step={0.01} value={epsilon} onChange={(e) => setEpsilon(parseFloat(e.target.value || '0.1'))} />
              </div>
              <div className={styles.row}>
                <label className={styles.label}>UCB c</label>
                <input className={styles.input} type="number" step={0.1} value={ucbC} onChange={(e) => setUcbC(parseFloat(e.target.value || '2.0'))} />
              </div>
            </>
          )}
        </div>
        <div className={styles.actions}>
          <button className={styles.button} onClick={() => run.mutate()} disabled={run.isPending || (sys.data && sys.data.host && sys.data.host.ok === false)}>
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
