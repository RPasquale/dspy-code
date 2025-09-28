import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Bar, Line } from 'react-chartjs-2';
import Card from '../components/Card';
import { api } from '../api/client';
import { ensureChartsRegistered } from '../lib/registerCharts';
import styles from './ActionsPage.module.css';

const ActionsPage = () => {
  ensureChartsRegistered();
  const [timeframe, setTimeframe] = useState<string>('24h');
  const [limit, setLimit] = useState<number>(600);
  const { data, refetch, isFetching } = useQuery({ queryKey: ['actions-analytics', timeframe, limit], queryFn: () => api.getActionsAnalytics(limit, timeframe), refetchInterval: 20000 });
  const counts = data?.counts_by_type ?? {};
  const typeLabels = Object.keys(counts);
  const typeCounts = typeLabels.map((k) => counts[k] ?? 0);

  const rewardBins = (data?.reward_hist?.bins ?? []) as number[];
  const rewardCounts = (data?.reward_hist?.counts ?? []) as number[];
  const rewardLabels = useMemo(() => rewardBins.slice(0, -1).map((b, i) => `${b.toFixed(1)}–${(rewardBins[i + 1] ?? b).toFixed(1)}`), [rewardBins]);

  const durBins = (data?.duration_hist?.bins ?? []) as number[];
  const durCounts = (data?.duration_hist?.counts ?? []) as number[];
  const durLabels = useMemo(() => durBins.slice(0, -1).map((b, i) => `${b.toFixed(2)}–${(durBins[i + 1] ?? b).toFixed(2)}s`), [durBins]);

  return (
    <div className={styles.wrapper}>
      <h1>Actions Analytics</h1>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <label>Timeframe</label>
        <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
          <option value="1h">1h</option>
          <option value="24h">24h</option>
          <option value="7d">7d</option>
          <option value="30d">30d</option>
        </select>
        <label>Limit</label>
        <input type="number" min={100} step={100} value={limit} onChange={(e) => setLimit(parseInt(e.target.value || '600', 10))} style={{ width: 90 }} />
        <button onClick={() => refetch()} disabled={isFetching}>Refresh</button>
      </div>
      <div className={styles.grid}>
        <Card title="Action Counts by Type">
          <div className={styles.chart}>
            <Bar
              data={{
                labels: typeLabels,
                datasets: [
                  { label: 'Count', data: typeCounts, backgroundColor: '#60a5fa' }
                ]
              }}
              options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { autoSkip: false } } } }}
            />
          </div>
        </Card>

        <Card title="Reward Distribution">
          <div className={styles.chart}>
            <Bar
              data={{
                labels: rewardLabels,
                datasets: [
                  { label: 'Rewards', data: rewardCounts, backgroundColor: '#f59e0b' }
                ]
              }}
              options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false } } }}
            />
          </div>
        </Card>

        <Card title="Execution Time Distribution">
          <div className={styles.chart}>
            <Bar
              data={{
                labels: durLabels,
                datasets: [
                  { label: 'Duration (s)', data: durCounts, backgroundColor: '#34d399' }
                ]
              }}
              options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false } } }}
            />
          </div>
        </Card>

        <Card title="Top Actions">
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Reward</th>
                <th>Conf.</th>
                <th>Dur.</th>
              </tr>
            </thead>
            <tbody>
              {(data?.top_actions ?? []).slice(0, 10).map((a) => (
                <tr key={a.id}>
                  <td>{new Date((a.timestamp || 0) * 1000).toLocaleTimeString()}</td>
                  <td>{a.type}</td>
                  <td>{(a.reward ?? 0).toFixed(2)}</td>
                  <td>{(a.confidence ?? 0).toFixed(2)}</td>
                  <td>{(a.execution_time ?? 0).toFixed(2)}s</td>
                </tr>
              ))}
              {(data?.top_actions ?? []).length === 0 && <tr><td colSpan={5} className={styles.muted}>No actions</td></tr>}
            </tbody>
          </table>
        </Card>

        <Card title="Worst Actions">
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Reward</th>
                <th>Conf.</th>
                <th>Dur.</th>
              </tr>
            </thead>
            <tbody>
              {(data?.worst_actions ?? []).slice(0, 10).map((a) => (
                <tr key={a.id}>
                  <td>{new Date((a.timestamp || 0) * 1000).toLocaleTimeString()}</td>
                  <td>{a.type}</td>
                  <td>{(a.reward ?? 0).toFixed(2)}</td>
                  <td>{(a.confidence ?? 0).toFixed(2)}</td>
                  <td>{(a.execution_time ?? 0).toFixed(2)}s</td>
                </tr>
              ))}
              {(data?.worst_actions ?? []).length === 0 && <tr><td colSpan={5} className={styles.muted}>No actions</td></tr>}
            </tbody>
          </table>
        </Card>
      </div>
    </div>
  );
};

export default ActionsPage;
