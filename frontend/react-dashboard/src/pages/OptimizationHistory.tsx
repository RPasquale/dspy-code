import { useEffect, useState } from 'react';
import Card from '../components/Card';
import { api } from '../api/client';
import styles from './SignaturesPage.module.css';

const OptimizationHistory = ({ name }: { name: string }) => {
  const [data, setData] = useState<{ history: any[]; metrics: any } | null>(null);
  useEffect(() => { (async () => { try { setData(await api.getSignatureOptHistory(name)); } catch {} })(); }, [name]);
  const hist = data?.history || [];
  return (
    <Card title="Optimization History" subtitle={hist.length ? `${hist.length} entries` : 'No entries'}>
      <table className={styles.table}>
        <thead><tr><th>When</th><th>Type</th><th>ΔPerf</th><th>ΔSuccess</th><th>ΔRT</th></tr></thead>
        <tbody>
          {hist.map((h: any, i: number) => (
            <tr key={i}>
              <td>{h.timestamp ? new Date(h.timestamp * 1000).toLocaleString() : '-'}</td>
              <td>{h.type || '-'}</td>
              <td>{typeof h.performance_gain === 'number' ? h.performance_gain.toFixed(2) : '-'}</td>
              <td>{typeof h.accuracy_improvement === 'number' ? h.accuracy_improvement.toFixed(2) : '-'}</td>
              <td>{typeof h.response_time_reduction === 'number' ? `-${h.response_time_reduction.toFixed(2)}s` : '-'}</td>
            </tr>
          ))}
          {hist.length === 0 && (<tr><td colSpan={5} className={styles.muted}>No optimization history</td></tr>)}
        </tbody>
      </table>
    </Card>
  );
};

export default OptimizationHistory;

