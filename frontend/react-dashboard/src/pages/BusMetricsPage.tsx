import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Line, Bar } from 'react-chartjs-2';
import { ensureChartsRegistered } from '../lib/registerCharts';
import { api } from '../api/client';
import type { BusMetricsResponse } from '../api/types';
import styles from './BusMetricsPage.module.css';

ensureChartsRegistered();

const Card: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className={styles.card}>
    <div className={styles.title}>{title}</div>
    {children}
  </div>
);

const BusMetricsPage = () => {
  const { data, isLoading, error } = useQuery<BusMetricsResponse>({
    queryKey: ['bus-metrics'],
    queryFn: api.getBusMetrics,
    refetchInterval: 10000
  });

  const dlqItems = useMemo(() => {
    const byTopic = data?.dlq.by_topic ?? {};
    return Object.entries(byTopic)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
  }, [data]);

  const queueDepthByTopic = useMemo(() => {
    const topics = data?.bus.topics ?? {};
    const labels: string[] = [];
    const depths: number[] = [];
    Object.entries(topics).forEach(([topic, sizes]) => {
      labels.push(topic);
      const maxDepth = (sizes ?? []).reduce((m, v) => (typeof v === 'number' ? Math.max(m, v) : m), 0);
      depths.push(maxDepth);
    });
    return { labels, depths };
  }, [data]);

  const queueBarData = useMemo(() => ({
    labels: queueDepthByTopic.labels,
    datasets: [
      {
        label: 'Queue Max Depth',
        data: queueDepthByTopic.depths,
        backgroundColor: 'rgba(96, 165, 250, 0.35)'
      }
    ]
  }), [queueDepthByTopic]);

  const dlqTrendData = useMemo(() => {
    const ts = data?.history?.timestamps ?? [];
    const dlq = data?.history?.dlq_total ?? [];
    return {
      labels: ts.map((t) => new Date(t * 1000).toLocaleTimeString()),
      datasets: [
        {
          label: 'DLQ Total',
          data: dlq,
          borderColor: '#f87171',
          backgroundColor: 'rgba(248, 113, 113, 0.15)',
          fill: true,
          tension: 0.3
        }
      ]
    };
  }, [data]);

  const depthTrendData = useMemo(() => {
    const ts = data?.history?.timestamps ?? [];
    const d = data?.history?.queue_max_depth ?? [];
    return {
      labels: ts.map((t) => new Date(t * 1000).toLocaleTimeString()),
      datasets: [
        {
          label: 'Max Queue Depth',
          data: d,
          borderColor: '#34d399',
          backgroundColor: 'rgba(52, 211, 153, 0.15)',
          fill: true,
          tension: 0.3
        }
      ]
    };
  }, [data]);

  return (
    <div className={styles.wrapper}>
      {isLoading && <div>Loading bus metrics…</div>}
      {error && <div>Error loading bus metrics</div>}
      {data && (
        <>
          <div className={styles.grid}>
            <Card title={`Dead Letter Queue (total: ${data.dlq.total})`}>
              <div className={styles.list}>
                {dlqItems.length === 0 && <div>No DLQ entries recorded.</div>}
                {dlqItems.map(([topic, count]) => (
                  <div key={topic}>
                    <span className={styles.badge}>{count}</span> <span>{topic}</span>
                  </div>
                ))}
              </div>
            </Card>
            <Card title="Current Queue Depth by Topic">
              <div className={styles.chartBox}>
                <Bar data={queueBarData} options={{ maintainAspectRatio: false }} />
              </div>
            </Card>
          </div>
          <div className={styles.grid}>
            <Card title="DLQ Trend (approx)">
              <div className={styles.chartBox}>
                <Line data={dlqTrendData} options={{ maintainAspectRatio: false }} />
              </div>
            </Card>
            <Card title="Max Queue Depth Trend">
              <div className={styles.chartBox}>
                <Line data={depthTrendData} options={{ maintainAspectRatio: false }} />
              </div>
            </Card>
          </div>
          {data.alerts?.length > 0 && (
            <Card title="Alerts">
              <div className={styles.list}>
                {data.alerts.map((a, i) => (
                  <div key={i}>
                    <span className={styles.badge}>{a.level.toUpperCase()}</span> {a.message}
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
};

export default BusMetricsPage;

