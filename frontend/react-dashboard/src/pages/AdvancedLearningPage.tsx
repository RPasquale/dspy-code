import { FormEvent, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Line, Bar } from 'react-chartjs-2';
import Card from '../components/Card';
import StatusPill from '../components/StatusPill';
import { api } from '../api/client';
import { ensureChartsRegistered } from '../lib/registerCharts';
import type { StreamMetricsResponse, OptimizeSignatureResponse } from '../api/types';
import styles from './AdvancedLearningPage.module.css';

ensureChartsRegistered();

const timeframes = [
  { label: 'Last Hour', value: '1h' },
  { label: '24 Hours', value: '24h' },
  { label: '7 Days', value: '7d' }
];

const configOptions = [
  { label: 'Auto Training', type: 'auto_training', defaultValue: true },
  { label: 'Stream Metrics', type: 'stream_metrics', defaultValue: true },
  { label: 'Safety Sandbox', type: 'sandbox_mode', defaultValue: false }
];

type ChatMessage = {
  role: 'user' | 'agent' | 'system';
  text: string;
  timestamp: string;
};

const AdvancedLearningPage = () => {
  const [timeframe, setTimeframe] = useState('24h');
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: 'system', text: 'Connected to agent interface. Ask about performance, deployment, or optimization.', timestamp: new Date().toISOString() }
  ]);
  const [configState, setConfigState] = useState(() =>
    Object.fromEntries(configOptions.map((option) => [option.type, option.defaultValue]))
  );
  const [timeoutSeconds, setTimeoutSeconds] = useState(120);
  const [memoryLimit, setMemoryLimit] = useState(4);
  const [optimizationResult, setOptimizationResult] = useState<OptimizeSignatureResponse | null>(null);

  const queryClient = useQueryClient();

  const signaturesQuery = useQuery({
    queryKey: ['signatures'],
    queryFn: api.getSignatures,
    refetchInterval: 20000
  });

  const verifiersQuery = useQuery({
    queryKey: ['verifiers'],
    queryFn: api.getVerifiers,
    refetchInterval: 25000
  });

  const learningMetricsQuery = useQuery({
    queryKey: ['learning-metrics'],
    queryFn: api.getLearningMetrics,
    refetchInterval: 30000
  });

  const performanceHistoryQuery = useQuery({
    queryKey: ['performance-history', timeframe],
    queryFn: () => api.getPerformanceHistory(timeframe),
    refetchInterval: 35000
  });

  const kafkaQuery = useQuery({
    queryKey: ['kafka-topics'],
    queryFn: api.getKafkaTopics,
    refetchInterval: 40000
  });

  const sparkQuery = useQuery({
    queryKey: ['spark-workers'],
    queryFn: api.getSparkWorkers,
    refetchInterval: 45000
  });

  const streamMetricsQuery = useQuery({
    queryKey: ['stream-metrics'],
    queryFn: api.getStreamMetrics,
    refetchInterval: 20000
  });

  const rlMetricsQuery = useQuery({
    queryKey: ['rl-metrics'],
    queryFn: api.getRlMetrics,
    refetchInterval: 25000
  });

  const optimizeMutation = useMutation({
    mutationFn: (signatureName: string) => api.optimizeSignature({ signature_name: signatureName, type: 'performance' }),
    onSuccess: (result) => {
      setOptimizationResult(result);
      queryClient.invalidateQueries({ queryKey: ['signatures'] });
    }
  });

  const chatMutation = useMutation({
    mutationFn: (message: string) => api.sendChat({ message }),
    onSuccess: (response, variables) => {
      const timestamp = new Date().toISOString();
      setChatMessages((prev) => [
        ...prev,
        { role: 'user', text: variables, timestamp },
        { role: 'agent', text: response.response, timestamp: new Date(response.timestamp * 1000).toISOString() }
      ]);
      setChatInput('');
    }
  });

  const configMutation = useMutation({
    mutationFn: api.updateConfig
  });

  const signatureChartData = useMemo(() => {
    const signatures = signaturesQuery.data?.signatures ?? [];
    return {
      labels: signatures.map((sig) => sig.name),
      datasets: [
        {
          label: 'Performance Score',
          data: signatures.map((sig) => sig.performance),
          backgroundColor: 'rgba(99, 102, 241, 0.6)',
          borderRadius: 8
        }
      ]
    };
  }, [signaturesQuery.data]);

  const learningChartData = useMemo(() => {
    const metrics = learningMetricsQuery.data?.performance_over_time;
    return {
      labels: metrics?.timestamps.map((timestamp) => new Date(timestamp).toLocaleTimeString()) ?? [],
      datasets: [
        {
          label: 'Training Accuracy',
          data: metrics?.training_accuracy ?? [],
          borderColor: '#34d399',
          backgroundColor: 'rgba(52, 211, 153, 0.1)',
          tension: 0.35,
          fill: true
        },
        {
          label: 'Validation Accuracy',
          data: metrics?.validation_accuracy ?? [],
          borderColor: '#60a5fa',
          backgroundColor: 'rgba(96, 165, 250, 0.1)',
          tension: 0.35,
          fill: true
        }
      ]
    };
  }, [learningMetricsQuery.data]);

  const performanceChartData = useMemo(() => {
    const history = performanceHistoryQuery.data;
    return {
      labels: history?.timestamps.map((timestamp) => new Date(timestamp).toLocaleString()) ?? [],
      datasets: [
        {
          label: 'Response Time (s)',
          data: history?.metrics.response_times ?? [],
          borderColor: '#fbbf24',
          backgroundColor: 'rgba(251, 191, 36, 0.1)',
          tension: 0.3,
          yAxisID: 'y'
        },
        {
          label: 'Success Rate (%)',
          data: history?.metrics.success_rates ?? [],
          borderColor: '#34d399',
          backgroundColor: 'rgba(52, 211, 153, 0.1)',
          tension: 0.3,
          yAxisID: 'y1'
        }
      ]
    };
  }, [performanceHistoryQuery.data]);

  const kafkaTopics = kafkaQuery.data?.topics ?? [];
  const sparkWorkers = sparkQuery.data?.workers ?? [];
  const rlMetrics = rlMetricsQuery.data?.metrics;

  const handleToggle = (type: string) => {
    setConfigState((prev) => {
      const next = !prev[type];
      configMutation.mutate({ type, value: next });
      return { ...prev, [type]: next };
    });
  };

  const handleSendChat = (event: FormEvent) => {
    event.preventDefault();
    const message = chatInput.trim();
    if (!message) return;
    chatMutation.mutate(message);
  };

  const handleTimeoutCommit = (value: number) => {
    configMutation.mutate({ type: 'timeout', value });
  };

  const handleMemoryCommit = (value: number) => {
    configMutation.mutate({ type: 'memory_limit', value });
  };

  return (
    <div className={styles.wrapper}>
      <div className={styles.pageHeading}>
        <div>
          <h1>Advanced Learning & Operations</h1>
          <p>Deep insights across signatures, verifiers, reinforcement learning, and streaming infrastructure.</p>
        </div>
        {rlMetrics && <StatusPill status={rlMetrics.training_status} text={`Episode ${rlMetrics.current_episode}`} />}
      </div>

      <div className={styles.grid}>
        <Card title="Signature Performance" subtitle="Top signatures ordered by current score">
          <div className={styles.signatureSection}>
            <div className={styles.signatureList}>
              {(signaturesQuery.data?.signatures ?? []).map((sig) => (
                <article key={sig.name} className={styles.signatureItem}>
                  <header>
                    <h3>{sig.name}</h3>
                    <StatusPill status={sig.active ? 'active' : 'paused'} text={sig.type} />
                  </header>
                  <dl>
                    <div>
                      <dt>Performance</dt>
                      <dd>{sig.performance.toFixed(1)}</dd>
                    </div>
                    <div>
                      <dt>Success Rate</dt>
                      <dd>{sig.success_rate ? `${sig.success_rate.toFixed(1)}%` : '—'}</dd>
                    </div>
                    <div>
                      <dt>Iterations</dt>
                      <dd>{sig.iterations}</dd>
                    </div>
                    <div>
                      <dt>Avg Response</dt>
                      <dd>{sig.avg_response_time ? `${sig.avg_response_time.toFixed(2)}s` : '—'}</dd>
                    </div>
                  </dl>
                  <footer>
                    <span>Updated {new Date(sig.last_updated).toLocaleString()}</span>
                    <button
                      className={styles.optimizeButton}
                      type="button"
                      onClick={() => optimizeMutation.mutate(sig.name)}
                      disabled={optimizeMutation.isPending}
                    >
                      {optimizeMutation.isPending ? 'Optimizing…' : 'Optimize'}
                    </button>
                  </footer>
                </article>
              ))}
            </div>
            <div className={styles.signatureChart}>
              <Bar
                data={signatureChartData}
                options={{
                  responsive: true,
                  plugins: {
                    legend: { display: false }
                  },
                  scales: {
                    x: { ticks: { color: 'rgba(148,163,184,0.7)' } },
                    y: { beginAtZero: true, ticks: { color: 'rgba(148,163,184,0.7)' } }
                  }
                }}
              />
            </div>
          </div>
          {optimizationResult && (
            <div className={styles.optimizationSummary}>
              <strong>{optimizationResult.signature_name}</strong>
              <span>Δ score +{optimizationResult.improvements.performance_gain.toFixed(2)}</span>
              <span>Accuracy +{optimizationResult.improvements.accuracy_improvement.toFixed(2)}%</span>
              <span>Avg response −{optimizationResult.improvements.response_time_reduction.toFixed(2)}s</span>
            </div>
          )}
        </Card>

        <Card title="Learning Trajectory" subtitle="Accuracy progression over time">
          <div className={styles.learningChart}>
            <Line
              data={learningChartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: {
                  x: { ticks: { color: 'rgba(148,163,184,0.7)' } },
                  y: {
                    ticks: { color: 'rgba(148,163,184,0.7)' },
                    suggestedMin: 60,
                    suggestedMax: 100
                  }
                }
              }}
            />
          </div>
          <div className={styles.learningStats}>
            <Stat label="Training Examples" value={learningMetricsQuery.data?.learning_stats.total_training_examples ?? '--'} />
            <Stat label="Successful Optimizations" value={learningMetricsQuery.data?.learning_stats.successful_optimizations ?? '--'} />
            <Stat label="Avg Improvement" value={learningMetricsQuery.data ? `${(learningMetricsQuery.data.learning_stats.avg_improvement_per_iteration * 100).toFixed(2)}%` : '--'} />
            <Stat label="Learning Rate" value={learningMetricsQuery.data ? learningMetricsQuery.data.learning_stats.current_learning_rate.toFixed(4) : '--'} />
          </div>
        </Card>

        <Card
          title="Performance History"
          subtitle="Hybrid view combining latency and success"
          actions={
            <div className={styles.timeframeToggle}>
              {timeframes.map((option) => (
                <button
                  key={option.value}
                  className={option.value === timeframe ? styles.timeframeActive : styles.timeframeButton}
                  onClick={() => setTimeframe(option.value)}
                  type="button"
                >
                  {option.label}
                </button>
              ))}
            </div>
          }
        >
          <div className={styles.performanceChart}>
            <Line
              data={performanceChartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                scales: {
                  y: {
                    type: 'linear',
                    position: 'left',
                    ticks: { color: '#fbbf24' },
                    title: { display: true, text: 'Seconds', color: '#fbbf24' }
                  },
                  y1: {
                    type: 'linear',
                    position: 'right',
                    ticks: { color: '#34d399' },
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Success %', color: '#34d399' }
                  }
                }
              }}
            />
          </div>
        </Card>

        <Card title="Verifier Fleet" subtitle="Quality gates across the agent pipeline">
          <div className={styles.verifierGrid}>
            {(verifiersQuery.data?.verifiers ?? []).map((verifier) => (
              <div key={verifier.name} className={styles.verifierItem}>
                <header>
                  <h3>{verifier.name}</h3>
                  <StatusPill status={verifier.status} text={`${verifier.accuracy.toFixed(1)}%`} />
                </header>
                <dl>
                  <div>
                    <dt>Checks</dt>
                    <dd>{verifier.checks_performed}</dd>
                  </div>
                  <div>
                    <dt>Issues</dt>
                    <dd>{verifier.issues_found}</dd>
                  </div>
                  <div>
                    <dt>Avg Runtime</dt>
                    <dd>{verifier.avg_execution_time.toFixed(2)}s</dd>
                  </div>
                  <div>
                    <dt>Last Run</dt>
                    <dd>{new Date(verifier.last_run).toLocaleString()}</dd>
                  </div>
                </dl>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Streaming & Kafka" subtitle="Key brokers and topic metrics">
          <div className={styles.kafkaSection}>
            <div className={styles.kafkaList}>
              {kafkaTopics.map((topic) => (
                <article key={topic.name} className={styles.kafkaTopic}>
                  <header>
                    <h3>{topic.name}</h3>
                    <span>{topic.partitions} partitions</span>
                  </header>
                  <dl>
                    <div>
                      <dt>Msgs / min</dt>
                      <dd>{topic.messages_per_minute}</dd>
                    </div>
                    <div>
                      <dt>Total</dt>
                      <dd>{topic.total_messages}</dd>
                    </div>
                    <div>
                      <dt>Lag</dt>
                      <dd>{topic.consumer_lag}</dd>
                    </div>
                    <div>
                      <dt>Producers</dt>
                      <dd>{topic.producers.join(', ')}</dd>
                    </div>
                  </dl>
                </article>
              ))}
            </div>
            <div className={styles.streamStats}>
              <StreamMetricsPanel data={streamMetricsQuery.data} />
            </div>
          </div>
        </Card>

        <Card title="Spark Cluster" subtitle="Worker utilization and running jobs">
          <div className={styles.sparkSection}>
            <div className={styles.sparkWorkers}>
              {sparkWorkers.map((worker) => (
                <div key={worker.id} className={styles.sparkWorker}>
                  <header>
                    <h3>{worker.host}</h3>
                    <StatusPill status={worker.status} text={`${worker.cores_used}/${worker.cores} cores`} />
                  </header>
                  <dl>
                    <div>
                      <dt>Memory</dt>
                      <dd>
                        {worker.memory_used} / {worker.memory}
                      </dd>
                    </div>
                    <div>
                      <dt>Executors</dt>
                      <dd>{worker.executors}</dd>
                    </div>
                    <div>
                      <dt>Heartbeat</dt>
                      <dd>{new Date(worker.last_heartbeat).toLocaleTimeString()}</dd>
                    </div>
                  </dl>
                </div>
              ))}
            </div>
            <div className={styles.sparkSummary}>
              {sparkQuery.data && (
                <ul>
                  <li>Applications Running: {sparkQuery.data.master.applications_running}</li>
                  <li>CPU Utilization: {sparkQuery.data.cluster_metrics.cpu_utilization.toFixed(1)}%</li>
                  <li>Memory Used: {sparkQuery.data.cluster_metrics.used_memory}</li>
                  <li>Completed Apps: {sparkQuery.data.master.applications_completed}</li>
                </ul>
              )}
            </div>
          </div>
        </Card>

        <Card title="Reinforcement Learning" subtitle="Latest policy stats and actions">
          {rlMetrics && (
            <div className={styles.rlGrid}>
              <Stat label="Average Reward" value={rlMetrics.avg_reward.toFixed(1)} />
              <Stat label="Best Reward" value={rlMetrics.best_reward.toFixed(1)} />
              <Stat label="Loss" value={rlMetrics.loss.toFixed(3)} />
              <Stat label="Replay Buffer" value={`${rlMetrics.replay_buffer_used} / ${rlMetrics.replay_buffer_size}`} />
            </div>
          )}
          <div className={styles.actionDistribution}>
            {rlMetricsQuery.data &&
              Object.entries(rlMetricsQuery.data.action_stats).map(([action, count]) => (
                <div key={action}>
                  <span>{action}</span>
                  <strong>{count}</strong>
                </div>
              ))}
          </div>
        </Card>

        <Card title="Configuration" subtitle="Toggle agent features and tune runtime limits" dense>
          <div className={styles.configGrid}>
            {configOptions.map((option) => (
              <button
                key={option.type}
                type="button"
                className={`${styles.configToggle} ${configState[option.type] ? styles.configToggleActive : ''}`}
                onClick={() => handleToggle(option.type)}
              >
                <span>{option.label}</span>
                <small>{configState[option.type] ? 'Enabled' : 'Disabled'}</small>
              </button>
            ))}
          </div>
          <div className={styles.sliderRow}>
            <label>
              Timeout — {timeoutSeconds}s
              <input
                type="range"
                min={30}
                max={300}
                value={timeoutSeconds}
                onChange={(event) => setTimeoutSeconds(Number(event.target.value))}
                onMouseUp={() => handleTimeoutCommit(timeoutSeconds)}
                onTouchEnd={() => handleTimeoutCommit(timeoutSeconds)}
              />
            </label>
            <label>
              Memory — {memoryLimit}GB
              <input
                type="range"
                min={1}
                max={16}
                value={memoryLimit}
                onChange={(event) => setMemoryLimit(Number(event.target.value))}
                onMouseUp={() => handleMemoryCommit(memoryLimit)}
                onTouchEnd={() => handleMemoryCommit(memoryLimit)}
              />
            </label>
          </div>
        </Card>

        <Card title="Agent Chat" subtitle="Ask the agent about current state or next actions">
          <div className={styles.chatPanel}>
            <div className={`${styles.chatHistory} scrollbar`}>
              {chatMessages.map((message, index) => {
                const roleClass = message.role === 'user' ? styles.user : message.role === 'agent' ? styles.agent : styles.system;
                return (
                  <div key={`${message.timestamp}-${index}`} className={`${styles.chatBubble} ${roleClass}`}>
                    <span>{message.text}</span>
                    <small>{new Date(message.timestamp).toLocaleTimeString()}</small>
                  </div>
                );
              })}
              {chatMutation.isPending && <div className={styles.chatBubble}>Agent is composing…</div>}
            </div>
            <form className={styles.chatInputRow} onSubmit={handleSendChat}>
              <input
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                placeholder="How is the training loop performing right now?"
              />
              <button type="submit" disabled={chatMutation.isPending}>
                Send
              </button>
            </form>
          </div>
        </Card>
      </div>
    </div>
  );
};

const Stat = ({ label, value }: { label: string; value: number | string }) => (
  <div className={styles.stat}>
    <span>{label}</span>
    <strong>{value}</strong>
  </div>
);

const StreamMetricsPanel = ({ data }: { data?: StreamMetricsResponse }) => {
  if (!data) {
    return <div className={styles.muted}>Waiting for stream metrics…</div>;
  }

  const { current_metrics, alerts } = data;
  return (
    <div className={styles.streamPanel}>
      <h3>Live Throughput</h3>
      <div className={styles.streamGrid}>
        <div>
          <span>Kafka Messages / s</span>
          <strong>{current_metrics.kafka_throughput.messages_per_second}</strong>
        </div>
        <div>
          <span>Producer Rate</span>
          <strong>{current_metrics.kafka_throughput.producer_rate}</strong>
        </div>
        <div>
          <span>Spark Batch Delay</span>
          <strong>{current_metrics.spark_streaming.total_delay.toFixed(2)}s</strong>
        </div>
        <div>
          <span>Pipeline Error %</span>
          <strong>{current_metrics.data_pipeline.error_rate.toFixed(2)}%</strong>
        </div>
      </div>
      {alerts.length > 0 && (
        <ul className={styles.alertList}>
          {alerts.map((alert) => (
            <li key={alert.timestamp}>
              {alert.level.toUpperCase()}: {alert.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default AdvancedLearningPage;
