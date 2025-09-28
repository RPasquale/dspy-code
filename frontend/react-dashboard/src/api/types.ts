export type ServiceHealth = 'healthy' | 'unhealthy' | 'warning' | 'unknown' | string;

export interface ServiceStatus {
  status: ServiceHealth;
  details?: string;
}

export interface StatusResponse {
  agent: ServiceStatus;
  ollama: ServiceStatus;
  kafka: ServiceStatus;
  containers: ServiceStatus;
  reddb?: ServiceStatus;
  spark?: ServiceStatus;
  embeddings?: ServiceStatus;
  pipeline?: ServiceStatus;
  learning_active?: boolean;
  auto_training?: boolean;
  timestamp?: number;
}

export interface LogsResponse {
  logs: string;
}

export interface MetricsResponse {
  timestamp: number;
  containers: number;
  memory_usage: string;
  response_time: number;
}

export interface ContainersResponse {
  containers: string;
}

export interface CommandResponse {
  success: boolean;
  output?: string;
  error?: string;
}

export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  response: string;
  timestamp: number;
  processing_time: number;
  confidence: number;
}

export interface SignatureSummary {
  name: string;
  performance: number;
  iterations: number;
  type: string;
  last_updated: string;
  success_rate?: number;
  avg_response_time?: number;
  memory_usage?: string;
  active?: boolean;
}

export interface SignaturesResponse {
  signatures: SignatureSummary[];
  total_active: number;
  avg_performance: number;
  timestamp: number;
}

export interface VerifierSummary {
  name: string;
  accuracy: number;
  status: string;
  checks_performed: number;
  issues_found: number;
  last_run: string;
  avg_execution_time: number;
}

export interface VerifiersResponse {
  verifiers: VerifierSummary[];
  total_active: number;
  avg_accuracy: number;
  total_checks: number;
  total_issues: number;
  timestamp: number;
}

export interface LearningMetricsResponse {
  performance_over_time: {
    timestamps: string[];
    overall_performance: number[];
    training_accuracy: number[];
    validation_accuracy: number[];
  };
  signature_performance: Record<string, number[]>;
  learning_stats: {
    total_training_examples: number;
    successful_optimizations: number;
    failed_optimizations: number;
    avg_improvement_per_iteration: number;
    current_learning_rate: number;
  };
  resource_usage: {
    memory_usage: number[];
    cpu_usage: number[];
    gpu_usage: number[];
  };
  timestamp: number;
}

export interface PerformanceHistoryResponse {
  timestamps: string[];
  metrics: {
    response_times: number[];
    success_rates: number[];
    throughput: number[];
    error_rates: number[];
  };
  timeframe: string;
  interval: string;
  timestamp: number;
}

export interface KafkaTopic {
  name: string;
  partitions: number;
  messages_per_minute: number;
  total_messages: number;
  consumer_lag: number;
  retention_ms: number;
  size_bytes: number;
  producers: string[];
  consumers: string[];
}

export interface KafkaTopicsResponse {
  topics: KafkaTopic[];
  broker_info: {
    cluster_id: string;
    broker_count: number;
    controller_id: number;
    total_partitions: number;
    under_replicated_partitions: number;
    offline_partitions: number;
  };
  timestamp: number;
}

export interface SparkWorker {
  id: string;
  host: string;
  port: number;
  status: string;
  cores: number;
  cores_used: number;
  memory: string;
  memory_used: string;
  last_heartbeat: string;
  executors: number;
}

export interface SparkApplication {
  id: string;
  name: string;
  status: string;
  duration: string;
  cores: number;
  memory_per_executor: string;
  executors: number;
}

export interface SparkWorkersResponse {
  master: {
    status: string;
    workers: number;
    cores_total: number;
    cores_used: number;
    memory_total: string;
    memory_used: string;
    applications_running: number;
    applications_completed: number;
  };
  workers: SparkWorker[];
  applications: SparkApplication[];
  cluster_metrics: {
    total_cores: number;
    used_cores: number;
    total_memory: string;
    used_memory: string;
    cpu_utilization: number;
  };
  timestamp: number;
}

export interface RlMetricsResponse {
  metrics: {
    training_status: string;
    current_episode: number;
    total_episodes: number;
    avg_reward: number;
    best_reward: number;
    worst_reward: number;
    epsilon: number;
    learning_rate: number;
    loss: number;
    q_value_mean: number;
    exploration_rate: number;
    replay_buffer_size: number;
    replay_buffer_used: number;
  };
  reward_history: {
    episode: number;
    reward: number;
    timestamp: string;
  }[];
  action_stats: Record<string, number>;
  environment_info: {
    state_space_size: number;
    action_space_size: number;
    observation_type: string;
    reward_range: [number, number];
  };
  timestamp: number;
}

export interface SystemNode {
  id: string;
  name: string;
  type: string;
  status: string;
  host: string;
  port: number;
  cpu_usage: number;
  memory_usage: number;
  connections: string[];
  model?: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface DataFlow {
  source: string;
  target: string;
  type: string;
  throughput: number;
  latency: number;
}

export interface SystemTopologyResponse {
  nodes: SystemNode[];
  data_flows: DataFlow[];
  cluster_info: {
    total_nodes: number;
    healthy_nodes: number;
    total_cpu_cores: number;
    total_memory_gb: number;
    network_throughput_mbps: number;
  };
  timestamp: number;
}

export interface StreamMetricsResponse {
  current_metrics: {
    kafka_throughput: {
      messages_per_second: number;
      bytes_per_second: number;
      producer_rate: number;
      consumer_rate: number;
    };
    spark_streaming: {
      batch_duration: string;
      processing_time: number;
      scheduling_delay: number;
      total_delay: number;
      records_per_batch: number;
      batches_completed: number;
    };
    data_pipeline: {
      input_rate: number;
      output_rate: number;
      error_rate: number;
      backpressure: boolean;
      queue_depth: number;
    };
    network_io: {
      bytes_in_per_sec: number;
      bytes_out_per_sec: number;
      packets_in_per_sec: number;
      packets_out_per_sec: number;
      connections_active: number;
    };
  };
  time_series: {
    timestamp: string;
    throughput: number;
    latency: number;
    error_rate: number;
    cpu_usage: number;
  }[];
  alerts: {
    level: string;
    message: string;
    timestamp: string;
  }[];
  timestamp: number;
}

// Aggregated overview payload (batched)
export interface OverviewResponse {
  status: StatusResponse;
  metrics: MetricsResponse;
  rl: RlMetricsResponse;
  bus: BusMetricsResponse;
  timestamp: number;
}

export interface BusMetricsResponse {
  bus: {
    dlq_total?: number;
    topics?: Record<string, number[]>; // per-topic queue sizes for non-group subs
    groups?: Record<string, Record<string, number[]>>; // topic -> groupId -> sizes
  };
  dlq: {
    total: number;
    by_topic: Record<string, number>;
    last_ts: number | null;
  };
  alerts: { level: string; message: string; timestamp: number }[];
  history?: {
    timestamps: number[];
    queue_max_depth: number[];
    dlq_total: number[];
  };
  thresholds?: {
    backpressure_depth: number;
    dlq_min: number;
  };
  timestamp: number;
}

export interface UpdateConfigRequest {
  type: string;
  value: unknown;
}

export interface UpdateConfigResponse {
  success: boolean;
  config_type?: string;
  new_value?: unknown;
  applied_at?: number;
  restart_required?: boolean;
  error?: string;
}

export interface OptimizeSignatureRequest {
  signature_name: string;
  type: string;
}

export interface OptimizeSignatureResponse {
  signature_name: string;
  optimization_type: string;
  success: boolean;
  improvements: {
    performance_gain: number;
    accuracy_improvement: number;
    response_time_reduction: number;
  };
  changes_made: string[];
  new_metrics: {
    performance_score: number;
    success_rate: number;
    avg_response_time: number;
  };
  timestamp: number;
  error?: string;
}

// Rewards / RL config editing
export interface RewardsConfigResponse {
  policy: string;
  epsilon: number;
  ucb_c: number;
  n_envs: number;
  puffer: boolean;
  weights: Record<string, number>;
  penalty_kinds: string[];
  clamp01_kinds: string[];
  scales: Record<string, [number, number]>;
  actions: string[];
  test_cmd?: string | null;
  lint_cmd?: string | null;
  build_cmd?: string | null;
  timeout_sec?: number | null;
  path?: string;
}

export type UpdateRewardsConfigRequest = Partial<RewardsConfigResponse> & { workspace?: string };
export interface UpdateRewardsConfigResponse {
  success: boolean;
  updated?: boolean;
  error?: string;
}

// Signature detail + update
export interface SignatureDetailResponse {
  metrics: {
    name: string;
    performance: number;
    success_rate: number;
    avg_response_time: number;
    memory_usage: string;
    iterations: number;
    last_updated: string;
    type: string;
    active: boolean;
  };
  optimization_history: Record<string, unknown>[];
  trend: Record<string, unknown>[];
  policy_summary?: {
    tools: Record<string, { last24h: { mean: number; count: number }; last7d: { mean: number; count: number }; delta?: number }>;
    rules: { regex: string; prefer_tools?: string[]; deny_tools?: string[] }[];
    rule_hits: { regex: string; hits24h: number; hits7d: number }[];
  };
  timestamp: number;
  error?: string;
}

export interface UpdateSignatureRequest {
  name: string;
  active?: boolean;
  type?: string;
}
export interface UpdateSignatureResponse {
  success?: boolean;
  updated?: { name: string; type: string; active: boolean; last_updated: string };
  error?: string;
}

export interface UpdateVerifierRequest {
  name: string;
  status: string;
}
export interface UpdateVerifierResponse {
  success?: boolean;
  updated?: { name: string; status: string; last_run: string };
  error?: string;
}

// Actions analytics
export interface ActionsAnalyticsResponse {
  counts_by_type: Record<string, number>;
  reward_hist: { bins: number[]; counts: number[] } | { bins: unknown[]; counts: number[] };
  duration_hist: { bins: number[]; counts: number[] } | { bins: unknown[]; counts: number[] };
  top_actions: {
    id: string;
    timestamp: number;
    type: string;
    reward: number;
    confidence: number;
    execution_time: number;
    environment: string;
  }[];
  worst_actions: ActionsAnalyticsResponse['top_actions'];
  recent: ActionsAnalyticsResponse['top_actions'];
  timestamp: number;
  error?: string;
}
