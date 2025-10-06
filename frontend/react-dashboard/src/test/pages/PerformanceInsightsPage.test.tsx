import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import PerformanceInsightsPage from '@/pages/PerformanceInsightsPage';

vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(() => <div data-testid="line-chart" />)
}));

const signatures = {
  signatures: [
    { name: 'answer_v1', performance: 82.5, success_rate: 78.2, iterations: 34, type: 'analysis', last_updated: '2024-01-02', avg_response_time: 1.2, memory_usage: '200MB', active: true },
    { name: 'verify', performance: 76.1, success_rate: 70.4, iterations: 20, type: 'verification', last_updated: '2024-01-01', avg_response_time: 1.8, memory_usage: '180MB', active: true }
  ],
  total_active: 2,
  avg_performance: 79.3,
  timestamp: 1_700_000_000
};

const verifiers = {
  verifiers: [
    { name: 'lint', accuracy: 91.2, status: 'active', checks_performed: 200, issues_found: 18, last_run: '2024-01-02', avg_execution_time: 0.5 }
  ],
  total_active: 1,
  avg_accuracy: 91.2,
  total_checks: 200,
  total_issues: 18,
  timestamp: 1_700_000_000
};

const learning = {
  performance_over_time: {
    timestamps: ['2024-01-01T00:00:00Z', '2024-01-01T01:00:00Z'],
    overall_performance: [70, 80],
    training_accuracy: [72, 82],
    validation_accuracy: [68, 78]
  },
  signature_performance: {},
  learning_stats: {
    total_training_examples: 1200,
    successful_optimizations: 45,
    failed_optimizations: 5,
    avg_improvement_per_iteration: 0.4,
    current_learning_rate: 0.0002
  },
  resource_usage: { memory_usage: [50], cpu_usage: [40], gpu_usage: [30] },
  timestamp: 1_700_000_000
};

const rl = {
  metrics: {
    training_status: 'RUNNING',
    current_episode: 12,
    total_episodes: 80,
    avg_reward: 0.52,
    best_reward: 0.9,
    worst_reward: -0.2,
    epsilon: 0.1,
    learning_rate: 0.0003,
    loss: 0.05,
    q_value_mean: 0.2,
    exploration_rate: 0.1,
    replay_buffer_size: 2000,
    replay_buffer_used: 600
  },
  reward_history: [
    { episode: 10, reward: 0.4, timestamp: '2024-01-01T00:00:00Z' },
    { episode: 11, reward: 0.6, timestamp: '2024-01-01T00:01:00Z' }
  ],
  action_stats: {},
  environment_info: { state_space_size: 64, action_space_size: 8, observation_type: 'vector', reward_range: [-1, 1] },
  timestamp: 1_700_000_000
};

const performanceHistory = {
  timestamps: ['2024-01-01T00:00:00Z', '2024-01-01T01:00:00Z'],
  metrics: {
    response_times: [0.8, 0.7],
    success_rates: [0.6, 0.72],
    throughput: [20, 32],
    error_rates: [0.1, 0.05]
  },
  timeframe: '7d',
  interval: 'hour',
  timestamp: 1_700_000_000
};

const actionsAnalytics = {
  counts_by_type: {},
  reward_hist: { bins: [], counts: [] },
  duration_hist: { bins: [], counts: [] },
  top_actions: [],
  worst_actions: [],
  recent: [
    { id: 'r1', timestamp: 1_700_000_000, type: 'training_step', reward: 0.5, confidence: 0.7, execution_time: 0.5, environment: 'training' }
  ],
  timestamp: 1_700_000_001
};

vi.mock('@/api/client', () => ({
  api: {
    getSignatures: vi.fn().mockResolvedValue(signatures),
    getVerifiers: vi.fn().mockResolvedValue(verifiers),
    getLearningMetrics: vi.fn().mockResolvedValue(learning),
    getRlMetrics: vi.fn().mockResolvedValue(rl),
    getPerformanceHistory: vi.fn().mockResolvedValue(performanceHistory),
    getActionsAnalytics: vi.fn().mockResolvedValue(actionsAnalytics)
  }
}));

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {children}
  </QueryClientProvider>
);

describe('PerformanceInsightsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders key sections with data', async () => {
    render(<PerformanceInsightsPage />, { wrapper: Wrapper });

    expect(await screen.findByText('Performance Command Center')).toBeInTheDocument();
    expect(screen.getByText('Avg Signature Performance')).toBeInTheDocument();
    expect(screen.getByText('Verifier Accuracy')).toBeInTheDocument();
    expect(screen.getByText('RL Avg Reward')).toBeInTheDocument();
    expect(screen.getAllByTestId('line-chart').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('answer_v1')).toBeInTheDocument();
    expect(screen.getByText('lint')).toBeInTheDocument();
  });
});
