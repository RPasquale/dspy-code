import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AgentPulsePage from '@/pages/AgentPulsePage';

vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(() => <div data-testid="line-chart" />),
  Bar: vi.fn(() => <div data-testid="bar-chart" />)
}));

const mockActions = {
  counts_by_type: { training: 5, inference: 3 },
  reward_hist: { bins: [0, 0.5, 1], counts: [2, 1] },
  duration_hist: { bins: [0, 1, 2], counts: [3, 2] },
  top_actions: [],
  worst_actions: [],
  recent: [
    { id: 'a1', timestamp: 1_700_000_000, type: 'training_step', reward: 0.6, confidence: 0.8, execution_time: 1.2, environment: 'training' },
    { id: 'a2', timestamp: 1_700_000_100, type: 'inference_call', reward: 0.4, confidence: 0.7, execution_time: 0.8, environment: 'inference' }
  ],
  timestamp: 1_700_000_200
};

const mockRl = {
  metrics: {
    training_status: 'RUNNING',
    current_episode: 12,
    total_episodes: 100,
    avg_reward: 0.42,
    best_reward: 0.9,
    worst_reward: -0.1,
    epsilon: 0.15,
    learning_rate: 0.00025,
    loss: 0.02,
    q_value_mean: 0.13,
    exploration_rate: 0.15,
    replay_buffer_size: 5000,
    replay_buffer_used: 1200
  },
  reward_history: [
    { episode: 10, reward: 0.2, timestamp: '2024-01-01T00:00:00Z' },
    { episode: 11, reward: 0.5, timestamp: '2024-01-01T00:01:00Z' }
  ],
  action_stats: { approve: 12, deny: 3 },
  environment_info: { state_space_size: 128, action_space_size: 16, observation_type: 'vector', reward_range: [-1, 1] },
  timestamp: 1_700_000_200
};

const mockStatus = {
  agent: { status: 'healthy', details: 'OK' },
  ollama: { status: 'healthy' },
  kafka: { status: 'healthy' },
  containers: { status: 'healthy' },
  reddb: { status: 'healthy' },
  spark: { status: 'healthy' },
  embeddings: { status: 'warning' },
  pipeline: { status: 'idle' }
};

const streamClose = vi.fn();
const streamEvents = vi.fn((_opts, onData) => {
  onData({
    delta: {
      'agent.action': [
        {
          id: 's1',
          ts: 1_700_000_300,
          event: { action: 'training_update', reward: 0.75, signature: 'answer', status: 'completed' }
        }
      ]
    }
  });
  return { close: streamClose } as unknown as EventSource;
});

vi.mock('@/api/client', () => ({
  api: {
    getActionsAnalytics: vi.fn().mockResolvedValue(mockActions),
    getRlMetrics: vi.fn().mockResolvedValue(mockRl),
    getStatus: vi.fn().mockResolvedValue(mockStatus),
    streamEvents
  }
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('AgentPulsePage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // @ts-ignore
    global.navigator.clipboard = {
      writeText: vi.fn().mockResolvedValue(undefined)
    };
  });

  it('renders summary cards and charts', async () => {
    render(<AgentPulsePage />, { wrapper: createWrapper() });

    expect(await screen.findByText('Agent Pulse')).toBeInTheDocument();
    expect(screen.getByText('Recent Reward')).toBeInTheDocument();
    expect(screen.getByText('10 min Actions')).toBeInTheDocument();
    expect(screen.getAllByTestId('bar-chart')).toHaveLength(3);
  });

  it('displays streaming timeline entries', async () => {
    render(<AgentPulsePage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByText(/training update/i)).toBeInTheDocument());
    expect(streamEvents).toHaveBeenCalled();
  });
});
