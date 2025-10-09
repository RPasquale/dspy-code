import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import App from '@/App';
import { ThemeProvider } from '@/contexts/ThemeContext';

vi.mock('@/pages/AgentPulsePage', () => ({
  default: () => <div data-testid="agent-pulse-page">Agent Pulse Page</div>
}));

vi.mock('@/pages/PerformanceInsightsPage', () => ({
  default: () => <div data-testid="performance-page">Performance Page</div>
}));

vi.mock('@/pages/DataStreamsPage', () => ({
  default: () => <div data-testid="streams-page">Streams Page</div>
}));

vi.mock('@/pages/SystemOverviewPage', () => ({
  default: () => <div data-testid="system-page">System Page</div>
}));

vi.mock('@/components/ProfileSwitcher', () => ({
  default: () => <div data-testid="profile-switcher">Profile</div>
}));

vi.mock('@/components/StatusPill', () => ({
  default: ({ status }: { status?: string }) => <span data-testid="status-pill">{status || 'ok'}</span>
}));

vi.mock('@/api/client', () => ({
  api: {
    getStatus: vi.fn().mockResolvedValue({
      agent: { status: 'healthy', details: 'OK' },
      kafka: { status: 'healthy', details: 'OK' },
      pipeline: { status: 'idle', details: 'Idle' }
    }),
    getBusMetrics: vi.fn().mockResolvedValue({
      dlq: { total: 0 },
      bus: { topics: {} },
      alerts: []
    })
  }
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false }
    }
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('Agent Console App Shell', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders brand and default page', async () => {
    render(<App />, { wrapper: createWrapper() });
    expect(screen.getByText('Agent Console')).toBeInTheDocument();
    expect(await screen.findByTestId('agent-pulse-page')).toBeInTheDocument();
  });

  it('shows core navigation items', () => {
    render(<App />, { wrapper: createWrapper() });
    expect(screen.getByText('Training Progress')).toBeInTheDocument();
    expect(screen.getByText('Results & Insights')).toBeInTheDocument();
    expect(screen.getByText('Data Sources')).toBeInTheDocument();
    expect(screen.getByText('Health Status')).toBeInTheDocument();
  });

  it('navigates between pages', async () => {
    const user = userEvent.setup();
    render(<App />, { wrapper: createWrapper() });

    expect(await screen.findByTestId('agent-pulse-page')).toBeInTheDocument();

    await user.click(screen.getByText('Results & Insights'));
    expect(screen.getByTestId('performance-page')).toBeInTheDocument();

    await user.click(screen.getByText('Data Sources'));
    expect(screen.getByTestId('streams-page')).toBeInTheDocument();
  });
});
