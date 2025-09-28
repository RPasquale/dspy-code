import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import userEvent from '@testing-library/user-event'
import App from '@/App'

// Mock all the page components to avoid complex dependencies
vi.mock('@/pages/OverviewPage', () => ({
  default: () => <div data-testid="overview-page">Overview Page</div>
}))

vi.mock('@/pages/MonitoringPage', () => ({
  default: () => <div data-testid="monitoring-page">Monitoring Page</div>
}))

vi.mock('@/pages/TrainingPage', () => ({
  default: () => <div data-testid="training-page">Training Page</div>
}))

vi.mock('@/pages/ActionsPage', () => ({
  default: () => <div data-testid="actions-page">Actions Page</div>
}))

vi.mock('@/pages/RewardsPage', () => ({
  default: () => <div data-testid="rewards-page">Rewards Page</div>
}))

vi.mock('@/pages/SignaturesPage', () => ({
  default: () => <div data-testid="signatures-page">Signatures Page</div>
}))

vi.mock('@/pages/VerifiersPage', () => ({
  default: () => <div data-testid="verifiers-page">Verifiers Page</div>
}))

vi.mock('@/pages/SystemMapPage', () => ({
  default: () => <div data-testid="system-map-page">System Map Page</div>
}))

vi.mock('@/pages/AdvancedLearningPage', () => ({
  default: () => <div data-testid="advanced-learning-page">Advanced Learning Page</div>
}))

vi.mock('@/pages/RealtimeMonitoringPage', () => ({
  default: () => <div data-testid="realtime-monitoring-page">Realtime Monitoring Page</div>
}))

vi.mock('@/pages/BusMetricsPage', () => ({
  default: () => <div data-testid="bus-metrics-page">Bus Metrics Page</div>
}))

// Mock the API client
vi.mock('@/api/client', () => ({
  api: {
    getBusMetrics: vi.fn().mockResolvedValue({
      alerts: [],
      thresholds: { backpressure_depth: 100, dlq_min: 1 },
      dlq: { total: 0 }
    })
  }
}))

// Mock CSS modules
vi.mock('@/styles/AppLayout.module.css', () => ({
  default: {
    appShell: 'appShell',
    header: 'header',
    brand: 'brand',
    badgeDanger: 'badgeDanger',
    badgeOk: 'badgeOk',
    navLinks: 'navLinks',
    activeLink: 'activeLink',
    link: 'link',
    mainContent: 'mainContent'
  }
}))

// Create a wrapper for React Query and Router
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  )
}

describe('DSPy Agent Frontend', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the main application', () => {
    render(<App />, { wrapper: createWrapper() })
    
    expect(screen.getByText('DSPy Monitor')).toBeInTheDocument()
  })

  it('renders navigation links', () => {
    render(<App />, { wrapper: createWrapper() })
    
    expect(screen.getByText('Overview')).toBeInTheDocument()
    expect(screen.getByText('Monitoring')).toBeInTheDocument()
    expect(screen.getByText('Training')).toBeInTheDocument()
    expect(screen.getByText('Actions')).toBeInTheDocument()
    expect(screen.getByText('Rewards')).toBeInTheDocument()
    expect(screen.getByText('Signatures')).toBeInTheDocument()
    expect(screen.getByText('Verifiers')).toBeInTheDocument()
    expect(screen.getByText('System Map')).toBeInTheDocument()
    expect(screen.getByText('Advanced Learning')).toBeInTheDocument()
    expect(screen.getByText('Real-Time')).toBeInTheDocument()
    expect(screen.getByText('Bus')).toBeInTheDocument()
  })

  it('renders overview page by default', () => {
    render(<App />, { wrapper: createWrapper() })
    
    expect(screen.getByTestId('overview-page')).toBeInTheDocument()
  })

  it('handles navigation between pages', async () => {
    const user = userEvent.setup()
    render(<App />, { wrapper: createWrapper() })
    
    // Initially shows overview
    expect(screen.getByTestId('overview-page')).toBeInTheDocument()
    
    // Click on monitoring - this will navigate to /monitoring route
    const monitoringLink = screen.getByText('Monitoring')
    await user.click(monitoringLink)
    
    // Wait for navigation to complete and check if monitoring page is rendered
    expect(screen.getByTestId('monitoring-page')).toBeInTheDocument()
  })

  it('displays header with title', () => {
    render(<App />, { wrapper: createWrapper() })
    
    expect(screen.getByText('DSPy Monitor')).toBeInTheDocument()
  })

  it('applies correct CSS classes', () => {
    render(<App />, { wrapper: createWrapper() })
    
    // Find the main app container div (the one with appShell class)
    const appElement = document.querySelector('.appShell')
    expect(appElement).toBeInTheDocument()
    expect(appElement).toHaveClass('appShell')
  })
})
