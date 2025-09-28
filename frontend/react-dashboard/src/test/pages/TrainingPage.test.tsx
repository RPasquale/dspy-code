import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import TrainingPage from '@/pages/TrainingPage'
import { mockApiResponses, mockFetch } from '../__mocks__/api'

// Mock fetch globally
global.fetch = vi.fn()

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

describe('TrainingPage Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // @ts-ignore
    global.fetch.mockImplementation(mockFetch)
  })

  it('renders training dashboard with RL metrics', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('Training Progress')).toBeInTheDocument()
    })

    // Check that training components are rendered
    expect(screen.getByText('Episodes')).toBeInTheDocument()
    expect(screen.getByText('Average Reward')).toBeInTheDocument()
    expect(screen.getByText('Success Rate')).toBeInTheDocument()
    expect(screen.getByText('Learning Progress')).toBeInTheDocument()
  })

  it('displays RL training metrics correctly', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('1000')).toBeInTheDocument() // episodes
    })

    // Check that RL metrics are displayed
    expect(screen.getByText('1000')).toBeInTheDocument() // episodes
    expect(screen.getByText('75%')).toBeInTheDocument() // avg reward
    expect(screen.getByText('68%')).toBeInTheDocument() // success rate
    expect(screen.getByText('10%')).toBeInTheDocument() // exploration rate
  })

  it('shows learning progress chart', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByTestId('learning-chart')).toBeInTheDocument()
    })

    // Check that learning chart is rendered
    expect(screen.getByTestId('learning-chart')).toBeInTheDocument()
  })

  it('handles training parameter updates', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('Training Progress')).toBeInTheDocument()
    })

    // Test updating learning rate
    const learningRateInput = screen.getByDisplayValue('0.001')
    fireEvent.change(learningRateInput, { target: { value: '0.01' } })

    // Should update the learning rate
    expect(learningRateInput).toHaveValue('0.01')
  })

  it('starts and stops training', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('Training Progress')).toBeInTheDocument()
    })

    // Test starting training
    const startButton = screen.getByText('Start Training')
    fireEvent.click(startButton)

    expect(screen.getByText('Training...')).toBeInTheDocument()

    // Test stopping training
    const stopButton = screen.getByText('Stop Training')
    fireEvent.click(stopButton)

    expect(screen.getByText('Training Stopped')).toBeInTheDocument()
  })

  it('displays recent rewards history', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('Recent Rewards')).toBeInTheDocument()
    })

    // Check that recent rewards are displayed
    expect(screen.getByText('0.8')).toBeInTheDocument()
    expect(screen.getByText('0.7')).toBeInTheDocument()
    expect(screen.getByText('0.9')).toBeInTheDocument()
  })

  it('handles training configuration changes', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('Training Progress')).toBeInTheDocument()
    })

    // Test changing exploration rate
    const explorationRateSlider = screen.getByRole('slider')
    fireEvent.change(explorationRateSlider, { target: { value: '0.2' } })

    expect(explorationRateSlider).toHaveValue('0.2')
  })

  it('shows training performance over time', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByTestId('performance-chart')).toBeInTheDocument()
    })

    // Check that performance chart is rendered
    expect(screen.getByTestId('performance-chart')).toBeInTheDocument()
  })

  it('handles training errors gracefully', async () => {
    // Mock training error
    // @ts-ignore
    global.fetch.mockImplementation((url) => {
      if (url.includes('rl-metrics')) {
        return Promise.resolve({
          ok: false,
          status: 500,
          json: () => Promise.resolve({ error: 'Training failed' })
        })
      }
      return mockFetch(url)
    })

    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText(/Error/)).toBeInTheDocument()
    })
  })

  it('exports training data', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('Training Progress')).toBeInTheDocument()
    })

    const exportButton = screen.getByText('Export Training Data')
    fireEvent.click(exportButton)

    // Should trigger download
    expect(screen.getByText('Exporting...')).toBeInTheDocument()
  })

  it('resets training progress', async () => {
    render(<TrainingPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('Training Progress')).toBeInTheDocument()
    })

    const resetButton = screen.getByText('Reset Training')
    fireEvent.click(resetButton)

    // Should confirm reset
    expect(screen.getByText('Are you sure?')).toBeInTheDocument()

    const confirmButton = screen.getByText('Yes, Reset')
    fireEvent.click(confirmButton)

    // Should reset metrics
    expect(screen.getByText('0')).toBeInTheDocument() // episodes reset
  })
})
