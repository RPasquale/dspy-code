import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import OverviewPage from '@/pages/OverviewPage'
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

describe('OverviewPage Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // @ts-ignore
    global.fetch.mockImplementation(mockFetch)
  })

  it('renders all dashboard components together', async () => {
    render(<OverviewPage />, { wrapper: createWrapper() })

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('Agent Status')).toBeInTheDocument()
    })

    // Check that all major components are rendered
    expect(screen.getByText('Agent Status')).toBeInTheDocument()
    expect(screen.getByText('System Metrics')).toBeInTheDocument()
    expect(screen.getByText('Learning Progress')).toBeInTheDocument()
    expect(screen.getByText('Bus Metrics')).toBeInTheDocument()
  })

  it('displays real-time data from API', async () => {
    render(<OverviewPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('healthy')).toBeInTheDocument()
    })

    // Check that API data is displayed
    expect(screen.getByText('Version: 1.0.0')).toBeInTheDocument()
    expect(screen.getByText('75%')).toBeInTheDocument() // avg reward
    expect(screen.getByText('68%')).toBeInTheDocument() // success rate
  })

  it('handles loading states', () => {
    // Mock slow API response
    // @ts-ignore
    global.fetch.mockImplementation(() => 
      new Promise(resolve => 
        setTimeout(() => resolve(mockFetch('/api/status')), 100)
      )
    )

    render(<OverviewPage />, { wrapper: createWrapper() })

    // Should show loading states initially
    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })

  it('handles API errors gracefully', async () => {
    // Mock API error
    // @ts-ignore
    global.fetch.mockImplementation(() => 
      Promise.resolve({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Server Error' })
      })
    )

    render(<OverviewPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText(/Error/)).toBeInTheDocument()
    })
  })

  it('updates data on refresh', async () => {
    render(<OverviewPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('healthy')).toBeInTheDocument()
    })

    // Simulate refresh
    const refreshButton = screen.getByText('Refresh')
    refreshButton.click()

    // Should show loading state during refresh
    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })

  it('displays charts and visualizations', async () => {
    render(<OverviewPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByTestId('learning-chart')).toBeInTheDocument()
    })

    // Check that charts are rendered
    expect(screen.getByTestId('learning-chart')).toBeInTheDocument()
  })

  it('handles WebSocket connection', async () => {
    render(<OverviewPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('Agent Status')).toBeInTheDocument()
    })

    // WebSocket should be initialized
    expect(global.WebSocket).toHaveBeenCalled()
  })
})
