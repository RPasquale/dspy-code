import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import MonitoringPage from '@/pages/MonitoringPage'
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

describe('MonitoringPage Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // @ts-ignore
    global.fetch.mockImplementation(mockFetch)
  })

  it('renders monitoring dashboard with all sections', async () => {
    render(<MonitoringPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('System Monitoring')).toBeInTheDocument()
    })

    // Check all monitoring sections are present
    expect(screen.getByText('CPU Usage')).toBeInTheDocument()
    expect(screen.getByText('Memory Usage')).toBeInTheDocument()
    expect(screen.getByText('Disk Usage')).toBeInTheDocument()
    expect(screen.getByText('Network I/O')).toBeInTheDocument()
  })

  it('displays real-time system metrics', async () => {
    render(<MonitoringPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('45.2%')).toBeInTheDocument() // CPU usage
    })

    // Check that metrics are displayed correctly
    expect(screen.getByText('45.2%')).toBeInTheDocument() // CPU
    expect(screen.getByText('67.8%')).toBeInTheDocument() // Memory
    expect(screen.getByText('23.1%')).toBeInTheDocument() // Disk
  })

  it('handles metric updates in real-time', async () => {
    render(<MonitoringPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('45.2%')).toBeInTheDocument()
    })

    // Simulate WebSocket message with updated metrics
    const updatedMetrics = {
      ...mockApiResponses.metrics,
      cpu_usage: 60.5,
      memory_usage: 75.2
    }

    // Mock updated API response
    // @ts-ignore
    global.fetch.mockImplementation((url) => {
      if (url.includes('metrics')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(updatedMetrics)
        })
      }
      return mockFetch(url)
    })

    // Trigger refresh
    const refreshButton = screen.getByText('Refresh')
    fireEvent.click(refreshButton)

    await waitFor(() => {
      expect(screen.getByText('60.5%')).toBeInTheDocument()
    })
  })

  it('displays performance charts', async () => {
    render(<MonitoringPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByTestId('performance-chart')).toBeInTheDocument()
    })

    // Check that charts are rendered
    expect(screen.getByTestId('performance-chart')).toBeInTheDocument()
  })

  it('handles different time ranges for performance history', async () => {
    render(<MonitoringPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('System Monitoring')).toBeInTheDocument()
    })

    // Test different time range buttons
    const timeRangeButtons = screen.getAllByRole('button')
    const hourButton = timeRangeButtons.find(button => 
      button.textContent?.includes('1h')
    )
    
    if (hourButton) {
      fireEvent.click(hourButton)
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('performance-history?timeframe=1h')
        )
      })
    }
  })

  it('shows alerts for high resource usage', async () => {
    // Mock high resource usage
    const highUsageMetrics = {
      ...mockApiResponses.metrics,
      cpu_usage: 95.0,
      memory_usage: 90.0
    }

    // @ts-ignore
    global.fetch.mockImplementation((url) => {
      if (url.includes('metrics')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(highUsageMetrics)
        })
      }
      return mockFetch(url)
    })

    render(<MonitoringPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('95.0%')).toBeInTheDocument()
    })

    // Should show warning indicators for high usage
    expect(screen.getByText('95.0%')).toHaveClass('warning')
    expect(screen.getByText('90.0%')).toHaveClass('warning')
  })

  it('handles WebSocket disconnection gracefully', async () => {
    render(<MonitoringPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('System Monitoring')).toBeInTheDocument()
    })

    // Simulate WebSocket disconnection
    const disconnectEvent = new Event('close')
    // @ts-ignore
    global.WebSocket.mock.calls[0][0].dispatchEvent(disconnectEvent)

    // Should show connection status
    expect(screen.getByText(/Disconnected/)).toBeInTheDocument()
  })

  it('filters metrics by category', async () => {
    render(<MonitoringPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('System Monitoring')).toBeInTheDocument()
    })

    // Test filtering by CPU metrics
    const cpuFilter = screen.getByText('CPU')
    fireEvent.click(cpuFilter)

    // Should only show CPU-related metrics
    expect(screen.getByText('CPU Usage')).toBeInTheDocument()
    expect(screen.queryByText('Memory Usage')).not.toBeInTheDocument()
  })

  it('exports monitoring data', async () => {
    render(<MonitoringPage />, { wrapper: createWrapper() })

    await waitFor(() => {
      expect(screen.getByText('System Monitoring')).toBeInTheDocument()
    })

    const exportButton = screen.getByText('Export Data')
    fireEvent.click(exportButton)

    // Should trigger download
    expect(screen.getByText('Exporting...')).toBeInTheDocument()
  })
})
