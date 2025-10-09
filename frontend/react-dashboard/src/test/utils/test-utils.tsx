import { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { vi } from 'vitest'
import { ToastProvider } from '@/components/ToastProvider'

// Create a custom render function that includes providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ToastProvider>
          {children}
        </ToastProvider>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options })

// Re-export everything
export * from '@testing-library/react'
export { customRender as render }

// Mock data generators
export const generateMockStatus = (overrides = {}) => ({
  status: 'healthy',
  timestamp: '2025-09-21T12:00:00Z',
  version: '1.0.0',
  uptime: 3600,
  ...overrides
})

export const generateMockMetrics = (overrides = {}) => ({
  cpu_usage: 45.2,
  memory_usage: 67.8,
  disk_usage: 23.1,
  network_io: {
    bytes_in: 1024000,
    bytes_out: 512000
  },
  ...overrides
})

export const generateMockRLMetrics = (overrides = {}) => ({
  episodes: 1000,
  avg_reward: 0.75,
  average_reward: 0.75,
  success_rate: 0.68,
  exploration_rate: 0.1,
  learning_rate: 0.001,
  recent_rewards: [0.8, 0.7, 0.9, 0.6, 0.85],
  timestamp: '2025-09-21T12:00:00Z',
  ...overrides
})

export const generateMockBusMetrics = (overrides = {}) => ({
  total_messages: 15000,
  active_consumers: 5,
  queue_depth: 120,
  processing_rate: 45.2,
  error_rate: 0.02,
  ...overrides
})

export const generateMockLogs = (overrides = {}) => ({
  logs: [
    {
      timestamp: '2025-09-21T11:00:00Z',
      level: 'INFO',
      message: 'Agent started successfully',
      status: 'success'
    },
    {
      timestamp: '2025-09-21T11:05:00Z',
      level: 'INFO',
      message: 'Processing task: code_generation',
      status: 'processing'
    }
  ],
  total_logs: 2,
  timestamp: '2025-09-21T12:00:00Z',
  ...overrides
})

export const generateMockKafkaTopics = (overrides = {}) => ({
  topics: [
    {
      name: 'agent-actions',
      partitions: 3,
      messages: 5000,
      consumers: 2
    },
    {
      name: 'learning-events',
      partitions: 2,
      messages: 3000,
      consumers: 1
    }
  ],
  total_topics: 2,
  timestamp: '2025-09-21T12:00:00Z',
  ...overrides
})

// Test helpers
export const waitForLoadingToFinish = () => 
  new Promise(resolve => setTimeout(resolve, 0))

export const mockFetchResponse = (data: any, ok = true, status = 200) => ({
  ok,
  status,
  json: () => Promise.resolve(data)
})

export const mockFetchError = (message = 'Network error') => 
  Promise.reject(new Error(message))

// Mock WebSocket helpers
export const mockWebSocket = {
  close: vi.fn(),
  send: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  readyState: WebSocket.OPEN
}

export const simulateWebSocketMessage = (message: any) => {
  const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
    call => call[0] === 'message'
  )?.[1]
  if (messageHandler) {
    messageHandler({ data: JSON.stringify(message) })
  }
}

export const simulateWebSocketOpen = () => {
  const openHandler = mockWebSocket.addEventListener.mock.calls.find(
    call => call[0] === 'open'
  )?.[1]
  if (openHandler) {
    openHandler()
  }
}

export const simulateWebSocketClose = () => {
  const closeHandler = mockWebSocket.addEventListener.mock.calls.find(
    call => call[0] === 'close'
  )?.[1]
  if (closeHandler) {
    closeHandler()
  }
}

export const simulateWebSocketError = (error = new Error('Connection failed')) => {
  const errorHandler = mockWebSocket.addEventListener.mock.calls.find(
    call => call[0] === 'error'
  )?.[1]
  if (errorHandler) {
    errorHandler(error)
  }
}
