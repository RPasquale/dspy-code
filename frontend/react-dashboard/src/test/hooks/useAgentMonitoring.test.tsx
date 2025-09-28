import { renderHook, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useAgentMonitoring } from '@/hooks/useAgentMonitoring'
import { mockApiResponses, mockFetch } from '../__mocks__/api'

// Mock fetch globally
global.fetch = vi.fn()

// Create a wrapper for React Query
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe('useAgentMonitoring', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // @ts-ignore
    global.fetch.mockImplementation(mockFetch)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('fetches agent status successfully', async () => {
    const wrapper = createWrapper()
    const { result } = renderHook(() => useAgentMonitoring(), { wrapper })

    await waitFor(() => {
      expect(result.current.status.isSuccess).toBe(true)
    })

    expect(result.current.status.data).toEqual(mockApiResponses.status)
  })

  it('fetches metrics successfully', async () => {
    const wrapper = createWrapper()
    const { result } = renderHook(() => useAgentMonitoring(), { wrapper })

    await waitFor(() => {
      expect(result.current.metrics.isSuccess).toBe(true)
    })

    expect(result.current.metrics.data).toEqual(mockApiResponses.metrics)
  })

  it('fetches RL metrics successfully', async () => {
    const wrapper = createWrapper()
    const { result } = renderHook(() => useAgentMonitoring(), { wrapper })

    await waitFor(() => {
      expect(result.current.rlMetrics.isSuccess).toBe(true)
    })

    expect(result.current.rlMetrics.data).toEqual(mockApiResponses.rlMetrics)
  })

  it('fetches bus metrics successfully', async () => {
    const wrapper = createWrapper()
    const { result } = renderHook(() => useAgentMonitoring(), { wrapper })

    await waitFor(() => {
      expect(result.current.busMetrics.isSuccess).toBe(true)
    })

    expect(result.current.busMetrics.data).toEqual(mockApiResponses.busMetrics)
  })

  it('handles API errors gracefully', async () => {
    // @ts-ignore
    global.fetch.mockImplementation(() => 
      Promise.resolve({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Internal Server Error' })
      })
    )

    const wrapper = createWrapper()
    const { result } = renderHook(() => useAgentMonitoring(), { wrapper })

    await waitFor(() => {
      expect(result.current.status.isError).toBe(true)
    })

    expect(result.current.status.error).toBeDefined()
  })

  it('provides loading states', () => {
    const wrapper = createWrapper()
    const { result } = renderHook(() => useAgentMonitoring(), { wrapper })

    expect(result.current.status.isLoading).toBe(true)
    expect(result.current.metrics.isLoading).toBe(true)
    expect(result.current.rlMetrics.isLoading).toBe(true)
    expect(result.current.busMetrics.isLoading).toBe(true)
  })

  it('refetches data on interval', async () => {
    const wrapper = createWrapper()
    const { result } = renderHook(() => useAgentMonitoring(), { wrapper })

    await waitFor(() => {
      expect(result.current.status.isSuccess).toBe(true)
    })

    // Verify fetch was called
    expect(global.fetch).toHaveBeenCalledWith('/api/status')
    expect(global.fetch).toHaveBeenCalledWith('/api/metrics')
    expect(global.fetch).toHaveBeenCalledWith('/api/rl-metrics')
    expect(global.fetch).toHaveBeenCalledWith('/api/bus-metrics')
  })

  it('handles network errors', async () => {
    // @ts-ignore
    global.fetch.mockImplementation(() => 
      Promise.reject(new Error('Network error'))
    )

    const wrapper = createWrapper()
    const { result } = renderHook(() => useAgentMonitoring(), { wrapper })

    await waitFor(() => {
      expect(result.current.status.isError).toBe(true)
    })

    expect(result.current.status.error).toBeDefined()
  })

  it('provides error states for failed requests', async () => {
    // @ts-ignore
    global.fetch.mockImplementation((url) => {
      if (url.includes('status')) {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: () => Promise.resolve({ error: 'Not found' })
        })
      }
      return mockFetch(url)
    })

    const wrapper = createWrapper()
    const { result } = renderHook(() => useAgentMonitoring(), { wrapper })

    await waitFor(() => {
      expect(result.current.status.isError).toBe(true)
    })

    expect(result.current.status.error).toBeDefined()
    expect(result.current.metrics.isSuccess).toBe(true) // Other requests should still work
  })
})
