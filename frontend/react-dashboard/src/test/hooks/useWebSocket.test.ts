import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useWebSocket } from '@/hooks/useWebSocket'

// Mock WebSocket
const mockWebSocket = {
  close: vi.fn(),
  send: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  readyState: WebSocket.OPEN
}

// Mock WebSocket constructor
global.WebSocket = vi.fn(() => mockWebSocket) as any

describe('useWebSocket', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset WebSocket readyState
    mockWebSocket.readyState = WebSocket.OPEN
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('connects to WebSocket on mount', () => {
    renderHook(() => useWebSocket('ws://localhost:8081'))

    expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:8081')
  })

  it('handles connection open event', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8081'))

    // Simulate connection open
    act(() => {
      const openHandler = mockWebSocket.addEventListener.mock.calls.find(
        call => call[0] === 'open'
      )?.[1]
      if (openHandler) openHandler()
    })

    expect(result.current.isConnected).toBe(true)
  })

  it('handles connection close event', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8081'))

    // Simulate connection close
    act(() => {
      const closeHandler = mockWebSocket.addEventListener.mock.calls.find(
        call => call[0] === 'close'
      )?.[1]
      if (closeHandler) closeHandler()
    })

    expect(result.current.isConnected).toBe(false)
  })

  it('handles incoming messages', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8081'))

    const testMessage = { type: 'metrics', data: { cpu: 50 } }

    // Simulate incoming message
    act(() => {
      const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
        call => call[0] === 'message'
      )?.[1]
      if (messageHandler) {
        messageHandler({ data: JSON.stringify(testMessage) })
      }
    })

    expect(result.current.lastMessage).toEqual(testMessage)
  })

  it('sends messages through WebSocket', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8081'))

    const testMessage = { type: 'ping', data: 'test' }

    act(() => {
      result.current.sendMessage(testMessage)
    })

    expect(mockWebSocket.send).toHaveBeenCalledWith(JSON.stringify(testMessage))
  })

  it('handles connection errors', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8081'))

    // Simulate connection error
    act(() => {
      const errorHandler = mockWebSocket.addEventListener.mock.calls.find(
        call => call[0] === 'error'
      )?.[1]
      if (errorHandler) errorHandler(new Error('Connection failed'))
    })

    expect(result.current.error).toBeDefined()
  })

  it('reconnects on connection loss', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8081'))

    // Simulate connection close
    act(() => {
      const closeHandler = mockWebSocket.addEventListener.mock.calls.find(
        call => call[0] === 'close'
      )?.[1]
      if (closeHandler) closeHandler()
    })

    // Wait for reconnection attempt
    act(() => {
      vi.advanceTimersByTime(2000)
    })

    // Should attempt to reconnect
    expect(global.WebSocket).toHaveBeenCalledTimes(2)
  })

  it('cleans up on unmount', () => {
    const { unmount } = renderHook(() => useWebSocket('ws://localhost:8081'))

    unmount()

    expect(mockWebSocket.close).toHaveBeenCalled()
  })

  it('handles invalid JSON messages gracefully', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8081'))

    // Simulate invalid JSON message
    act(() => {
      const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
        call => call[0] === 'message'
      )?.[1]
      if (messageHandler) {
        messageHandler({ data: 'invalid json' })
      }
    })

    // Should not crash and should handle gracefully
    expect(result.current.lastMessage).toBeNull()
  })

  it('tracks connection state correctly', () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8081'))

    // Initially not connected
    expect(result.current.isConnected).toBe(false)

    // Simulate connection open
    act(() => {
      const openHandler = mockWebSocket.addEventListener.mock.calls.find(
        call => call[0] === 'open'
      )?.[1]
      if (openHandler) openHandler()
    })

    expect(result.current.isConnected).toBe(true)

    // Simulate connection close
    act(() => {
      const closeHandler = mockWebSocket.addEventListener.mock.calls.find(
        call => call[0] === 'close'
      )?.[1]
      if (closeHandler) closeHandler()
    })

    expect(result.current.isConnected).toBe(false)
  })
})
