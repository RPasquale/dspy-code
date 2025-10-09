import { act, renderHook } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useAgentMonitoring } from '@/hooks/useAgentMonitoring'

type Message = { type: string; data: any; timestamp?: number }

const mockState = {
  isConnected: true,
  error: null as string | null,
  lastMessage: null as Message | null,
  sendMessage: vi.fn()
}

vi.mock('@/hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    isConnected: mockState.isConnected,
    error: mockState.error,
    lastMessage: mockState.lastMessage,
    sendMessage: mockState.sendMessage,
    readyState: 1,
    reconnectAttempts: 0,
    isConnecting: false,
    url: 'ws://localhost',
    connect: vi.fn(),
    disconnect: vi.fn(),
    reconnect: vi.fn(),
    send: mockState.sendMessage,
    sendRaw: mockState.sendMessage
  })
}))

describe('useAgentMonitoring', () => {
  beforeEach(() => {
    mockState.lastMessage = null
    mockState.error = null
    mockState.sendMessage.mockReset()
  })

  it('returns default monitoring data', () => {
    const { result } = renderHook(() => useAgentMonitoring())

    expect(result.current.data.learningMetrics.training_sessions).toBe(0)
    expect(result.current.isConnected).toBe(true)
    expect(result.current.error).toBeNull()
  })

  it('updates learning metrics when websocket message arrives', () => {
    const { result, rerender } = renderHook(() => useAgentMonitoring())

    const learningMessage: Message = {
      type: 'learning_update',
      data: {
        training_sessions: 4,
        learning_trends: {
          training_accuracy: { current: 0.9, trend: 'improving' },
          validation_accuracy: { current: 0.8, trend: 'stable' },
          loss: { current: 0.12, trend: 'declining' }
        },
        signature_performance: {},
        retrieval_statistics: {},
        active_signatures: 2
      },
      timestamp: Date.now()
    }

    act(() => {
      mockState.lastMessage = learningMessage
      rerender()
    })

    expect(result.current.data.learningMetrics.training_sessions).toBe(4)
    expect(result.current.data.learningMetrics.learning_trends.training_accuracy.current).toBe(0.9)
  })

  it('updates action statistics from actions_update message', () => {
    const { result, rerender } = renderHook(() => useAgentMonitoring())

    act(() => {
      mockState.lastMessage = {
        type: 'actions_update',
        timestamp: 123,
        data: {
          actions: [{ action_id: '1', timestamp: 1, action_type: 'plan', reward: 0.5, confidence: 0.8, execution_time: 2, result_summary: 'ok', environment: 'dev' }],
          statistics: {
            total_actions: 1,
            avg_reward: 0.5,
            avg_confidence: 0.8,
            avg_execution_time: 2,
            high_reward_actions: 1,
            action_types: { plan: 1 }
          }
        }
      }
      rerender()
    })

    expect(result.current.data.actions.length).toBe(1)
    expect(result.current.data.actionStatistics.total_actions).toBe(1)
  })

  it('sends subscription requests via helper', () => {
    const { result } = renderHook(() => useAgentMonitoring())

    act(() => {
      result.current.subscribe(['learning_update'])
      result.current.requestData('status')
    })

    expect(mockState.sendMessage).toHaveBeenCalledWith({ type: 'subscribe', subscriptions: ['learning_update'] })
    expect(mockState.sendMessage).toHaveBeenCalledWith({ type: 'get_data', data_type: 'status' })
  })
})
