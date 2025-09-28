import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mockApiResponses, mockFetch } from '../__mocks__/api'

// Mock fetch globally
global.fetch = vi.fn()

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // @ts-ignore
    global.fetch.mockImplementation(mockFetch)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Status API', () => {
    it('fetches agent status successfully', async () => {
      const response = await fetch('/api/status')
      const data = await response.json()

      expect(response.ok).toBe(true)
      expect(data).toHaveProperty('agent')
      expect(data).toHaveProperty('ollama')
      expect(data).toHaveProperty('kafka')
      expect(data).toHaveProperty('containers')
    })

    it('handles status API errors', async () => {
      // @ts-ignore
      global.fetch.mockImplementation(() => 
        Promise.resolve({
          ok: false,
          status: 500,
          json: () => Promise.resolve({ error: 'Server Error' })
        })
      )

      const response = await fetch('/api/status')
      const data = await response.json()

      expect(response.ok).toBe(false)
      expect(data.error).toBe('Server Error')
    })
  })

  describe('Metrics API', () => {
    it('fetches system metrics successfully', async () => {
      const response = await fetch('/api/metrics')
      const data = await response.json()

      expect(response.ok).toBe(true)
      expect(data).toHaveProperty('timestamp')
      expect(data).toHaveProperty('containers')
      expect(data).toHaveProperty('memory_usage')
      expect(data).toHaveProperty('response_time')
    })

    it('validates metrics data structure', async () => {
      const response = await fetch('/api/metrics')
      const data = await response.json()

      expect(data).toHaveProperty('timestamp')
      expect(data).toHaveProperty('containers')
      expect(data).toHaveProperty('memory_usage')
      expect(data).toHaveProperty('response_time')
      
      expect(typeof data.timestamp).toBe('number')
      expect(typeof data.containers).toBe('number')
      expect(typeof data.response_time).toBe('number')
    })
  })

  describe('RL Metrics API', () => {
    it('fetches RL metrics successfully', async () => {
      const response = await fetch('/api/rl-metrics')
      const data = await response.json()

      expect(response.ok).toBe(true)
      expect(data).toEqual(mockApiResponses.rlMetrics)
      expect(data.avg_reward).toBe(0.75)
      expect(data.episodes).toBe(1000)
    })

    it('validates RL metrics data structure', async () => {
      const response = await fetch('/api/rl-metrics')
      const data = await response.json()

      expect(data).toHaveProperty('episodes')
      expect(data).toHaveProperty('avg_reward')
      expect(data).toHaveProperty('success_rate')
      expect(data).toHaveProperty('recent_rewards')
      
      expect(Array.isArray(data.recent_rewards)).toBe(true)
      expect(data.recent_rewards.length).toBeGreaterThan(0)
    })

    it('handles missing avg_reward field', async () => {
      // Mock response without avg_reward
      // @ts-ignore
      global.fetch.mockImplementation((url) => {
        if (url.includes('rl-metrics')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              episodes: 1000,
              success_rate: 0.68,
              recent_rewards: [0.8, 0.7, 0.9]
            })
          })
        }
        return mockFetch(url)
      })

      const response = await fetch('/api/rl-metrics')
      const data = await response.json()

      expect(data.avg_reward).toBeUndefined()
    })
  })

  describe('Bus Metrics API', () => {
    it('fetches bus metrics successfully', async () => {
      const response = await fetch('/api/bus-metrics')
      const data = await response.json()

      expect(response.ok).toBe(true)
      expect(data).toEqual(mockApiResponses.busMetrics)
      expect(data.total_messages).toBe(15000)
      expect(data.active_consumers).toBe(5)
    })

    it('validates bus metrics data structure', async () => {
      const response = await fetch('/api/bus-metrics')
      const data = await response.json()

      expect(data).toHaveProperty('total_messages')
      expect(data).toHaveProperty('active_consumers')
      expect(data).toHaveProperty('queue_depth')
      expect(data).toHaveProperty('processing_rate')
      expect(data).toHaveProperty('error_rate')
    })
  })

  describe('Logs API', () => {
    it('fetches logs successfully', async () => {
      const response = await fetch('/api/logs')
      const data = await response.json()

      expect(response.ok).toBe(true)
      expect(data).toEqual(mockApiResponses.logs)
      expect(data.logs).toHaveLength(2)
    })

    it('validates logs data structure', async () => {
      const response = await fetch('/api/logs')
      const data = await response.json()

      expect(data).toHaveProperty('logs')
      expect(data).toHaveProperty('total_logs')
      expect(Array.isArray(data.logs)).toBe(true)
      
      if (data.logs.length > 0) {
        const log = data.logs[0]
        expect(log).toHaveProperty('timestamp')
        expect(log).toHaveProperty('level')
        expect(log).toHaveProperty('message')
        expect(log).toHaveProperty('status')
      }
    })

    it('handles empty logs array', async () => {
      // @ts-ignore
      global.fetch.mockImplementation((url) => {
        if (url.includes('logs')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              logs: [],
              total_logs: 0,
              timestamp: '2025-09-21T12:00:00Z'
            })
          })
        }
        return mockFetch(url)
      })

      const response = await fetch('/api/logs')
      const data = await response.json()

      expect(data.logs).toHaveLength(0)
      expect(data.total_logs).toBe(0)
    })
  })

  describe('Kafka Topics API', () => {
    it('fetches Kafka topics successfully', async () => {
      const response = await fetch('/api/kafka-topics')
      const data = await response.json()

      expect(response.ok).toBe(true)
      expect(data).toEqual(mockApiResponses.kafkaTopics)
      expect(data.topics).toHaveLength(2)
    })

    it('validates Kafka topics data structure', async () => {
      const response = await fetch('/api/kafka-topics')
      const data = await response.json()

      expect(data).toHaveProperty('topics')
      expect(data).toHaveProperty('total_topics')
      expect(Array.isArray(data.topics)).toBe(true)
      
      if (data.topics.length > 0) {
        const topic = data.topics[0]
        expect(topic).toHaveProperty('name')
        expect(topic).toHaveProperty('partitions')
        expect(topic).toHaveProperty('messages')
      }
    })
  })

  describe('Performance History API', () => {
    it('fetches performance history with timeframe', async () => {
      const response = await fetch('/api/performance-history?timeframe=1h')
      const data = await response.json()

      expect(response.ok).toBe(true)
      expect(data).toHaveProperty('data')
      expect(data).toHaveProperty('timeframe')
      expect(data.timeframe).toBe('1h')
    })

    it('handles different timeframes', async () => {
      const timeframes = ['1h', '24h', '7d']
      
      for (const timeframe of timeframes) {
        const response = await fetch(`/api/performance-history?timeframe=${timeframe}`)
        const data = await response.json()

        expect(response.ok).toBe(true)
        expect(data.timeframe).toBe(timeframe)
      }
    })
  })

  describe('RL Sweep API', () => {
    it('fetches sweep state with Pareto points', async () => {
      const resp = await fetch('/api/rl/sweep/state')
      const data = await resp.json()
      expect(resp.ok).toBe(true)
      expect(data.exists).toBe(true)
      expect(Array.isArray(data.pareto)).toBe(true)
    })

    it('fetches sweep history', async () => {
      const resp = await fetch('/api/rl/sweep/history')
      const data = await resp.json()
      expect(resp.ok).toBe(true)
      expect(Array.isArray(data.experiments)).toBe(true)
    })

    it('starts a sweep run', async () => {
      const resp = await fetch('/api/rl/sweep/run', { method: 'POST', body: JSON.stringify({ method: 'eprotein', iterations: 4 }) })
      const data = await resp.json()
      expect(resp.ok).toBe(true)
      expect(data.started).toBe(true)
    })
  })

  describe('Error Handling', () => {
    it('handles network errors', async () => {
      // @ts-ignore
      global.fetch.mockImplementation(() => 
        Promise.reject(new Error('Network error'))
      )

      try {
        await fetch('/api/status')
      } catch (error) {
        expect(error).toBeInstanceOf(Error)
        expect(error.message).toBe('Network error')
      }
    })

    it('handles 404 errors', async () => {
      // @ts-ignore
      global.fetch.mockImplementation(() => 
        Promise.resolve({
          ok: false,
          status: 404,
          json: () => Promise.resolve({ error: 'Not found' })
        })
      )

      const response = await fetch('/api/non-existent')
      const data = await response.json()

      expect(response.ok).toBe(false)
      expect(response.status).toBe(404)
      expect(data.error).toBe('Not found')
    })

    it('handles timeout errors', async () => {
      // @ts-ignore
      global.fetch.mockImplementation(() => 
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Timeout')), 100)
        )
      )

      try {
        await fetch('/api/status')
      } catch (error) {
        expect(error).toBeInstanceOf(Error)
        expect(error.message).toBe('Timeout')
      }
    })
  })

  describe('Profile & Retention APIs', () => {
    it('fetches profile', async () => {
      const resp = await fetch('/api/profile')
      const data = await resp.json()
      expect(resp.ok).toBe(true)
      expect(['fast','balanced','maxquality', undefined].includes(data.profile)).toBe(true)
    })

    it('fetches RedDB summary', async () => {
      const resp = await fetch('/api/reddb/summary')
      const data = await resp.json()
      expect(resp.ok).toBe(true)
      expect(typeof data.signatures).toBe('number')
      expect(typeof data.recent_actions).toBe('number')
      expect(typeof data.recent_training).toBe('number')
    })
  })

  describe('Data Validation', () => {
    it('validates numeric fields are numbers', async () => {
      const response = await fetch('/api/metrics')
      const data = await response.json()

      expect(typeof data.cpu_usage).toBe('number')
      expect(typeof data.memory_usage).toBe('number')
      expect(typeof data.disk_usage).toBe('number')
      
      expect(data.cpu_usage).toBeGreaterThanOrEqual(0)
      expect(data.cpu_usage).toBeLessThanOrEqual(100)
    })

    it('validates array fields are arrays', async () => {
      const response = await fetch('/api/rl-metrics')
      const data = await response.json()

      expect(Array.isArray(data.recent_rewards)).toBe(true)
      expect(data.recent_rewards.every((reward: any) => typeof reward === 'number')).toBe(true)
    })

    it('validates string fields are strings', async () => {
      const response = await fetch('/api/status')
      const data = await response.json()

      expect(typeof data.status).toBe('string')
      expect(typeof data.version).toBe('string')
      expect(typeof data.timestamp).toBe('string')
    })
  })
})
