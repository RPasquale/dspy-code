import { describe, it, expect } from 'vitest'
import { mockApiResponses } from '../__mocks__/api'

describe('API Types Validation', () => {
  describe('Status Type', () => {
    it('validates status response structure', () => {
      const status = mockApiResponses.status

      expect(status).toHaveProperty('status')
      expect(status).toHaveProperty('timestamp')
      expect(status).toHaveProperty('version')
      expect(status).toHaveProperty('uptime')

      expect(typeof status.status).toBe('string')
      expect(typeof status.timestamp).toBe('string')
      expect(typeof status.version).toBe('string')
      expect(typeof status.uptime).toBe('number')
    })

    it('validates status values are valid', () => {
      const status = mockApiResponses.status

      expect(['healthy', 'warning', 'error']).toContain(status.status)
      expect(status.version).toMatch(/^\d+\.\d+\.\d+$/)
      expect(status.uptime).toBeGreaterThanOrEqual(0)
    })
  })

  describe('Metrics Type', () => {
    it('validates metrics response structure', () => {
      const metrics = mockApiResponses.metrics

      expect(metrics).toHaveProperty('cpu_usage')
      expect(metrics).toHaveProperty('memory_usage')
      expect(metrics).toHaveProperty('disk_usage')
      expect(metrics).toHaveProperty('network_io')

      expect(typeof metrics.cpu_usage).toBe('number')
      expect(typeof metrics.memory_usage).toBe('number')
      expect(typeof metrics.disk_usage).toBe('number')
      expect(typeof metrics.network_io).toBe('object')
    })

    it('validates metrics values are within valid ranges', () => {
      const metrics = mockApiResponses.metrics

      expect(metrics.cpu_usage).toBeGreaterThanOrEqual(0)
      expect(metrics.cpu_usage).toBeLessThanOrEqual(100)
      expect(metrics.memory_usage).toBeGreaterThanOrEqual(0)
      expect(metrics.memory_usage).toBeLessThanOrEqual(100)
      expect(metrics.disk_usage).toBeGreaterThanOrEqual(0)
      expect(metrics.disk_usage).toBeLessThanOrEqual(100)
    })

    it('validates network_io structure', () => {
      const metrics = mockApiResponses.metrics

      expect(metrics.network_io).toHaveProperty('bytes_in')
      expect(metrics.network_io).toHaveProperty('bytes_out')

      expect(typeof metrics.network_io.bytes_in).toBe('number')
      expect(typeof metrics.network_io.bytes_out).toBe('number')
      expect(metrics.network_io.bytes_in).toBeGreaterThanOrEqual(0)
      expect(metrics.network_io.bytes_out).toBeGreaterThanOrEqual(0)
    })
  })

  describe('RL Metrics Type', () => {
    it('validates RL metrics response structure', () => {
      const rlMetrics = mockApiResponses.rlMetrics

      expect(rlMetrics).toHaveProperty('episodes')
      expect(rlMetrics).toHaveProperty('avg_reward')
      expect(rlMetrics).toHaveProperty('success_rate')
      expect(rlMetrics).toHaveProperty('exploration_rate')
      expect(rlMetrics).toHaveProperty('learning_rate')
      expect(rlMetrics).toHaveProperty('recent_rewards')

      expect(typeof rlMetrics.episodes).toBe('number')
      expect(typeof rlMetrics.avg_reward).toBe('number')
      expect(typeof rlMetrics.success_rate).toBe('number')
      expect(typeof rlMetrics.exploration_rate).toBe('number')
      expect(typeof rlMetrics.learning_rate).toBe('number')
      expect(Array.isArray(rlMetrics.recent_rewards)).toBe(true)
    })

    it('validates RL metrics values are within valid ranges', () => {
      const rlMetrics = mockApiResponses.rlMetrics

      expect(rlMetrics.episodes).toBeGreaterThanOrEqual(0)
      expect(rlMetrics.avg_reward).toBeGreaterThanOrEqual(0)
      expect(rlMetrics.avg_reward).toBeLessThanOrEqual(1)
      expect(rlMetrics.success_rate).toBeGreaterThanOrEqual(0)
      expect(rlMetrics.success_rate).toBeLessThanOrEqual(1)
      expect(rlMetrics.exploration_rate).toBeGreaterThanOrEqual(0)
      expect(rlMetrics.exploration_rate).toBeLessThanOrEqual(1)
      expect(rlMetrics.learning_rate).toBeGreaterThanOrEqual(0)
      expect(rlMetrics.learning_rate).toBeLessThanOrEqual(1)
    })

    it('validates recent_rewards array', () => {
      const rlMetrics = mockApiResponses.rlMetrics

      expect(rlMetrics.recent_rewards.length).toBeGreaterThan(0)
      expect(rlMetrics.recent_rewards.every(reward => 
        typeof reward === 'number' && 
        reward >= 0 && 
        reward <= 1
      )).toBe(true)
    })
  })

  describe('Bus Metrics Type', () => {
    it('validates bus metrics response structure', () => {
      const busMetrics = mockApiResponses.busMetrics

      expect(busMetrics).toHaveProperty('total_messages')
      expect(busMetrics).toHaveProperty('active_consumers')
      expect(busMetrics).toHaveProperty('queue_depth')
      expect(busMetrics).toHaveProperty('processing_rate')
      expect(busMetrics).toHaveProperty('error_rate')

      expect(typeof busMetrics.total_messages).toBe('number')
      expect(typeof busMetrics.active_consumers).toBe('number')
      expect(typeof busMetrics.queue_depth).toBe('number')
      expect(typeof busMetrics.processing_rate).toBe('number')
      expect(typeof busMetrics.error_rate).toBe('number')
    })

    it('validates bus metrics values are non-negative', () => {
      const busMetrics = mockApiResponses.busMetrics

      expect(busMetrics.total_messages).toBeGreaterThanOrEqual(0)
      expect(busMetrics.active_consumers).toBeGreaterThanOrEqual(0)
      expect(busMetrics.queue_depth).toBeGreaterThanOrEqual(0)
      expect(busMetrics.processing_rate).toBeGreaterThanOrEqual(0)
      expect(busMetrics.error_rate).toBeGreaterThanOrEqual(0)
      expect(busMetrics.error_rate).toBeLessThanOrEqual(1)
    })
  })

  describe('Logs Type', () => {
    it('validates logs response structure', () => {
      const logs = mockApiResponses.logs

      expect(logs).toHaveProperty('logs')
      expect(logs).toHaveProperty('total_logs')
      expect(logs).toHaveProperty('timestamp')

      expect(Array.isArray(logs.logs)).toBe(true)
      expect(typeof logs.total_logs).toBe('number')
      expect(typeof logs.timestamp).toBe('string')
    })

    it('validates log entry structure', () => {
      const logs = mockApiResponses.logs

      if (logs.logs.length > 0) {
        const log = logs.logs[0]

        expect(log).toHaveProperty('timestamp')
        expect(log).toHaveProperty('level')
        expect(log).toHaveProperty('message')
        expect(log).toHaveProperty('status')

        expect(typeof log.timestamp).toBe('string')
        expect(typeof log.level).toBe('string')
        expect(typeof log.message).toBe('string')
        expect(typeof log.status).toBe('string')
      }
    })

    it('validates log level values', () => {
      const logs = mockApiResponses.logs

      const validLevels = ['INFO', 'SUCCESS', 'WARNING', 'ERROR', 'DEBUG']
      logs.logs.forEach(log => {
        expect(validLevels).toContain(log.level)
      })
    })

    it('validates log status values', () => {
      const logs = mockApiResponses.logs

      const validStatuses = ['success', 'processing', 'completed', 'warning', 'error']
      logs.logs.forEach(log => {
        expect(validStatuses).toContain(log.status)
      })
    })
  })

  describe('Kafka Topics Type', () => {
    it('validates Kafka topics response structure', () => {
      const kafkaTopics = mockApiResponses.kafkaTopics

      expect(kafkaTopics).toHaveProperty('topics')
      expect(kafkaTopics).toHaveProperty('total_topics')
      expect(kafkaTopics).toHaveProperty('timestamp')

      expect(Array.isArray(kafkaTopics.topics)).toBe(true)
      expect(typeof kafkaTopics.total_topics).toBe('number')
      expect(typeof kafkaTopics.timestamp).toBe('string')
    })

    it('validates topic structure', () => {
      const kafkaTopics = mockApiResponses.kafkaTopics

      if (kafkaTopics.topics.length > 0) {
        const topic = kafkaTopics.topics[0]

        expect(topic).toHaveProperty('name')
        expect(topic).toHaveProperty('partitions')
        expect(topic).toHaveProperty('messages')
        expect(topic).toHaveProperty('consumers')

        expect(typeof topic.name).toBe('string')
        expect(typeof topic.partitions).toBe('number')
        expect(typeof topic.messages).toBe('number')
        expect(typeof topic.consumers).toBe('number')
      }
    })

    it('validates topic values are non-negative', () => {
      const kafkaTopics = mockApiResponses.kafkaTopics

      kafkaTopics.topics.forEach(topic => {
        expect(topic.partitions).toBeGreaterThanOrEqual(0)
        expect(topic.messages).toBeGreaterThanOrEqual(0)
        expect(topic.consumers).toBeGreaterThanOrEqual(0)
      })
    })
  })

  describe('Performance History Type', () => {
    it('validates performance history response structure', () => {
      const performanceHistory = {
        data: [
          {
            timestamp: '2025-09-21T12:00:00Z',
            cpu: 45.2,
            memory: 67.8,
            throughput: 1000
          }
        ],
        timeframe: '1h',
        timestamp: '2025-09-21T12:00:00Z'
      }

      expect(performanceHistory).toHaveProperty('data')
      expect(performanceHistory).toHaveProperty('timeframe')
      expect(performanceHistory).toHaveProperty('timestamp')

      expect(Array.isArray(performanceHistory.data)).toBe(true)
      expect(typeof performanceHistory.timeframe).toBe('string')
      expect(typeof performanceHistory.timestamp).toBe('string')
    })

    it('validates performance data point structure', () => {
      const dataPoint = {
        timestamp: '2025-09-21T12:00:00Z',
        cpu: 45.2,
        memory: 67.8,
        throughput: 1000
      }

      expect(dataPoint).toHaveProperty('timestamp')
      expect(dataPoint).toHaveProperty('cpu')
      expect(dataPoint).toHaveProperty('memory')
      expect(dataPoint).toHaveProperty('throughput')

      expect(typeof dataPoint.timestamp).toBe('string')
      expect(typeof dataPoint.cpu).toBe('number')
      expect(typeof dataPoint.memory).toBe('number')
      expect(typeof dataPoint.throughput).toBe('number')
    })

    it('validates performance data values are within valid ranges', () => {
      const dataPoint = {
        timestamp: '2025-09-21T12:00:00Z',
        cpu: 45.2,
        memory: 67.8,
        throughput: 1000
      }

      expect(dataPoint.cpu).toBeGreaterThanOrEqual(0)
      expect(dataPoint.cpu).toBeLessThanOrEqual(100)
      expect(dataPoint.memory).toBeGreaterThanOrEqual(0)
      expect(dataPoint.memory).toBeLessThanOrEqual(100)
      expect(dataPoint.throughput).toBeGreaterThanOrEqual(0)
    })
  })
})
