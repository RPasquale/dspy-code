// Mock API responses for testing
export const mockApiResponses = {
  status: {
    status: 'healthy',
    timestamp: '2025-09-21T12:00:00Z',
    version: '1.0.0',
    uptime: 3600
  },
  metrics: {
    cpu_usage: 45.2,
    memory_usage: 67.8,
    disk_usage: 23.1,
    network_io: {
      bytes_in: 1024000,
      bytes_out: 512000
    }
  },
  rlMetrics: {
    episodes: 1000,
    avg_reward: 0.75,
    average_reward: 0.75,
    success_rate: 0.68,
    exploration_rate: 0.1,
    learning_rate: 0.001,
    recent_rewards: [0.8, 0.7, 0.9, 0.6, 0.85],
    timestamp: '2025-09-21T12:00:00Z'
  },
  busMetrics: {
    total_messages: 15000,
    active_consumers: 5,
    queue_depth: 120,
    processing_rate: 45.2,
    error_rate: 0.02
  },
  logs: {
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
    timestamp: '2025-09-21T12:00:00Z'
  },
  kafkaTopics: {
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
    timestamp: '2025-09-21T12:00:00Z'
  }
}

// Mock fetch function
export const mockFetch = (url: string) => {
  const endpoint = url.split('/').pop()
  if (url.includes('/api/rl/sweep/state')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ exists: true, state: { method: 'eprotein', iteration: 3 }, pareto: [{ output: 1.2, cost: 2.5 }, { output: 1.1, cost: 1.8 }] }) })
  }
  if (url.includes('/api/rl/sweep/history')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ best: { summary: { metric: 1.3 } }, experiments: [{ ts: 1, method: 'eprotein', best_metric: 1.2 }] }) })
  }
  if (url.includes('/api/rl/sweep/run')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ started: true, method: 'eprotein', iterations: 4 }) })
  }
  if (url.includes('/api/profile')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ profile: 'balanced' }) })
  }
  if (url.includes('/api/reddb/summary')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ signatures: 12, recent_actions: 200, recent_training: 50, timestamp: Date.now()/1000 }) })
  }
  
  switch (endpoint) {
    case 'status':
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockApiResponses.status)
      })
    case 'metrics':
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockApiResponses.metrics)
      })
    case 'rl-metrics':
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockApiResponses.rlMetrics)
      })
    case 'bus-metrics':
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockApiResponses.busMetrics)
      })
    case 'logs':
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockApiResponses.logs)
      })
    case 'kafka-topics':
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockApiResponses.kafkaTopics)
      })
    default:
      return Promise.resolve({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ error: 'Not found' })
      })
  }
}
