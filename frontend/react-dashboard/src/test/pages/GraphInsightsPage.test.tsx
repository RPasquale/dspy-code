import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import GraphInsightsPage from '@/pages/GraphInsightsPage'

const snapshotsResponse = {
  namespace: 'default',
  snapshots: [
    { timestamp: 1000, nodes: { 'src/a.py': {} }, edges: ['src/a.py->src/b.py'] },
    { timestamp: 2000, nodes: { 'src/b.py': {} }, edges: [] }
  ]
}

const diffResponse = {
  namespace: 'default',
  a: '1000',
  b: '2000',
  nodes_added: ['src/c.py'],
  nodes_removed: [],
  edges_added: ['src/a.py->src/c.py'],
  edges_removed: []
}

const mctsResponse = {
  namespace: 'default',
  nodes: [{ id: 'src/a.py', label: 'code_file', priority: 1.23, language: 'python', owner: 'alice' }]
}

const patternsResponse = {
  namespace: 'default',
  mode: 'cycles',
  cycles: [['src/a.py', 'src/b.py', 'src/a.py']]
}

const memoryReportResponse = {
  namespace: 'default',
  query: 'graph memory review',
  fusion_score: 0.82,
  metrics: { graph_signal: 0.6 },
  reward: 1.23,
  reward_breakdown: { graph_signal: 0.6, memory_precision: 0.4 },
  summary: 'Focus on orchestrator hot spots.',
  recommended_paths: ['dspy_agent/agents/orchestrator_runtime.py'],
  verifier_targets: ['graph_mcts_alignment', 'memory_precision'],
  followups: ['Run mcts refresh'],
  rationale: 'alignment check',
  rag: { mode: 'multi', answer: 'ok', context: {} },
  context: {},
  generated_at: 1700,
  top_k: 12
}

const createWrapper = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }}})
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  )
}

describe('GraphInsightsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    global.fetch = vi.fn((url: string) => {
      if (url.includes('/api/graph/snapshots')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(snapshotsResponse) }) as any
      }
      if (url.includes('/api/graph/diff')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(diffResponse) }) as any
      }
      if (url.includes('/api/graph/mcts-top')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mctsResponse) }) as any
      }
      if (url.includes('/api/graph/patterns')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(patternsResponse) }) as any
      }
      if (url.includes('/api/graph/memory-report')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(memoryReportResponse) }) as any
      }
      return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) }) as any
    })
  })

  it('renders snapshot table and diff controls', async () => {
    render(<GraphInsightsPage />, { wrapper: createWrapper() })
    expect(screen.getByText('Graph Insights')).toBeInTheDocument()
    expect(screen.getByText('Snapshots')).toBeInTheDocument()
    expect(screen.getByText('Snapshot Diff')).toBeInTheDocument()
    expect(screen.getByText('Top MCTS Priorities')).toBeInTheDocument()
    expect(screen.getByText('Graph Memory Summary')).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText('src/c.py')).toBeInTheDocument()
      expect(screen.getAllByText('1.230').length).toBeGreaterThan(0)
      expect(screen.getByText('Focus on orchestrator hot spots.')).toBeInTheDocument()
      expect(screen.getByText('dspy_agent/agents/orchestrator_runtime.py')).toBeInTheDocument()
    })
  })
})
