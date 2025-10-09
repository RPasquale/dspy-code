import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import SignatureGraphPage from '@/pages/SignatureGraphPage'

// Mock fetch globally
global.fetch = vi.fn()

const graphPayload = {
  nodes: [ { id: 'SigA', type: 'signature' }, { id: 'VerifierX', type: 'verifier' } ],
  edges: [ { source: 'SigA', target: 'VerifierX', avg: 0.9, count: 5 } ],
  edges_sig_sig: [ { a: 'SigA', b: 'SigB', count: 2 } ]
}

const snapshotPayload = {
  namespace: 'default',
  snapshots: [
    { timestamp: 1000, nodes: {}, edges: [] },
    { timestamp: 2000, nodes: {}, edges: [] }
  ]
}

const diffPayload = {
  namespace: 'default',
  a: '1000',
  b: '2000',
  nodes_added: ['src/new.py'],
  nodes_removed: [],
  edges_added: ['src/main.py->src/new.py'],
  edges_removed: []
}

const mctsPayload = {
  namespace: 'default',
  nodes: [
    { id: 'src/main.py', label: 'code_file', priority: 1.23, language: 'python', owner: 'alice' }
  ]
}

const createWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }}})
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  )
}

describe('SignatureGraphPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // @ts-ignore
    global.fetch.mockImplementation((url: string) => {
      if (url.includes('/api/signature/graph')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(graphPayload) })
      }
      if (url.includes('/api/graph/snapshots')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(snapshotPayload) })
      }
      if (url.includes('/api/graph/diff')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(diffPayload) })
      }
      if (url.includes('/api/graph/mcts-top')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mctsPayload) })
      }
      if (url.includes('/api/graph/patterns')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ namespace: 'default', mode: 'mixed-language', nodes: ['src/main.py'] }) })
      }
      return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) })
    })
  })

  it('renders controls and loads graph', async () => {
    render(<SignatureGraphPage />, { wrapper: createWrapper() })
    expect(screen.getByText('Signature Graph')).toBeInTheDocument()
    expect(screen.getByText('Graph Snapshot Diff')).toBeInTheDocument()
    expect(screen.getByText('Top MCTS Priorities')).toBeInTheDocument()
    expect(screen.getByText('Timeframe')).toBeInTheDocument()
    expect(screen.getByText('Env')).toBeInTheDocument()
    expect(screen.getByText('Export JSON')).toBeInTheDocument()
    // Refresh triggers fetch
    fireEvent.click(screen.getByText('Refresh'))
    await waitFor(() => {
      // presence of the SVG canvas is enough
      const svgs = document.getElementsByTagName('svg')
      expect(svgs.length).toBeGreaterThan(0)
    })
  })
})
