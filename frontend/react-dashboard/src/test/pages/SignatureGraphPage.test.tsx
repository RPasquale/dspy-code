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
      return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) })
    })
  })

  it('renders controls and loads graph', async () => {
    render(<SignatureGraphPage />, { wrapper: createWrapper() })
    expect(screen.getByText('Signature Graph')).toBeInTheDocument()
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

