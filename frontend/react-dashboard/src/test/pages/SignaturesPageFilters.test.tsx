import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import SignaturesPage from '@/pages/SignaturesPage'

// Mock fetch for signatures APIs
global.fetch = vi.fn()

const sigList = { signatures: [ { name: 'CodeContextSig', type: 'analysis', performance: 90, success_rate: 95, avg_response_time: 1.5, iterations: 10, active: true } ] }
const sigDetail = { metrics: { name: 'CodeContextSig', performance: 90, success_rate: 95, avg_response_time: 1.5, iterations: 10, type: 'analysis', active: true }, trend: [] }
const sigSchema = { name: 'CodeContextSig', inputs: [ { name: 'snapshot' } ], outputs: [ { name: 'summary' } ] }
  const sigAnalytics = { signature: 'CodeContextSig', related_verifiers: [ { name: 'VerifierX', avg_score: 0.9, count: 5 } ], reward_summary: { avg: 0.8, min: 0.1, max: 1.0, count: 10, hist: { bins: [0,1], counts: [5,5] } }, context_keywords: {}, actions_sample: [] }

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

describe('SignaturesPage filters', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // @ts-ignore
    global.fetch.mockImplementation((url: string) => {
      if (url.includes('/api/signatures')) return Promise.resolve({ ok: true, json: () => Promise.resolve(sigList) })
      if (url.includes('/api/signature-detail')) return Promise.resolve({ ok: true, json: () => Promise.resolve(sigDetail) })
      if (url.includes('/api/signature-schema')) return Promise.resolve({ ok: true, json: () => Promise.resolve(sigSchema) })
      if (url.includes('/api/signature-analytics')) return Promise.resolve({ ok: true, json: () => Promise.resolve(sigAnalytics) })
      return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) })
    })
  })

  it('renders and updates filters', async () => {
    render(<SignaturesPage />, { wrapper: createWrapper() })
    await waitFor(() => {
      expect(screen.getByText('Signatures')).toBeInTheDocument()
    })
    // Timeframe & Env filters
    const selects = screen.getAllByRole('combobox')
    expect(selects.length).toBeGreaterThan(0)
    // Change timeframe
    fireEvent.change(selects[0], { target: { value: '7d' } })
    // Change env
    fireEvent.change(selects[1], { target: { value: 'production' } })
    // Verifier dropdown should include VerifierX option
    const verSelect = selects.find(s => (s as HTMLSelectElement).options && Array.from((s as HTMLSelectElement).options).some(o => o.textContent === 'VerifierX'))
    if (verSelect) {
      fireEvent.change(verSelect, { target: { value: 'VerifierX' } })
      await waitFor(() => {
        // After selecting verifier, the dropdown still shows the chosen value
        expect((verSelect as HTMLSelectElement).value).toBe('VerifierX')
      })
    }
    // Expect a verifier chip to appear in the UI
    await waitFor(() => {
      expect(screen.getByTestId('verifier-chip-VerifierX')).toBeInTheDocument()
    })
  })
})
