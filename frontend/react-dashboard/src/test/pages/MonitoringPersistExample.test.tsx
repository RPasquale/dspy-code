import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import MonitoringPage from '@/pages/MonitoringPage'

global.fetch = vi.fn()

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

describe('Monitoring Persist Example', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Default mock for record-result
    // @ts-ignore
    global.fetch.mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/api/action/record-result')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ success: true }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
    })
    // Mock prompts
    vi.spyOn(window, 'prompt').mockImplementation((msg: string) => {
      if (msg.toLowerCase().includes('signature')) return 'CodeContextSig'
      if (msg.toLowerCase().includes('reward')) return '1.0'
      if (msg.toLowerCase().includes('environment')) return 'development'
      return ''
    })
    vi.spyOn(window, 'alert').mockImplementation(() => {})
  })

  it('persists kNN example via record-result', async () => {
    render(<MonitoringPage />, { wrapper: createWrapper() })
    // Fill doc_id input inside kNN form
    const docIdInput = await waitFor(() => screen.getByPlaceholderText('doc_id'))
    fireEvent.change(docIdInput, { target: { value: 'abc123' } })
    const persistBtn = screen.getByText('Persist as example')
    fireEvent.click(persistBtn)
    await waitFor(() => {
      expect(fetch).toHaveBeenCalled()
      const calls = (fetch as any).mock.calls
      const last = calls[calls.length - 1]
      expect(String(last[0])).toContain('/api/action/record-result')
      const body = JSON.parse(String((last[1] && last[1].body) || '{}'))
      expect(body.doc_id).toBe('abc123')
      expect(body.signature_name).toBe('CodeContextSig')
    })
  })
})

