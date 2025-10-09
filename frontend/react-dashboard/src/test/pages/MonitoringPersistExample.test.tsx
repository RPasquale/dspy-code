import { fireEvent, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import MonitoringPage from '@/pages/MonitoringPage'
import { render } from '../utils/test-utils'

const mockFetchImpl = vi.fn()

global.fetch = mockFetchImpl as unknown as typeof fetch

const createResponse = (data: unknown) => ({ ok: true, json: () => Promise.resolve(data) })

describe('MonitoringPage persistence helpers', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockFetchImpl.mockImplementation((url: string) => {
      if (url.includes('/api/action/record-result')) {
        return Promise.resolve(createResponse({ success: true }))
      }
      return Promise.resolve(createResponse({}))
    })
    vi.spyOn(window, 'prompt').mockImplementation((msg: string) => {
      if (msg.toLowerCase().includes('signature')) return 'CodeContextSig'
      if (msg.toLowerCase().includes('reward')) return '0.8'
      if (msg.toLowerCase().includes('environment')) return 'development'
      return ''
    })
    vi.spyOn(window, 'alert').mockImplementation(() => {})
  })

  it('submits a persist example request', async () => {
    render(<MonitoringPage />)

    const docIdInput = await waitFor(() => screen.getByPlaceholderText('doc_id'))
    fireEvent.change(docIdInput, { target: { value: 'abc123' } })
    fireEvent.click(screen.getByText('Persist as example'))

    await waitFor(() => {
      expect(mockFetchImpl).toHaveBeenCalledWith('/api/action/record-result', expect.any(Object))
      const last = mockFetchImpl.mock.calls.at(-1) as [string, RequestInit]
      const body = JSON.parse(String(last[1]?.body))
      expect(body.doc_id).toBe('abc123')
      expect(body.signature_name).toBe('CodeContextSig')
    })
  })
})
