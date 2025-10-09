import { fireEvent, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import SignaturesPage from '@/pages/SignaturesPage'
import { render } from '../utils/test-utils'

const getSignatures = vi.fn()
const getSignatureDetail = vi.fn()
const getSignatureSchema = vi.fn()
const getVerifiers = vi.fn()
const getSignatureAnalytics = vi.fn()
const updateSignature = vi.fn()
const optimizeSignature = vi.fn()
const createSignature = vi.fn()
const deleteSignature = vi.fn()
const guardSystem = vi.fn().mockResolvedValue({ ok: true })

vi.mock('@/lib/registerCharts', () => ({ ensureChartsRegistered: vi.fn() }))
vi.mock('@/api/client', () => ({
  api: {
    getSignatures,
    getSignatureDetail,
    getSignatureSchema,
    getVerifiers,
    getSignatureAnalytics,
    updateSignature,
    optimizeSignature,
    createSignature,
    deleteSignature,
    guardSystem,
  }
}))

describe('SignaturesPage', () => {
  const signatures = [{
    name: 'CodeContextSig',
    type: 'analysis',
    performance: 90.1,
    success_rate: 95.2,
    avg_response_time: 1.23,
    iterations: 12,
    active: true,
  }]
  const detail = {
    metrics: {
      name: 'CodeContextSig',
      performance: 90.1,
      success_rate: 95.2,
      avg_response_time: 1.23,
      iterations: 12,
      type: 'analysis',
      active: true,
      last_updated: Date.now(),
    },
    trend: [],
  }
  const schema = { name: 'CodeContextSig', inputs: [], outputs: [] }
  const analytics = {
    reward_summary: { avg: 0.9, min: 0.1, max: 1, count: 5, hist: { bins: [], counts: [] } },
    related_verifiers: [],
    actions_sample: [],
  }

  beforeEach(() => {
    vi.clearAllMocks()
    getSignatures.mockResolvedValue({ signatures })
    getSignatureDetail.mockResolvedValue(detail)
    getSignatureSchema.mockResolvedValue(schema)
    getVerifiers.mockResolvedValue({ verifiers: [{ name: 'VerifierX' }] })
    getSignatureAnalytics.mockResolvedValue(analytics)
  })

  it('renders signatures and selects a signature', async () => {
    render(<SignaturesPage />)

    await waitFor(() => expect(getSignatures).toHaveBeenCalled())
    expect(await screen.findByText('CodeContextSig')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /View/i }))

    expect(await screen.findByText(/Details: CodeContextSig/)).toBeInTheDocument()
    expect(screen.getByText('90.10')).toBeInTheDocument()
  })

  it('creates a new signature', async () => {
    createSignature.mockResolvedValue({ success: true })
    render(<SignaturesPage />)

    await screen.findByText('CodeContextSig')
    fireEvent.change(screen.getByPlaceholderText(/Signature name/i), { target: { value: 'NewSig' } })
    fireEvent.submit(screen.getByPlaceholderText(/Signature name/i).closest('form') as HTMLFormElement)

    await waitFor(() => expect(createSignature).toHaveBeenCalled())
    expect(createSignature).toHaveBeenCalledWith(expect.objectContaining({ name: 'NewSig' }))
  })
})
