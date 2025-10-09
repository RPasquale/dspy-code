import { fireEvent, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import MonitoringPage from '@/pages/MonitoringPage'
import { render } from '../utils/test-utils'

const mockSendCommand = vi.fn().mockResolvedValue({ success: true, output: 'ok' })
const mockRestart = vi.fn().mockResolvedValue(undefined)

vi.mock('@/api/client', () => ({
  api: {
    sendCommand: () => mockSendCommand(),
    restartAgent: () => mockRestart(),
    getProposedActions: () => Promise.resolve([])
  }
}))

describe('MonitoringPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the runtime monitoring header and fallback status', () => {
    render(<MonitoringPage />)

    expect(screen.getByText('Runtime Monitoring')).toBeInTheDocument()
    expect(screen.getByText(/Stream logs/)).toBeInTheDocument()
    expect(screen.getByText('Service Health')).toBeInTheDocument()
    expect(screen.getByText('Waiting for status…')).toBeInTheDocument()
  })

  it('allows issuing restart commands', async () => {
    render(<MonitoringPage />)

    const restartButton = screen.getByRole('button', { name: /Restart Agent/i })
    fireEvent.click(restartButton)

    await waitFor(() => expect(mockRestart).toHaveBeenCalled())
  })

  it('sends shell commands from command palette', async () => {
    render(<MonitoringPage />)

    const input = screen.getByPlaceholderText('rl train --steps 200')
    fireEvent.change(input, { target: { value: 'tail logs' } })
    fireEvent.submit(input.closest('form') as HTMLFormElement)

    await waitFor(() => expect(mockSendCommand).toHaveBeenCalledWith('tail logs'))
  })
})
