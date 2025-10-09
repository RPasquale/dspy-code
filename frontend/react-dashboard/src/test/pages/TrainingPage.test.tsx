import { fireEvent, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import TrainingPage from '@/pages/TrainingPage'
import { render } from '../utils/test-utils'

const sendCommand = vi.fn().mockResolvedValue({ output: 'ok' })
const getSystemResources = vi.fn().mockResolvedValue({ host: { ok: true } })

vi.mock('@/components/GrpoControls', () => ({
  __esModule: true,
  default: () => <div data-testid="grpo-controls" />,
}))

vi.mock('@/api/client', () => ({
  api: {
    sendCommand,
    getSystemResources,
  },
}))

describe('TrainingPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders training configuration form', async () => {
    render(<TrainingPage />)

    expect(screen.getByText('Training Setup')).toBeInTheDocument()
    expect(await screen.findByDisplayValue('/app/test_project')).toBeInTheDocument()
    expect(screen.getByLabelText(/Steps/i)).toBeInTheDocument()
    expect(screen.getByTestId('grpo-controls')).toBeInTheDocument()
  })

  it('updates training parameters', async () => {
    render(<TrainingPage />)

    const stepsInput = await screen.findByDisplayValue('500')
    fireEvent.change(stepsInput, { target: { value: '900' } })
    expect(stepsInput).toHaveValue(900)
  })

  it('starts training with sendCommand payload', async () => {
    render(<TrainingPage />)

    const startButton = await screen.findByRole('button', { name: /Start Training/i })
    fireEvent.click(startButton)

    await waitFor(() => expect(sendCommand).toHaveBeenCalled())
    expect(sendCommand.mock.calls[0][0]).toContain('rl train')
  })
})
