import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import EventsTable from '../../components/EventsTable'

describe('EventsTable', () => {
  beforeAll(() => {
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined)
      }
    })
  })

  it('renders rows and copies', async () => {
    const rows = [
      { topic: 'spark.app', ts: 1, event: { event: 'submitted', name: 'job1' } },
      { topic: 'ui.action', ts: 2, event: { action: 'click', name: 'btn' } },
    ]
    render(<EventsTable rows={rows} /> as any)
    expect(screen.getByText('spark.app')).toBeTruthy()
    fireEvent.click(screen.getByText('Select All'))
    fireEvent.click(screen.getByText('Copy JSON'))
    expect(navigator.clipboard.writeText).toHaveBeenCalled()
  })
})

