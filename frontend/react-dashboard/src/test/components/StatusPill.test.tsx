import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import StatusPill from '@/components/StatusPill'

describe('StatusPill', () => {
  it('normalizes positive statuses to online', () => {
    render(<StatusPill status="Running" />)

    const pill = screen.getByText(/Running/i)
    expect(pill).toBeInTheDocument()
    expect(pill.parentElement).toHaveClass('status-pill', 'status-online')
  })

  it('renders explicit text when provided', () => {
    render(<StatusPill status="healthy" text="All systems" />)

    expect(screen.getByText('All systems')).toBeInTheDocument()
  })

  it('maps warning-like statuses', () => {
    render(<StatusPill status="degraded" />)

    const pill = screen.getByText(/degraded/i)
    expect(pill.parentElement).toHaveClass('status-pill', 'status-warning')
  })

  it('maps failure statuses to error', () => {
    render(<StatusPill status="Failed" />)

    const pill = screen.getByText(/Failed/i)
    expect(pill.parentElement).toHaveClass('status-pill', 'status-error')
  })

  it('supports processing state and size overrides', () => {
    render(<StatusPill status="processing" size="lg" />)

    const pill = screen.getByText(/processing/i)
    expect(pill.parentElement).toHaveClass('status-pill', 'status-processing')
    expect(pill.parentElement).toHaveClass('px-4')
  })

  it('falls back to unknown when status absent', () => {
    render(<StatusPill status={undefined} showIcon={false} />)

    expect(screen.getByText('Unknown')).toBeInTheDocument()
  })
})
