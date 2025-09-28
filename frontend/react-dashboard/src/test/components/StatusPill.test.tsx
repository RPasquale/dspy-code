import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import StatusPill from '@/components/StatusPill'

// Mock the CSS module
vi.mock('@/components/StatusPill.module.css', () => ({
  pill: 'pill',
  success: 'success',
  warning: 'warning',
  error: 'error',
  processing: 'processing'
}))

describe('StatusPill', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders with success status', () => {
    render(<StatusPill status="success" />)
    
    const pill = screen.getByText('Success')
    expect(pill).toBeInTheDocument()
    expect(pill).toHaveClass('pill', 'success')
  })

  it('renders with warning status', () => {
    render(<StatusPill status="warning" />)
    
    const pill = screen.getByText('Warning')
    expect(pill).toBeInTheDocument()
    expect(pill).toHaveClass('pill', 'warning')
  })

  it('renders with error status', () => {
    render(<StatusPill status="error" />)
    
    const pill = screen.getByText('Error')
    expect(pill).toBeInTheDocument()
    expect(pill).toHaveClass('pill', 'error')
  })

  it('renders with processing status', () => {
    render(<StatusPill status="processing" />)
    
    const pill = screen.getByText('Processing')
    expect(pill).toBeInTheDocument()
    expect(pill).toHaveClass('pill', 'processing')
  })

  it('renders with custom text', () => {
    render(<StatusPill status="success" text="Custom Status" />)
    
    const pill = screen.getByText('Custom Status')
    expect(pill).toBeInTheDocument()
    expect(pill).toHaveClass('pill', 'success')
  })

  it('handles unknown status gracefully', () => {
    render(<StatusPill status="unknown" as any />)
    
    const pill = screen.getByText('Unknown')
    expect(pill).toBeInTheDocument()
    expect(pill).toHaveClass('pill')
  })

  it('applies custom className', () => {
    render(<StatusPill status="success" className="custom-class" />)
    
    const pill = screen.getByText('Success')
    expect(pill).toHaveClass('pill', 'success', 'custom-class')
  })

  it('renders with different status values', () => {
    const statuses = ['success', 'warning', 'error', 'processing'] as const
    
    statuses.forEach(status => {
      const { unmount } = render(<StatusPill status={status} />)
      
      const pill = screen.getByText(status.charAt(0).toUpperCase() + status.slice(1))
      expect(pill).toBeInTheDocument()
      expect(pill).toHaveClass('pill', status)
      
      unmount()
    })
  })

  it('handles empty or undefined text', () => {
    render(<StatusPill status="success" text="" />)
    
    const pill = screen.getByText('Success')
    expect(pill).toBeInTheDocument()
  })
})
