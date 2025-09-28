import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import AgentStatusCard from '@/components/AgentStatusCard'

// Mock the CSS module
vi.mock('@/components/AgentStatusCard.module.css', () => ({
  card: 'card',
  status: 'status',
  healthy: 'healthy',
  warning: 'warning',
  error: 'error'
}))

describe('AgentStatusCard', () => {
  const mockProps = {
    status: 'healthy',
    timestamp: '2025-09-21T12:00:00Z',
    version: '1.0.0',
    uptime: 3600
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders with healthy status', () => {
    render(<AgentStatusCard {...mockProps} />)
    
    expect(screen.getByText('Agent Status')).toBeInTheDocument()
    expect(screen.getByText('healthy')).toBeInTheDocument()
    expect(screen.getByText('Version: 1.0.0')).toBeInTheDocument()
    expect(screen.getByText('Uptime: 1h 0m')).toBeInTheDocument()
  })

  it('renders with warning status', () => {
    const warningProps = { ...mockProps, status: 'warning' }
    render(<AgentStatusCard {...warningProps} />)
    
    expect(screen.getByText('warning')).toBeInTheDocument()
  })

  it('renders with error status', () => {
    const errorProps = { ...mockProps, status: 'error' }
    render(<AgentStatusCard {...errorProps} />)
    
    expect(screen.getByText('error')).toBeInTheDocument()
  })

  it('formats uptime correctly for different durations', () => {
    const { rerender } = render(<AgentStatusCard {...mockProps} />)
    expect(screen.getByText('Uptime: 1h 0m')).toBeInTheDocument()

    // Test with different uptime values
    rerender(<AgentStatusCard {...mockProps} uptime={3661} />)
    expect(screen.getByText('Uptime: 1h 1m 1s')).toBeInTheDocument()

    rerender(<AgentStatusCard {...mockProps} uptime={86400} />)
    expect(screen.getByText('Uptime: 1d 0h 0m')).toBeInTheDocument()
  })

  it('displays timestamp in readable format', () => {
    render(<AgentStatusCard {...mockProps} />)
    
    // The component should display the timestamp
    expect(screen.getByText(/2025-09-21/)).toBeInTheDocument()
  })

  it('applies correct CSS classes based on status', () => {
    const { rerender } = render(<AgentStatusCard {...mockProps} />)
    
    // Test healthy status
    let statusElement = screen.getByText('healthy')
    expect(statusElement).toHaveClass('status', 'healthy')

    // Test warning status
    rerender(<AgentStatusCard {...mockProps} status="warning" />)
    statusElement = screen.getByText('warning')
    expect(statusElement).toHaveClass('status', 'warning')

    // Test error status
    rerender(<AgentStatusCard {...mockProps} status="error" />)
    statusElement = screen.getByText('error')
    expect(statusElement).toHaveClass('status', 'error')
  })

  it('handles missing optional props gracefully', () => {
    const minimalProps = { status: 'healthy' }
    render(<AgentStatusCard {...minimalProps} />)
    
    expect(screen.getByText('Agent Status')).toBeInTheDocument()
    expect(screen.getByText('healthy')).toBeInTheDocument()
  })
})
