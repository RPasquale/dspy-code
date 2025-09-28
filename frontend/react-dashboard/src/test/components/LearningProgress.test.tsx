import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import LearningProgress from '@/components/LearningProgress'

// Mock the CSS module
vi.mock('@/components/LearningProgress.module.css', () => ({
  container: 'container',
  progressBar: 'progressBar',
  progressFill: 'progressFill',
  metrics: 'metrics',
  metric: 'metric',
  label: 'label',
  value: 'value'
}))

// Mock Chart.js components
vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(() => <div data-testid="learning-chart">Learning Chart</div>)
}))

describe('LearningProgress', () => {
  const mockProps = {
    episodes: 1000,
    avgReward: 0.75,
    successRate: 0.68,
    recentRewards: [0.8, 0.7, 0.9, 0.6, 0.85],
    explorationRate: 0.1,
    learningRate: 0.001
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders learning metrics correctly', () => {
    render(<LearningProgress {...mockProps} />)
    
    expect(screen.getByText('Learning Progress')).toBeInTheDocument()
    expect(screen.getByText('1000')).toBeInTheDocument() // episodes
    expect(screen.getByText('75%')).toBeInTheDocument() // avg reward as percentage
    expect(screen.getByText('68%')).toBeInTheDocument() // success rate
  })

  it('displays progress bar with correct percentage', () => {
    render(<LearningProgress {...mockProps} />)
    
    const progressBar = screen.getByRole('progressbar')
    expect(progressBar).toHaveAttribute('aria-valuenow', '75')
    expect(progressBar).toHaveAttribute('aria-valuemax', '100')
  })

  it('renders learning chart', () => {
    render(<LearningProgress {...mockProps} />)
    
    expect(screen.getByTestId('learning-chart')).toBeInTheDocument()
  })

  it('formats percentages correctly', () => {
    const { rerender } = render(<LearningProgress {...mockProps} />)
    
    // Test with different values
    rerender(<LearningProgress {...mockProps} avgReward={0.5} successRate={0.3} />)
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getByText('30%')).toBeInTheDocument()
  })

  it('handles edge cases for metrics', () => {
    const edgeCaseProps = {
      ...mockProps,
      episodes: 0,
      avgReward: 0,
      successRate: 0,
      recentRewards: []
    }
    
    render(<LearningProgress {...edgeCaseProps} />)
    
    expect(screen.getByText('0')).toBeInTheDocument() // episodes
    expect(screen.getByText('0%')).toBeInTheDocument() // avg reward
    expect(screen.getByText('0%')).toBeInTheDocument() // success rate
  })

  it('displays exploration and learning rates', () => {
    render(<LearningProgress {...mockProps} />)
    
    expect(screen.getByText('Exploration Rate')).toBeInTheDocument()
    expect(screen.getByText('Learning Rate')).toBeInTheDocument()
    expect(screen.getByText('10%')).toBeInTheDocument() // exploration rate
    expect(screen.getByText('0.1%')).toBeInTheDocument() // learning rate
  })

  it('handles missing recent rewards gracefully', () => {
    const propsWithoutRewards = {
      ...mockProps,
      recentRewards: undefined as any
    }
    
    render(<LearningProgress {...propsWithoutRewards} />)
    
    expect(screen.getByText('Learning Progress')).toBeInTheDocument()
    expect(screen.getByTestId('learning-chart')).toBeInTheDocument()
  })

  it('applies correct CSS classes', () => {
    render(<LearningProgress {...mockProps} />)
    
    const container = screen.getByText('Learning Progress').closest('div')
    expect(container).toHaveClass('container')
  })
})
