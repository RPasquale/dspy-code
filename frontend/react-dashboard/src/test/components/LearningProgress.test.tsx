import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import LearningProgress from '@/components/LearningProgress'
import { LearningMetrics } from '@/hooks/useAgentMonitoring'

vi.mock('react-chartjs-2', () => ({
  Line: () => <div data-testid="learning-chart" />
}))

describe('LearningProgress', () => {
  const metrics: LearningMetrics = {
    training_sessions: 8,
    learning_progress: {},
    learning_trends: {
      training_accuracy: { current: 0.82, trend: 'improving' },
      validation_accuracy: { current: 0.74, trend: 'stable' },
      loss: { current: 0.1234, trend: 'declining' }
    },
    active_signatures: 3,
    signature_performance: {
      CodeExplain: {
        active: true,
        performance_score: 92.3,
        success_rate: 88.5,
        avg_response_time: 1.21
      },
      TestBench: {
        active: false,
        performance_score: 70.1,
        success_rate: 64.2,
        avg_response_time: 2.54
      }
    },
    retrieval_statistics: {
      total_events: 120,
      avg_score: 0.731,
      avg_hits_per_query: 4.5,
      unique_queries: 28
    }
  }

  it('renders advanced learning metrics', () => {
    render(<LearningProgress metrics={metrics} />)

    expect(screen.getByText('Learning Progress')).toBeInTheDocument()
    expect(screen.getByText('8 sessions')).toBeInTheDocument()
    expect(screen.getByText('82.0%')).toBeInTheDocument()
    expect(screen.getByText('74.0%')).toBeInTheDocument()
    expect(screen.getByText('0.1234')).toBeInTheDocument()
    expect(screen.getByText('Active Signatures')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('renders signature performance cards', () => {
    render(<LearningProgress metrics={metrics} />)

    expect(screen.getByText('CodeExplain')).toBeInTheDocument()
    expect(screen.getByText('92.3%')).toBeInTheDocument()
    expect(screen.getByText('Active')).toBeInTheDocument()
    expect(screen.getByText('88.5%')).toBeInTheDocument()
    expect(screen.getByText('1.21s')).toBeInTheDocument()
  })

  it('renders retrieval statistics', () => {
    render(<LearningProgress metrics={metrics} />)

    expect(screen.getByText('Retrieval Statistics')).toBeInTheDocument()
    expect(screen.getByText('120')).toBeInTheDocument()
    expect(screen.getByText('0.731')).toBeInTheDocument()
    expect(screen.getByText('4.5')).toBeInTheDocument()
    expect(screen.getByText('28')).toBeInTheDocument()
  })

  it('renders fallback when metrics missing', () => {
    render(<LearningProgress metrics={{ ...metrics, training_sessions: 0 }} />)

    expect(screen.getByText('No learning metrics available')).toBeInTheDocument()
  })
})
