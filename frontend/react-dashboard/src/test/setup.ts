import '@testing-library/jest-dom'
import { vi } from 'vitest'
import React from 'react'

// Mock window.matchMedia for responsive design
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock ResizeObserver for chart components
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock IntersectionObserver for lazy loading
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

vi.mock('react-chartjs-2', () => {
  const makeMock = (chartName: string) => (props: Record<string, unknown>) => {
    const testId = typeof props['data-testid'] === 'string' ? props['data-testid'] : `chart-${chartName}`
    return React.createElement('div', { 'data-testid': testId })
  }

  return {
    Line: makeMock('line'),
    Bar: makeMock('bar'),
    Doughnut: makeMock('doughnut'),
    Scatter: makeMock('scatter'),
    Radar: makeMock('radar'),
    PolarArea: makeMock('polarArea')
  }
})
