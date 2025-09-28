import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import StreamsExplorerPage from '../StreamsExplorerPage'

describe('StreamsExplorerPage', () => {
  beforeEach(() => {
    // Mock fetch to avoid network in tests
    vi.spyOn(global, 'fetch' as any).mockResolvedValue({ ok: true, json: async () => ({ topic: 'x', items: [] }) } as any)
  })

  it('renders heading', () => {
    render(<BrowserRouter><StreamsExplorerPage /></BrowserRouter>)
    expect(screen.getByText('Streams Explorer')).toBeTruthy()
  })
})

