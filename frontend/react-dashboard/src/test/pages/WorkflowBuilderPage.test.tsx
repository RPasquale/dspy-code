import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import WorkflowBuilderPage from '@/pages/WorkflowBuilderPage'

const createWrapper = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  )
}

describe('WorkflowBuilderPage', () => {
  beforeEach(() => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ items: [] })
      }) as any
    )
  })

  it('enables save once a node and edge are configured', async () => {
    render(<WorkflowBuilderPage />, { wrapper: createWrapper() })

    expect(screen.getByText('Workflow Composer')).toBeInTheDocument()
    const addSignature = screen.getByRole('button', { name: '+ Signature' })
    fireEvent.click(addSignature)

    const addEdge = screen.getByRole('button', { name: '+ Add Edge' })
    fireEvent.click(addEdge)

    expect(screen.getAllByText('Signature').length).toBeGreaterThan(0)

    const saveButton = screen.getByRole('button', { name: /create workflow/i })
    expect(saveButton).not.toBeDisabled()
  })
})
