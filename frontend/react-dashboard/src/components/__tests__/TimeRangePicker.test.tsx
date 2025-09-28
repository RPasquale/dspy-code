import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import TimeRangePicker from '../../components/TimeRangePicker'

describe('TimeRangePicker', () => {
  it('emits preset change', () => {
    const onChange = vi.fn()
    render(<TimeRangePicker onChange={onChange} /> as any)
    const btn = screen.getByText('Last 5m')
    fireEvent.click(btn)
    expect(onChange).toHaveBeenCalled()
    const arg = onChange.mock.calls[0][0]
    expect(arg.label).toBe('Last 5m')
    expect(typeof arg.since).toBe('number')
  })
})

