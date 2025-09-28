# DSPy Agent Frontend Testing Guide

This document provides comprehensive information about the React testing suite for the DSPy Agent frontend dashboard.

## 🧪 Test Suite Overview

The frontend test suite includes:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Page-level component interactions
- **API Tests**: Data fetching and error handling
- **Hook Tests**: Custom React hooks functionality
- **App Tests**: Main application routing and structure

## 📁 Test Structure

```
src/test/
├── components/           # Unit tests for React components
│   ├── AgentStatusCard.test.tsx
│   ├── LearningProgress.test.tsx
│   └── StatusPill.test.tsx
├── pages/               # Integration tests for pages
│   ├── OverviewPage.test.tsx
│   ├── MonitoringPage.test.tsx
│   └── TrainingPage.test.tsx
├── hooks/               # Tests for custom hooks
│   ├── useAgentMonitoring.test.ts
│   └── useWebSocket.test.ts
├── api/                 # API service tests
│   ├── client.test.ts
│   └── types.test.ts
├── __mocks__/           # Mock data and functions
│   └── api.ts
├── utils/               # Test utilities
│   └── test-utils.tsx
├── setup.ts             # Test setup configuration
└── run-tests.sh         # Test runner script
```

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
npm test

# Run tests with UI
npm run test:ui

# Run tests once (CI mode)
npm run test:run

# Run tests with coverage
npm run test:coverage
```

### Individual Test Suites

```bash
# Run only component tests
npm run test:run -- src/test/components/

# Run only page tests
npm run test:run -- src/test/pages/

# Run only API tests
npm run test:run -- src/test/api/

# Run only hook tests
npm run test:run -- src/test/hooks/
```

### Using the Test Runner Script

```bash
# Run the comprehensive test suite
./src/test/run-tests.sh
```

## 🔧 Test Configuration

### Vitest Configuration

The tests use Vitest with the following configuration:

```typescript
// vitest.config.ts
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html']
    }
  }
})
```

### Test Setup

```typescript
// src/test/setup.ts
import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock Chart.js, WebSocket, fetch, etc.
```

## 📝 Writing Tests

### Component Tests

```typescript
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import MyComponent from '@/components/MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Expected Text')).toBeInTheDocument()
  })
})
```

### Hook Tests

```typescript
import { renderHook, waitFor } from '@testing-library/react'
import { useMyHook } from '@/hooks/useMyHook'

describe('useMyHook', () => {
  it('returns expected data', async () => {
    const { result } = renderHook(() => useMyHook())
    
    await waitFor(() => {
      expect(result.current.data).toBeDefined()
    })
  })
})
```

### API Tests

```typescript
import { describe, it, expect, vi } from 'vitest'

describe('API Client', () => {
  it('fetches data successfully', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData)
    })
    
    const response = await fetch('/api/endpoint')
    const data = await response.json()
    
    expect(data).toEqual(mockData)
  })
})
```

## 🎯 Test Categories

### 1. Unit Tests

Test individual components in isolation:

- **AgentStatusCard**: Status display, CSS classes, prop handling
- **LearningProgress**: Progress bars, charts, metric display
- **StatusPill**: Status indicators, color coding, text display

### 2. Integration Tests

Test page-level component interactions:

- **OverviewPage**: Dashboard layout, data loading, real-time updates
- **MonitoringPage**: System metrics, performance charts, alerts
- **TrainingPage**: RL metrics, learning progress, training controls

### 3. API Tests

Test data fetching and error handling:

- **Client Tests**: HTTP requests, response handling, error states
- **Type Tests**: Data structure validation, type checking

### 4. Hook Tests

Test custom React hooks:

- **useAgentMonitoring**: Data fetching, loading states, error handling
- **useWebSocket**: Connection management, message handling, reconnection

## 🔍 Mock Data

### API Mocks

```typescript
// src/test/__mocks__/api.ts
export const mockApiResponses = {
  status: { status: 'healthy', version: '1.0.0' },
  metrics: { cpu_usage: 45.2, memory_usage: 67.8 },
  rlMetrics: { episodes: 1000, avg_reward: 0.75 }
}
```

### Component Mocks

```typescript
// Mock Chart.js components
vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(() => <div data-testid="line-chart">Line Chart</div>)
}))
```

## 📊 Coverage Reports

The test suite generates comprehensive coverage reports:

- **Text Report**: Console output with coverage percentages
- **HTML Report**: Detailed coverage in `coverage/` directory
- **JSON Report**: Machine-readable coverage data

### Coverage Targets

- **Statements**: > 80%
- **Branches**: > 75%
- **Functions**: > 80%
- **Lines**: > 80%

## 🐛 Debugging Tests

### Common Issues

1. **Async Operations**: Use `waitFor()` for async operations
2. **Mocking**: Ensure mocks are properly configured
3. **Cleanup**: Use `beforeEach` and `afterEach` for cleanup
4. **Timers**: Use `vi.useFakeTimers()` for timer-based tests

### Debug Commands

```bash
# Run tests in watch mode
npm test

# Run specific test file
npm run test:run -- src/test/components/AgentStatusCard.test.tsx

# Run tests with verbose output
npm run test:run -- --reporter=verbose

# Run tests with coverage
npm run test:coverage
```

## 🚀 CI/CD Integration

### GitHub Actions

```yaml
- name: Run Frontend Tests
  run: |
    cd frontend/react-dashboard
    npm ci
    npm run test:run
    npm run test:coverage
```

### Docker

```dockerfile
# Add to Dockerfile
RUN npm run test:run
RUN npm run test:coverage
```

## 📈 Best Practices

1. **Test Behavior, Not Implementation**: Focus on what the component does, not how it does it
2. **Use Descriptive Test Names**: Clear, specific test descriptions
3. **Mock External Dependencies**: Mock APIs, WebSockets, and external libraries
4. **Test Error States**: Ensure components handle errors gracefully
5. **Test Accessibility**: Use `@testing-library/jest-dom` matchers
6. **Keep Tests Fast**: Use mocks and avoid real network requests

## 🔧 Troubleshooting

### Common Problems

1. **Import Errors**: Check file paths and extensions
2. **Mock Issues**: Ensure mocks are properly configured
3. **Async Issues**: Use proper async/await patterns
4. **Cleanup Issues**: Clean up mocks and timers

### Getting Help

- Check the test output for specific error messages
- Use `console.log()` for debugging
- Check the Vitest documentation for advanced features
- Review the mock data in `__mocks__/api.ts`

## 🎉 Success Criteria

A successful test run should show:

- ✅ All tests passing
- ✅ High coverage percentages
- ✅ No console errors
- ✅ Fast execution time
- ✅ Clear test descriptions

The test suite ensures the DSPy Agent frontend is robust, reliable, and ready for production use.
