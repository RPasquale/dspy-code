#!/bin/bash

# DSPy Agent Frontend Test Runner
# Comprehensive test suite for React components, hooks, and API integration

set -e

echo "🧪 DSPy Agent Frontend Test Suite"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the frontend directory."
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_warning "node_modules not found. Installing dependencies..."
    npm install
fi

# Check if vitest is installed
if ! npm list vitest > /dev/null 2>&1; then
    print_warning "Vitest not found. Installing test dependencies..."
    npm install --save-dev vitest @vitest/ui @testing-library/react @testing-library/jest-dom @testing-library/user-event
fi

print_status "Running test suite..."

# Run different types of tests
echo ""
echo "🔍 Running Unit Tests..."
echo "------------------------"
npm run test:run -- --reporter=verbose src/test/components/

echo ""
echo "🔗 Running Integration Tests..."
echo "-------------------------------"
npm run test:run -- --reporter=verbose src/test/pages/

echo ""
echo "🌐 Running API Tests..."
echo "-----------------------"
npm run test:run -- --reporter=verbose src/test/api/

echo ""
echo "🎣 Running Hook Tests..."
echo "------------------------"
npm run test:run -- --reporter=verbose src/test/hooks/

echo ""
echo "📱 Running App Tests..."
echo "------------------------"
npm run test:run -- --reporter=verbose src/test/App.test.tsx

echo ""
echo "📊 Running Coverage Report..."
echo "-----------------------------"
npm run test:coverage

echo ""
print_success "All tests completed!"
echo ""
echo "📋 Test Summary:"
echo "  • Unit Tests: Component rendering and behavior"
echo "  • Integration Tests: Page-level component interactions"
echo "  • API Tests: Data fetching and error handling"
echo "  • Hook Tests: Custom React hooks functionality"
echo "  • App Tests: Main application routing and structure"
echo ""
echo "🎉 Frontend test suite is ready for production!"
