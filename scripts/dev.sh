#!/bin/bash
# DSPy Agent Development Workflow - Easy Commands for Human Users

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}🚀 Blampert Development Workflow${NC}"
    echo "=================================="
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Not in DSPy agent directory. Please run from project root."
    exit 1
fi

# Main command handling
case "$1" in
    "help"|"")
        print_header
        echo "Available commands:"
        echo ""
        echo "📦 Package Management:"
        echo "  dev.sh build          - Build the package"
        echo "  dev.sh test           - Run comprehensive test suite"
        echo "  dev.sh test --quick   - Run quick tests for development"
        echo "  dev.sh test --type X  - Run specific test type (unit/integration/rl/agent/quality/import/cli)"
        echo "  dev.sh lint           - Run linter"
        echo "  dev.sh format         - Format code"
        echo ""
        echo "🔄 Development Cycle:"
        echo "  dev.sh quick [msg]    - Quick dev cycle (format, test, commit, push)"
        echo "  dev.sh commit [msg]   - Commit and push changes"
        echo ""
        echo "📈 Version & Release:"
        echo "  dev.sh version        - Show current version"
        echo "  dev.sh bump [type]    - Bump version (patch/minor/major)"
        echo "  dev.sh release [type] - Full release workflow"
        echo ""
        echo "📤 Publishing:"
        echo "  dev.sh publish        - Publish to PyPI"
        echo "  dev.sh publish-test   - Publish to Test PyPI"
        echo ""
        echo "🔍 Status & Info:"
        echo "  dev.sh status         - Show git status"
        echo "  dev.sh info           - Show project info"
        echo ""
        echo "Examples:"
        echo "  dev.sh quick 'Fix bug in CLI'"
        echo "  dev.sh release patch"
        echo "  dev.sh bump minor"
        ;;
    
    "build")
        print_header
        echo "Building package..."
        uv run python scripts/dev_workflow.py build
        print_success "Package built successfully!"
        ;;
    
    "test")
        print_header
        echo "Running tests..."
        if [ "$2" = "--quick" ]; then
            echo "Running quick tests..."
            uv run python scripts/dev_workflow.py test --quick
        elif [ "$2" = "--type" ] && [ -n "$3" ]; then
            echo "Running $3 tests..."
            uv run python scripts/dev_workflow.py test --type "$3"
        else
            echo "Running comprehensive test suite..."
            uv run python scripts/dev_workflow.py test
        fi
        print_success "Tests completed!"
        ;;
    
    "lint")
        print_header
        echo "Running linter..."
        uv run python scripts/dev_workflow.py lint
        print_success "Linting completed!"
        ;;
    
    "format")
        print_header
        echo "Formatting code..."
        uv run python scripts/dev_workflow.py lint --fix
        print_success "Code formatted!"
        ;;
    
    "quick")
        print_header
        echo "Starting quick development cycle..."
        MESSAGE="${2:-Quick dev update: $(date '+%Y-%m-%d %H:%M')}"
        uv run python scripts/dev_workflow.py quick --message "$MESSAGE"
        print_success "Quick dev cycle completed!"
        ;;
    
    "commit")
        print_header
        echo "Committing and pushing changes..."
        MESSAGE="${2:-Update: $(date '+%Y-%m-%d %H:%M')}"
        uv run python scripts/dev_workflow.py git --commit "$MESSAGE" --push
        print_success "Changes committed and pushed!"
        ;;
    
    "version")
        print_header
        echo "Current version:"
        uv run python scripts/dev_workflow.py version --show
        ;;
    
    "bump")
        print_header
        TYPE="${2:-patch}"
        echo "Bumping version ($TYPE)..."
        uv run python scripts/dev_workflow.py version --type "$TYPE"
        print_success "Version bumped!"
        ;;
    
    "release")
        print_header
        TYPE="${2:-patch}"
        echo "Starting full release workflow ($TYPE)..."
        read -p "Are you sure you want to release? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            uv run python scripts/dev_workflow.py release --type "$TYPE"
            print_success "Release completed!"
        else
            print_warning "Release cancelled."
        fi
        ;;
    
    "publish")
        print_header
        echo "Publishing to PyPI..."
        read -p "Are you sure you want to publish to PyPI? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            uv run python scripts/dev_workflow.py publish
            print_success "Published to PyPI!"
        else
            print_warning "Publish cancelled."
        fi
        ;;
    
    "publish-test")
        print_header
        echo "Publishing to Test PyPI..."
        uv run python scripts/dev_workflow.py publish --test
        print_success "Published to Test PyPI!"
        ;;
    
    "status")
        print_header
        echo "Git status:"
        uv run python scripts/dev_workflow.py git --status
        ;;
    
    "info")
        print_header
        echo "Project Information:"
        echo "==================="
        echo "Current version: $(uv run python scripts/dev_workflow.py version --show | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')"
        echo "Git branch: $(git branch --show-current)"
        echo "Git status:"
        git status --short
        echo ""
        echo "Recent commits:"
        git log --oneline -5
        ;;
    
    *)
        print_error "Unknown command: $1"
        echo "Run 'dev.sh help' for available commands."
        exit 1
        ;;
esac
