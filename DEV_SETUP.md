# DSPy Development Setup Guide

## Quick Start

### 1. Start the Full Stack
```bash
# Make the startup script executable
chmod +x start-dev.sh

# Start everything (Docker + Frontend + Backend)
./start-dev.sh
```

### 2. Alternative: Manual Setup

#### Prerequisites
- Docker Desktop running
- Node.js 18+ installed
- Python 3.8+ installed

#### Backend Services
```bash
# Start the full DSPy stack with RedDB
make stack-env    # Create environment file
make stack-build  # Build Docker images
make stack-up     # Start all services
```

#### Frontend Development
```bash
# Start the React frontend
cd frontend/react-dashboard
npm install
npm run dev
```

## Services

Once running, you'll have access to:

- **Frontend**: http://localhost:5176 (React Dashboard)
- **Backend API**: http://localhost:8080 (Enhanced Dashboard Server)
- **RedDB**: http://localhost:8080 (with authentication)
- **Agent**: http://localhost:8765
- **Kafka**: localhost:9092
- **Redis**: localhost:6379

## Development Commands

### Stack Management
```bash
make stack-up      # Start all services
make stack-down    # Stop all services
make stack-logs    # View logs
make stack-ps      # Check service status
make health-check  # Run health checks
```

### Frontend Development
```bash
cd frontend/react-dashboard
npm run dev        # Start development server
npm run build      # Build for production
npm run preview    # Preview production build
```

### Backend Development
```bash
# Start just the dashboard server
python3 enhanced_dashboard_server.py 8080

# Start with specific port
python3 enhanced_dashboard_server.py 8081
```

### Training Quickstart
Run a quick RL session and persist analytics locally (no external DB required):

```bash
python -m dspy_agent.cli rl quick-train --workspace . --signature CodeContextSig --steps 50
python -m dspy_agent.cli rl report
python -m dspy_agent.cli rl recent --limit 5
```

To use RedDB for persistence (optional):

```bash
export REDDB_URL=http://localhost:8080
export REDDB_NAMESPACE=dspy
export REDDB_TOKEN=<token if required>
```

 

## Troubleshooting

### Docker Issues
```bash
# Check if Docker is running
docker info

# If not running, start Docker Desktop
open -a Docker
```

### Port Conflicts
```bash
# Kill processes on common ports
lsof -ti:8080 | xargs kill -9
lsof -ti:5176 | xargs kill -9
```

### Frontend Build Issues
```bash
# Clear node modules and reinstall
cd frontend/react-dashboard
rm -rf node_modules package-lock.json
npm install
```

## Architecture

The system consists of:

1. **Frontend**: React dashboard with TypeScript
2. **Backend**: Python enhanced dashboard server
3. **Database**: RedDB for coordination and data storage
4. **Message Queue**: Kafka for event streaming
5. **Cache**: Redis for fast data access
6. **Agent**: DSPy agent for AI operations

## API Endpoints

The backend provides these key endpoints:

- `GET /api/status` - System status
- `GET /api/signatures` - Signature data
- `GET /api/verifiers` - Verifier data
- `GET /api/system-topology` - System topology
- `GET /api/bus-metrics` - Bus metrics
- `GET /api/dev-cycle/status` - Development cycle status

## Development Workflow

1. **Start the stack**: `./start-dev.sh`
2. **Develop frontend**: Edit files in `frontend/react-dashboard/src/`
3. **Develop backend**: Edit `enhanced_dashboard_server.py`
4. **Test changes**: Frontend auto-reloads, backend needs restart
5. **Stop when done**: `make stack-down`

## Production Deployment

For production deployment, see:
- `docs/DEPLOYMENT.md`
- `deploy/` directory for Kubernetes/Helm charts
- `docker/` directory for containerized deployment
