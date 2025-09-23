# DSPy Agent Comprehensive Testing and Deployment Guide

This guide provides a complete overview of the testing, build, and deployment automation for the DSPy Agent project.

## 🚀 Quick Start

### Automated Setup (Recommended)
```bash
# Full automated setup
./scripts/automated_agent_setup.sh full

# Or step by step
./scripts/automated_agent_setup.sh setup    # Check prerequisites
./scripts/automated_agent_setup.sh docker   # Setup Docker environment
./scripts/automated_agent_setup.sh start     # Start all services
```

### Manual Testing
```bash
# Run comprehensive tests
python scripts/run_comprehensive_tests.py --test-types unit integration docker

# Run health monitoring
python scripts/health_monitor.py --continuous --interval 300

# Build and deploy
./scripts/comprehensive_build_and_deploy.sh all
```

## 📋 Overview

This project now includes comprehensive automation for:

1. **Testing** - Unit, integration, performance, security, and Docker tests
2. **Building** - Automated package and Docker image builds
3. **Deployment** - Dev, test, and production deployment pipelines
4. **Monitoring** - Health checks and continuous monitoring
5. **CI/CD** - GitHub Actions pipeline for automated testing and deployment

## 🧪 Testing Framework

### Test Types

#### 1. Unit Tests
- **Location**: `tests/test_comprehensive_agent.py`
- **Coverage**: All core components
- **Command**: `python scripts/run_comprehensive_tests.py --test-types unit`

#### 2. Integration Tests
- **Location**: `tests/test_comprehensive_agent.py::TestAgentComprehensive`
- **Coverage**: Component interactions, workflows
- **Command**: `python scripts/run_comprehensive_tests.py --test-types integration`

#### 3. Docker Tests
- **Coverage**: Container builds, Docker Compose validation
- **Command**: `python scripts/run_comprehensive_tests.py --test-types docker`

#### 4. Performance Tests
- **Coverage**: Memory usage, CPU performance, benchmarks
- **Command**: `python scripts/run_comprehensive_tests.py --test-types performance`

#### 5. Security Tests
- **Coverage**: Hardcoded secrets, file permissions, SQL injection
- **Command**: `python scripts/run_comprehensive_tests.py --test-types security`

#### 6. Lint Tests
- **Coverage**: Code style (flake8, black, mypy)
- **Command**: `python scripts/run_comprehensive_tests.py --test-types lint`

#### 7. Package Tests
- **Coverage**: Package build, validation, installation
- **Command**: `python scripts/run_comprehensive_tests.py --test-types package`

### Running Tests

```bash
# Run all tests
python scripts/run_comprehensive_tests.py

# Run specific test types
python scripts/run_comprehensive_tests.py --test-types unit integration

# Verbose output
python scripts/run_comprehensive_tests.py --verbose

# Save report to file
python scripts/run_comprehensive_tests.py --output custom_report.json
```

## 🏗️ Build and Deployment

### Build Pipeline

#### 1. Package Build
```bash
# Build Python package
./scripts/comprehensive_build_and_deploy.sh build

# Test package
python -m twine check dist/*
```

#### 2. Docker Build
```bash
# Build Docker images
cd docker/lightweight
make stack-build

# Test Docker images
docker run --rm dspy-lightweight:latest --help
```

#### 3. Full Build Pipeline
```bash
# Complete build process
./scripts/comprehensive_build_and_deploy.sh all
```

### Deployment Environments

#### Development
```bash
# Deploy to development
./scripts/comprehensive_build_and_deploy.sh deploy-dev

# Or use automated setup
./scripts/automated_agent_setup.sh full
```

#### Test
```bash
# Deploy to test environment
./scripts/comprehensive_build_and_deploy.sh deploy-test
```

#### Production
```bash
# Deploy to production (requires confirmation)
./scripts/comprehensive_build_and_deploy.sh deploy-prod
```

### Docker Compose Stack

The project includes a complete Docker Compose stack with:

- **dspy-agent**: Main agent service
- **ollama**: Local LLM service
- **kafka**: Message streaming
- **zookeeper**: Kafka coordination
- **infermesh**: Embedding service
- **embed-worker**: Embedding worker
- **spark**: Data processing
- **dspy-worker**: Background workers

#### Stack Management
```bash
# Start stack
make stack-up

# Stop stack
make stack-down

# View logs
make stack-logs

# Health checks
make health-check

# Run tests
make test-lightweight
```

## 📊 Monitoring and Health Checks

### Health Monitor

The health monitoring system checks:

- **System Resources**: CPU, memory, disk usage
- **Docker Services**: Container status and health
- **HTTP Endpoints**: Service availability
- **Agent Logs**: Error rates and warnings
- **Workspace Health**: Permissions and disk space

#### Usage
```bash
# Single health check
python scripts/health_monitor.py

# Continuous monitoring
python scripts/health_monitor.py --continuous --interval 300

# Save report
python scripts/health_monitor.py --output health_report.json
```

#### Health Check Endpoints
- Agent: `http://localhost:8765`
- Dashboard: `http://localhost:18081`
- Ollama: `http://localhost:11435`
- Kafka: `localhost:9092`
- InferMesh: `http://localhost:9000`
- Embed Worker: `http://localhost:9101`
- Spark: `http://localhost:4041`

### Monitoring Dashboard

Access the monitoring dashboard at `http://localhost:8765` to view:

- Real-time agent activity
- Performance metrics
- Error logs
- System health status

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The project includes a comprehensive CI/CD pipeline (`.github/workflows/ci-cd.yml`) that:

1. **Code Quality**: Linting, type checking, formatting
2. **Testing**: Unit, integration, Docker, performance tests
3. **Security**: Vulnerability scanning
4. **Building**: Package and Docker image builds
5. **Deployment**: Automated deployment to dev/test/prod

#### Pipeline Triggers
- **Push to main/develop**: Full test suite
- **Pull Requests**: Code quality and testing
- **Releases**: Build and publish to PyPI

### Local CI/CD Simulation

```bash
# Run full pipeline locally
./scripts/comprehensive_build_and_deploy.sh all

# Test specific components
./scripts/comprehensive_build_and_deploy.sh test
./scripts/comprehensive_build_and_deploy.sh build
```

## 🛠️ Development Workflow

### Daily Development
```bash
# Start development environment
./scripts/automated_agent_setup.sh start

# Run tests during development
python scripts/run_comprehensive_tests.py --test-types unit integration

# Monitor health
python scripts/health_monitor.py --continuous
```

### Code Changes
```bash
# Run linting
python scripts/run_comprehensive_tests.py --test-types lint

# Run tests
python scripts/run_comprehensive_tests.py --test-types unit integration

# Build and test
./scripts/comprehensive_build_and_deploy.sh build
```

### Release Process
```bash
# 1. Run full test suite
./scripts/comprehensive_build_and_deploy.sh all

# 2. Create release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# 3. Deploy to production
./scripts/comprehensive_build_and_deploy.sh deploy-prod
```

## 📁 Project Structure

```
dspy_stuff/
├── scripts/
│   ├── automated_agent_setup.sh          # Automated setup
│   ├── comprehensive_build_and_deploy.sh # Build and deploy
│   ├── health_monitor.py                  # Health monitoring
│   └── run_comprehensive_tests.py         # Test runner
├── tests/
│   └── test_comprehensive_agent.py        # Comprehensive test suite
├── .github/workflows/
│   └── ci-cd.yml                          # CI/CD pipeline
├── docker/lightweight/
│   ├── docker-compose.yml                # Docker stack
│   ├── Dockerfile                        # Main image
│   └── embed_worker.Dockerfile           # Worker image
├── logs/                                 # Application logs
└── test-results/                         # Test reports
```

## 🔧 Configuration

### Environment Variables

```bash
# Core configuration
export DSPY_LOG_LEVEL=INFO
export DSPY_AUTO_TRAIN=false
export USE_OLLAMA=true
export OLLAMA_MODEL=qwen3:1.7b

# Service endpoints
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export REDDB_URL=http://localhost:8080
export REDDB_NAMESPACE=dspy
export REDDB_TOKEN=development-token

# Workspace
export WORKSPACE_DIR=/path/to/workspace
```

### Docker Environment

```bash
# Docker Compose environment
WORKSPACE_DIR=/path/to/workspace
REDDB_URL=http://localhost:8080
REDDB_NAMESPACE=dspy
REDDB_TOKEN=development-token
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Docker Services Not Starting
```bash
# Check Docker daemon
docker info

# Restart Docker services
make stack-down
make stack-up

# Check logs
make stack-logs
```

#### 2. Test Failures
```bash
# Run specific test types
python scripts/run_comprehensive_tests.py --test-types unit

# Check test output
cat logs/test_runner.log

# Run with verbose output
python scripts/run_comprehensive_tests.py --verbose
```

#### 3. Health Check Failures
```bash
# Check individual components
python scripts/health_monitor.py

# Check Docker services
docker-compose ps

# Check system resources
python scripts/health_monitor.py | grep system_resources
```

#### 4. Build Failures
```bash
# Clean build
rm -rf dist/ build/
./scripts/comprehensive_build_and_deploy.sh build

# Check dependencies
pip list
```

### Log Files

- **Test Results**: `logs/test_runner.log`
- **Health Monitor**: `logs/health_monitor.log`
- **Agent Logs**: `logs/agent_*.jsonl`
- **Docker Logs**: `make stack-logs`

### Performance Issues

```bash
# Check system resources
python scripts/health_monitor.py

# Monitor Docker containers
docker stats

# Check disk space
df -h
```

## 📈 Metrics and Reporting

### Test Reports

Test reports are generated in JSON format and include:

- Test results and coverage
- Performance metrics
- Security scan results
- Recommendations for improvements

### Health Reports

Health reports include:

- System resource usage
- Service status
- Error rates
- Performance metrics
- Recommendations

### Continuous Monitoring

```bash
# Start continuous monitoring
python scripts/health_monitor.py --continuous --interval 300

# View real-time dashboard
open http://localhost:8765
```

## 🎯 Best Practices

### Development
1. Run tests before committing code
2. Use the automated setup for consistent environments
3. Monitor health during development
4. Follow the CI/CD pipeline for releases

### Testing
1. Run unit tests frequently during development
2. Run integration tests before major changes
3. Use performance tests for optimization
4. Run security tests before releases

### Deployment
1. Test in development environment first
2. Use test environment for validation
3. Deploy to production only after thorough testing
4. Monitor health after deployment

### Monitoring
1. Set up continuous health monitoring
2. Review logs regularly
3. Monitor system resources
4. Set up alerts for critical issues

## 🆘 Support

### Documentation
- **API Documentation**: `docs/API.md`
- **Deployment Guide**: `docs/DEPLOYMENT.md`
- **Testing Guide**: `docs/TESTING.md`
- **Production Guide**: `docs/PRODUCTION.md`

### Getting Help
1. Check the logs for error messages
2. Run health checks to identify issues
3. Review the troubleshooting section
4. Check GitHub issues for known problems

### Contributing
1. Follow the development workflow
2. Run all tests before submitting PRs
3. Update documentation for new features
4. Follow the code style guidelines

---

This comprehensive testing and deployment system ensures the DSPy Agent is robust, reliable, and easy to deploy across different environments. The automation reduces manual errors and provides consistent, repeatable processes for development, testing, and deployment.
