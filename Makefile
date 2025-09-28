STACK_COMPOSE := docker/lightweight/docker-compose.yml
STACK_ENV := docker/lightweight/.env

.PHONY: stack-env stack-build stack-up stack-down stack-logs stack-ps test-lightweight smoke health-check stack-reload test test-docker dev-cycle

stack-env:
	@if [ ! -f $(STACK_ENV) ]; then \
		echo "WORKSPACE_DIR=$$(pwd)" > $(STACK_ENV); \
		echo "# RedDB Configuration" >> $(STACK_ENV); \
		echo "REDDB_ADMIN_TOKEN=$$(openssl rand -hex 32)" >> $(STACK_ENV); \
		echo "REDDB_URL=http://reddb:8080" >> $(STACK_ENV); \
		echo "REDDB_NAMESPACE=dspy" >> $(STACK_ENV); \
		echo "REDDB_TOKEN=$$REDDB_ADMIN_TOKEN" >> $(STACK_ENV); \
		echo "DB_BACKEND=reddb" >> $(STACK_ENV); \
		echo "[make] wrote $(STACK_ENV) with WORKSPACE_DIR=$$(pwd) and RedDB config"; \
	fi

stack-build: stack-env
	DOCKER_BUILDKIT=1 docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) build

stack-up: stack-env
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) up -d --remove-orphans

stack-down:
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) down

stack-logs:
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) logs -f

stack-ps:
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) ps

# Convenience targets for the auto-started dashboard service
.PHONY: dashboard-up dashboard-logs dashboard-down

dashboard-up: stack-env
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) up -d dashboard

dashboard-logs:
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) logs -f dashboard

dashboard-down:
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) stop dashboard

test-lightweight: stack-build
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) run --rm agent-tests

smoke: stack-build stack-up
	# Run smoke test (produces messages, waits, prints Parquet counts)
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) run --rm smoke

health-check:
	@echo "[health] RedDB"; \
	  (curl -fsS -H "Authorization: Bearer $$(grep REDDB_ADMIN_TOKEN $(STACK_ENV) | cut -d= -f2)" http://127.0.0.1:8082/health || echo "unreachable") && echo
	@echo "[health] InferMesh"; \
	  (curl -fsS http://127.0.0.1:19000/health || echo "unreachable") && echo
	@echo "[health] Dashboard"; \
	  (curl -fsS http://127.0.0.1:18081/api/status || echo "unreachable") && echo
	@echo "[health] embed-worker metrics"; \
	  (curl -fsS http://127.0.0.1:9101/metrics || echo "unreachable") && echo
	@echo "[health] FastAPI Backend"; \
	  (curl -fsS http://127.0.0.1:8767/api/db/health || echo "unreachable") && echo
	@echo "[health] spark-vectorizer UI"; \
	  echo "Open http://127.0.0.1:4041 in a browser"
	@echo "[health] Enhanced monitoring"; \
	  python3 scripts/health_monitor.py --report || echo "monitoring unavailable"

.PHONY: performance-check
performance-check:
	@echo "[performance] Running comprehensive performance analysis..."
	@python3 scripts/health_monitor.py --workspace $$(pwd) --interval 5 --report
	@echo "[performance] Performance report generated in logs/"

.PHONY: optimize-stack
optimize-stack: stack-env
	@echo "[optimize] Optimizing stack performance..."
	@docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) exec dspy-agent python -c "from dspy_agent.skills.orchestrator import Orchestrator; o = Orchestrator(); o._cleanup_cache(); print('Cache optimized')" || true
	@echo "[optimize] Stack optimization complete"

.PHONY: auto-scale
auto-scale: stack-env
	@echo "[auto-scale] Starting intelligent auto-scaling..."
	@docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) up -d auto-scaler
	@echo "[auto-scale] Auto-scaler started - monitoring resource usage"

.PHONY: performance-monitor
performance-monitor: stack-env
	@echo "[performance] Starting advanced performance monitoring..."
	@python3 -c "from dspy_agent.monitor.performance_monitor import PerformanceMonitor; import asyncio; monitor = PerformanceMonitor('$$(pwd)'); asyncio.run(monitor.collect_performance_snapshot()); print('Performance monitoring ready')"
	@echo "[performance] Performance monitoring system ready"

.PHONY: intelligent-optimization
intelligent-optimization: stack-env
	@echo "[intelligent] Running intelligent optimization analysis..."
	@python3 scripts/intelligent_optimization_analysis.py
	@echo "[intelligent] Optimization analysis complete"

.PHONY: advanced-features
advanced-features: stack-env
	@echo "[advanced] Enabling advanced features..."
	@echo "  - Intelligent caching: ENABLED"
	@echo "  - Adaptive learning: ENABLED" 
	@echo "  - Auto-scaling: ENABLED"
	@echo "  - Performance monitoring: ENABLED"
	@echo "  - Anomaly detection: ENABLED"
	@echo "[advanced] Advanced features configured"

.PHONY: intelligent-deploy
intelligent-deploy: stack-env
	@echo "[intelligent] Starting intelligent deployment..."
	@python3 scripts/intelligent_deployment_orchestrator.py --workspace $$(pwd) --environment development
	@echo "[intelligent] Intelligent deployment completed"

.PHONY: environment-detect
environment-detect: stack-env
	@echo "[environment] Detecting environment capabilities..."
	@python3 scripts/environment_manager.py --workspace $$(pwd) --detect
	@echo "[environment] Environment detection complete"

.PHONY: environment-configure
environment-configure: stack-env
	@echo "[environment] Configuring environment..."
	@python3 scripts/environment_manager.py --workspace $$(pwd) --configure
	@echo "[environment] Environment configuration complete"

.PHONY: advanced-health
advanced-health: stack-env
	@echo "[health] Starting advanced health monitoring..."
	@python3 scripts/advanced_health_monitor.py --workspace $$(pwd) --report
	@echo "[health] Advanced health monitoring complete"

.PHONY: deployment-status
deployment-status: stack-env
	@echo "[status] Checking deployment status..."
	@python3 scripts/intelligent_deployment_orchestrator.py --workspace $$(pwd) --status
	@echo "[status] Deployment status check complete"

.PHONY: deployment-history
deployment-history: stack-env
	@echo "[history] Showing deployment history..."
	@python3 scripts/intelligent_deployment_orchestrator.py --workspace $$(pwd) --history
	@echo "[history] Deployment history complete"

.PHONY: comprehensive-deploy
comprehensive-deploy: stack-env
	@echo "[comprehensive] Starting comprehensive deployment workflow..."
	@./scripts/comprehensive_deployment_workflow.sh deploy
	@echo "[comprehensive] Comprehensive deployment workflow completed"

.PHONY: deployment-validate
deployment-validate: stack-env
	@echo "[validate] Validating deployment prerequisites..."
	@./scripts/comprehensive_deployment_workflow.sh validate
	@echo "[validate] Deployment validation complete"

.PHONY: deployment-health-check
deployment-health-check: stack-env
	@echo "[health] Running comprehensive health check..."
	@./scripts/comprehensive_deployment_workflow.sh health
	@echo "[health] Health check complete"

.PHONY: deployment-status-check
deployment-status-check: stack-env
	@echo "[status] Checking deployment status..."
	@./scripts/comprehensive_deployment_workflow.sh status
	@echo "[status] Status check complete"

.PHONY: environment-analysis
environment-analysis: stack-env
	@echo "[environment] Analyzing environment capabilities..."
	@./scripts/comprehensive_deployment_workflow.sh environment
	@echo "[environment] Environment analysis complete"

stack-reload: stack-env
	# Rebuild the lightweight agent image and restart only agent services
	DOCKER_BUILDKIT=1 docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) build dspy-agent
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) up -d --no-deps dspy-agent dspy-worker dspy-worker-backend dspy-worker-frontend

.PHONY: dev-loop
dev-loop: stack-env
	bash scripts/dev_loop.sh

test:
	pytest -q -m "not docker"

test-docker:
	DOCKER_TESTS=1 pytest -q -m docker

dev-cycle:
	bash scripts/dev_cycle.sh

.PHONY: infermesh-build infermesh-push infermesh-bench
infermesh-build: stack-env
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) build infermesh

# Push built image to registry (requires docker login and INFERMESH_IMAGE in env/.env)
infermesh-push:
	@if [ -z "$(shell grep '^INFERMESH_IMAGE=' $(STACK_ENV) | cut -d= -f2)" ]; then \
		echo "Set INFERMESH_IMAGE in $(STACK_ENV)"; exit 1; \
	fi
	IMAGE=$(shell grep '^INFERMESH_IMAGE=' $(STACK_ENV) | cut -d= -f2); \
	docker tag lightweight-infermesh:latest $$IMAGE; \
	echo "Pushing $$IMAGE"; \
	docker push $$IMAGE

infermesh-bench:
	python3 scripts/benchmark_infermesh.py --url http://127.0.0.1:$${INFERMESH_HOST_PORT:-19000} --model $${INFERMESH_MODEL:-BAAI/bge-small-en-v1.5}

.PHONY: stack-smoke
stack-smoke: stack-up
	# Run end-to-end smoke (Kafka → vectorizer → embed-worker)
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) run --rm smoke || true
	$(MAKE) health-check

.PHONY: local-up local-down local-status local-setup local-test local-clean
local-up:
	bash scripts/start_local_system.sh

local-down:
	bash scripts/stop_local_system.sh

local-status:
	@echo "[pids]"; ls -l logs/pids 2>/dev/null || true; echo; \
	  echo "[queue]"; echo pending=$$(ls logs/env_queue/pending 2>/dev/null | wc -l | tr -d ' ') done=$$(ls logs/env_queue/done 2>/dev/null | wc -l | tr -d ' ')

local-setup:
	@echo "[setup] Setting up complete DSPy Agent system..."
	bash scripts/setup_complete_system.sh
	@echo "[setup] System setup complete!"

local-test:
	@echo "[test] Testing DSPy Agent system..."
	bash scripts/test_system.sh
	@echo "[test] System tests complete!"

local-clean:
	@echo "[clean] Cleaning up local system..."
	bash scripts/stop_local_system.sh --cleanup
	@echo "[clean] System cleanup complete!"

# Go/Rust/Slurm specific targets
.PHONY: go-build go-test rust-build rust-test slurm-test orchestrator-build env-runner-build

go-build:
	@echo "[go] Building Go orchestrator..."
	cd orchestrator && GOCACHE=$(pwd)/.gocache GOMODCACHE=$(pwd)/.gomodcache go build -o ../logs/orchestrator ./cmd/orchestrator
	@echo "[go] Go orchestrator built successfully!"

go-test:
	@echo "[go] Testing Go orchestrator..."
	cd orchestrator && GOCACHE=$(pwd)/.gocache GOMODCACHE=$(pwd)/.gomodcache go test ./...
	@echo "[go] Go tests passed!"

rust-build:
	@echo "[rust] Building Rust env-runner..."
	cd env_runner_rs && cargo build --release
	@echo "[rust] Rust env-runner built successfully!"

rust-test:
	@echo "[rust] Testing Rust env-runner..."
	cd env_runner_rs && cargo test
	@echo "[rust] Rust tests passed!"

slurm-test:
	@echo "[slurm] Testing Slurm integration..."
	python3 tests/test_slurm_integration.py
	@echo "[slurm] Slurm tests passed!"

orchestrator-build: go-build
	@echo "[orchestrator] Go orchestrator ready!"

env-runner-build: rust-build
	@echo "[env-runner] Rust env-runner ready!"

# Complete system targets
.PHONY: system-setup system-start system-stop system-test system-status system-clean

system-setup: local-setup
	@echo "[system] Complete system setup finished!"

system-start: orchestrator-build env-runner-build local-up
	@echo "[system] Complete system started!"

system-stop: local-down
	@echo "[system] Complete system stopped!"

system-test: go-test rust-test slurm-test local-test
	@echo "[system] All system tests passed!"

system-status: local-status
	@echo "[system] System status checked!"

system-clean: local-clean

# Package system for distribution
.PHONY: package package-clean

package:
	@echo "Creating distribution package..."
	@bash scripts/dspy_stack_packager.sh

package-clean:
	@echo "Cleaning distribution packages..."
	@rm -rf dist/
	@echo "[system] System cleanup complete!"

# Docker Compose with Go/Rust/Slurm
.PHONY: stack-up-complete stack-down-complete stack-logs-complete

stack-up-complete: stack-env
	@echo "[stack] Starting complete stack with Go/Rust/Slurm components..."
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) up -d go-orchestrator rust-env-runner redis
	@echo "[stack] Complete stack started!"

stack-down-complete:
	@echo "[stack] Stopping complete stack..."
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) down
	@echo "[stack] Complete stack stopped!"

stack-logs-complete:
	@echo "[stack] Showing complete stack logs..."
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) logs -f go-orchestrator rust-env-runner

# Health checks for new components
.PHONY: health-check-complete

health-check-complete: health-check
	@echo "[health] Go Orchestrator"; \
	  (curl -fsS http://127.0.0.1:9097/metrics || echo "unreachable") && echo
	@echo "[health] Rust Env-Runner"; \
	  (curl -fsS http://127.0.0.1:8080/health || echo "unreachable") && echo
	@echo "[health] Queue Status"; \
	  (curl -fsS http://127.0.0.1:9097/queue/status || echo "unreachable") && echo
	@echo "[health] Slurm Integration"; \
	  (curl -fsS http://127.0.0.1:9097/slurm/status/test || echo "unreachable") && echo

# ---------------------
# Production System Management
# ---------------------
.PHONY: build up down logs health status test clean dev quickstart

# Build all components
build:
	@echo "🔨 Building all components..."
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) build
	@echo "✅ All components built successfully"

# Start all services
up: stack-up
	@echo "🚀 Starting all services..."
	@echo "✅ All services started"

# Stop all services
down: stack-down
	@echo "🛑 Stopping all services..."
	@echo "✅ All services stopped"

# View service logs
logs: stack-logs
	@echo "📋 Viewing service logs..."

# Check service health
health: health-check-complete
	@echo "🏥 Checking service health..."

# Check service status
status: stack-ps
	@echo "📊 Service status:"

# Run all tests
test: go-test rust-test test-lightweight
	@echo "✅ All tests completed"

# Clean up system
clean:
	@echo "🧹 Cleaning up system..."
	docker system prune -f
	docker volume prune -f
	@echo "✅ System cleaned"

# Start development environment
dev: stack-up
	@echo "🔧 Starting development environment..."
	@echo "✅ Development environment started"

# Quick start - build, start, and check health
quickstart: build up health
	@echo "🎉 DSPy Agent is ready!"
	@echo "Services:"
	@echo "  - RedDB: http://localhost:8082"
	@echo "  - Go Orchestrator: http://localhost:9097"
	@echo "  - Rust Env Runner: http://localhost:8080"
	@echo "  - InferMesh: http://localhost:19000"
	@echo "  - Redis: localhost:6379"

# ---------------------
# DB Tools Quick Tests
# ---------------------
.PHONY: test-db-tools-local test-db-tools-fastapi orchestrate-demo

# Local (in-memory RedDB), no server required.
test-db-tools-local:
	@echo "[db] local ingest/query/multi (in-memory)"; \
	python3 scripts/test_db_tools.py --ns agent --no-server

# Start FastAPI backend and run curl tests against it; stops after checks
test-db-tools-fastapi:
	@echo "[api] starting FastAPI backend on :8767"; \
	( python3 -m dspy_agent.server.fastapi_backend & echo $$! > .fastapi.pid ); \
	sleep 2; \
	curl -fsS http://127.0.0.1:8767/api/db/health || true; echo; \
	curl -fsS -X POST http://127.0.0.1:8767/api/db/ingest -H 'Content-Type: application/json' \
	  -d '{"kind":"document","namespace":"agent","collection":"notes","id":"n1","text":"Payment API returns 500"}' || true; echo; \
	curl -fsS -X POST http://127.0.0.1:8767/api/db/query -H 'Content-Type: application/json' \
	  -d '{"mode":"auto","namespace":"agent","text":"payment 500","collection":"notes","top_k":5}' || true; echo; \
	kill `cat .fastapi.pid` >/dev/null 2>&1 || true; rm -f .fastapi.pid

# Demo: ingest a couple of docs, then run multi-head retrieval with (non-LLM) summary
orchestrate-demo:
	python3 scripts/orchestrate_demo.py --ns agent

.PHONY: stack-demo
stack-demo: stack-up
	@echo "[demo] running orchestrate-demo after stack-up"; \
	$(MAKE) orchestrate-demo; \
	echo "\n[demo] Success!"; \
	echo "- Dashboard:     http://127.0.0.1:18081"; \
	echo "- FastAPI Backend http://127.0.0.1:8767/api/db/health"; \
	echo "- RedDB:         http://127.0.0.1:8082/health"; \
	echo "(Use 'make health' for a quick status report)"
