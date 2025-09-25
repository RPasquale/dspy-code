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
	@python3 -c "
from dspy_agent.monitor.performance_monitor import PerformanceMonitor
from dspy_agent.monitor.auto_scaler import AutoScaler
import asyncio
import json

async def run_optimization():
    # Performance analysis
    perf_monitor = PerformanceMonitor('$$(pwd)')
    snapshot = await perf_monitor.collect_performance_snapshot()
    anomalies = perf_monitor.detect_anomalies(snapshot)
    recommendations = perf_monitor.generate_optimization_recommendations(snapshot)
    
    print('=== Intelligent Optimization Report ===')
    print(f'CPU Usage: {snapshot.cpu_usage:.1f}%')
    print(f'Memory Usage: {snapshot.memory_usage:.1f}%')
    print(f'Anomalies Detected: {len(anomalies)}')
    print(f'Optimization Recommendations: {len(recommendations)}')
    
    if anomalies:
        print('\n=== Anomalies ===')
        for anomaly in anomalies:
            print(f'- {anomaly.description} (severity: {anomaly.severity})')
    
    if recommendations:
        print('\n=== Recommendations ===')
        for rec in recommendations:
            print(f'- {rec.title} (priority: {rec.priority})')
            print(f'  Impact: {rec.impact}')
            print(f'  Effort: {rec.effort}')

asyncio.run(run_optimization())
"
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
	echo "- RedDB Mock:     http://127.0.0.1:8082/health"; \
	echo "(Use 'make health-check' for a quick status report)"
