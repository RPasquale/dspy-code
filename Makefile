STACK_COMPOSE := docker/lightweight/docker-compose.yml
STACK_ENV := docker/lightweight/.env

.PHONY: stack-env stack-build stack-up stack-down stack-logs stack-ps test-lightweight smoke health-check stack-reload test test-docker dev-cycle

stack-env:
	@if [ ! -f $(STACK_ENV) ]; then \
		echo "WORKSPACE_DIR=$$(pwd)" > $(STACK_ENV); \
		echo "# You can also set REDDB_URL/REDDB_NAMESPACE/REDDB_TOKEN here" >> $(STACK_ENV); \
		echo "[make] wrote $(STACK_ENV) with WORKSPACE_DIR=$$(pwd)"; \
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

test-lightweight: stack-build
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) run --rm agent-tests

smoke: stack-build stack-up
	# Run smoke test (produces messages, waits, prints Parquet counts)
	docker compose -f $(STACK_COMPOSE) --env-file $(STACK_ENV) run --rm smoke

health-check:
	@echo "[health] InferMesh"; \
	  (curl -fsS http://127.0.0.1:9000/health || echo "unreachable") && echo
	@echo "[health] embed-worker metrics"; \
	  (curl -fsS http://127.0.0.1:9101/metrics || echo "unreachable") && echo
	@echo "[health] spark-vectorizer UI"; \
	  echo "Open http://127.0.0.1:4041 in a browser"

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
