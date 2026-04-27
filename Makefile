export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

COMPOSE = docker compose
DEV = docker compose -f compose.yml -f compose.dev.yml

SERVICE ?=
SERVICES ?=

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make prod-up                 Start production stack"
	@echo "  make prod-down               Stop production stack"
	@echo "  make prod-logs               Follow production logs"
	@echo ""
	@echo "  make dev-up                  Start dev stack"
	@echo "  make dev-up SERVICES='proxy webui'"
	@echo "  make dev-down                Stop dev stack"
	@echo "  make dev-logs                Follow dev logs"
	@echo ""
	@echo "  make build SERVICE=proxy     Build one service"
	@echo "  make rebuild SERVICE=proxy   Build one service without cache"
	@echo "  make restart SERVICE=proxy   Restart one dev service"
	@echo "  make up SERVICE=proxy        Start/recreate one dev service without deps"
	@echo ""
	@echo "  make ps                      Show services"
	@echo "  make health                  Check health endpoints"

.PHONY: prod-up
prod-up:
	$(COMPOSE) up -d --build $(SERVICES)

.PHONY: prod-down
prod-down:
	$(COMPOSE) down

.PHONY: prod-logs
prod-logs:
	$(COMPOSE) logs -f --tail=200 $(SERVICE)

.PHONY: dev-up
dev-up:
	$(DEV) up -d $(SERVICES)

.PHONY: dev-build-up
dev-build-up:
	$(DEV) up -d --build $(SERVICES)

.PHONY: dev-down
dev-down:
	$(DEV) down

.PHONY: dev-logs
dev-logs:
	$(DEV) logs -f --tail=200 $(SERVICE)

.PHONY: build
build:
	@if [ -z "$(SERVICE)" ]; then echo "Use: make build SERVICE=proxy"; exit 1; fi
	$(COMPOSE) build $(SERVICE)

.PHONY: rebuild
rebuild:
	@if [ -z "$(SERVICE)" ]; then echo "Use: make rebuild SERVICE=proxy"; exit 1; fi
	$(COMPOSE) build --no-cache $(SERVICE)

.PHONY: restart
restart:
	@if [ -z "$(SERVICE)" ]; then echo "Use: make restart SERVICE=proxy"; exit 1; fi
	$(DEV) restart $(SERVICE)

.PHONY: up
up:
	@if [ -z "$(SERVICE)" ]; then echo "Use: make up SERVICE=proxy"; exit 1; fi
	$(DEV) up -d --no-deps $(SERVICE)

.PHONY: stop
stop:
	@if [ -z "$(SERVICE)" ]; then echo "Use: make stop SERVICE=proxy"; exit 1; fi
	$(DEV) stop $(SERVICE)

.PHONY: ps
ps:
	$(DEV) ps

.PHONY: health
health:
	@echo "server:"
	@curl -fsS http://localhost:8080/v1/health || true
	@echo "\nproxy:"
	@curl -fsS http://localhost:9000/health || true
	@echo "\nwebui:"
	@curl -fsS http://localhost:9001/health || true
	@echo ""