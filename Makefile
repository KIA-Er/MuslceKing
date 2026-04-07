	.PHONY: help build-dev build-prod up-dev up-prod restart-dev restart-prod format pre-commit update-submodule

.DEFAULT_GOAL := help

help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build-dev: ## Build development docker images
	docker compose --env-file resources/docker/.env.dev -f resources/docker/docker-compose.yml build

build-prod: ## Build production docker images
	docker compose --env-file resources/docker/.env.prod -f resources/docker/docker-compose.yml build

up-dev: ## Start development environment (rebuild images)
	docker compose --env-file resources/docker/.env.dev -f resources/docker/docker-compose.yml up -d --build

up-prod: ## Start production environment (rebuild images)
	docker compose --env-file resources/docker/.env.prod -f resources/docker/docker-compose.yml up -d --build

restart-dev: ## Restart development environment
	docker compose --env-file resources/docker/.env.dev -f resources/docker/docker-compose.yml up -d

restart-prod: ## Restart production environment
	docker compose --env-file resources/docker/.env.prod -f resources/docker/docker-compose.yml up -d

down-dev: ## Stop development environment
	docker compose --env-file resources/docker/.env.dev -f resources/docker/docker-compose.yml down

down-prod: ## Stop production environment
	docker compose --env-file resources/docker/.env.prod -f resources/docker/docker-compose.yml down

format: ## Format code with ruff
	uvx ruff check --fix .
	uvx ruff format .

pre-commit: ## Run pre-commit hooks (ruff + pre-commit)
	uvx ruff check --fix .
	uvx ruff format .
	uvx pre-commit run --all-files

update-submodule: ## Update git submodules to latest remote version
	git submodule update --remote

init-submodule: ## Initialize git submodules
	git submodule update --init --recursive --remote
