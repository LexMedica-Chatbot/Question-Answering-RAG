.PHONY: help install install-dev test lint format clean run-api run-demo docker-build docker-run

help: ## Tampilkan help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

lint: ## Run linting
	flake8 src/
	mypy src/

format: ## Format code
	black src/
	isort src/

clean: ## Clean cache and build files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

run-api: ## Run Basic RAG Pipeline server
	PYTHONPATH=. python -m src.api.simple_api

run-multi-api: ## Run Enhanced Multi-Step RAG server
	PYTHONPATH=. python -m src.api.multi_api

run-demo: ## Run demo application
	PYTHONPATH=. python -m src.demo.demo_simple

run-api-direct: ## Run Basic RAG Pipeline server (direct method)
	cd src/api && python simple_api.py

run-multi-api-direct: ## Run Enhanced Multi-Step RAG server (direct method)
	cd src/api && python multi_api.py

run-demo-direct: ## Run demo application (direct method)
	cd src/demo && python demo_simple.py

run-health: ## Run basic health check API (for testing)
	PYTHONPATH=. python -m src.api.health_api

docker-build: ## Build Docker image
	docker-compose build

docker-run: ## Run with Docker Compose
	docker-compose up

docker-stop: ## Stop Docker containers
	docker-compose down

setup-env: ## Setup virtual environment
	python -m venv qa-system
	@echo "Activate with: qa-system\\Scripts\\activate (Windows) or source qa-system/bin/activate (Linux/Mac)"

ingest-data: ## Run data ingestion
	python src/ingestion/ingest_in_csv_db.py

process-data: ## Run data processing
	python src/processing/export_pasal_csv.py

check-deps: ## Check dependencies and environment setup
	python scripts/check_dependencies.py 