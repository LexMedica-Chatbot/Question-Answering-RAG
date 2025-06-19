.PHONY: help install clean run-simple-api run-multi-api docker-build docker-run

help: ## Tampilkan help message
	@echo Available commands:
	@echo   install           Install dependencies
	@echo   clean             Clean cache and build files
	@echo   run-simple-api    Run Simple RAG API (Port 8000)
	@echo   run-multi-api     Run Multi-Step RAG API (Port 8001)
	@echo   docker-build      Build Docker image
	@echo   docker-run        Run with Docker Compose
	@echo   docker-stop       Stop Docker containers
	@echo   setup-env         Setup virtual environment
	@echo   ingest-data       Run data ingestion
	@echo   process-data      Run data processing

install: ## Install dependencies
	pip install -r requirements.txt

clean: ## Clean cache and build files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

run-simple-api: ## Run Simple RAG API (Port 8000)
	cd src/api && python simple_api.py

run-multi-api: ## Run Multi-Step RAG API (Port 8001)
	cd src/api && python multi_api.py

docker-build: ## Build Docker image
	docker-compose build

docker-run: ## Run with Docker Compose
	docker-compose up

docker-stop: ## Stop Docker containers
	docker-compose down

setup-env: ## Setup virtual environment
	python -m venv venv
	@echo "Activate with: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Linux/Mac)"

ingest-data: ## Run data ingestion
	python src/ingestion/ingest_in_csv_db.py

process-data: ## Run data processing
	python src/processing/export_pasal_csv.py 