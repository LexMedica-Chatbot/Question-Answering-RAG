# 📁 Project Structure - Clean & Organized

## 🎯 Overview

Struktur project Question-Answering-RAG yang sudah dibersihkan dan dirapikan, menghilangkan file-file yang tidak diperlukan seperti pytest, development tools, dan cache files.

## 📂 Directory Structure

```
Question-Answering-RAG/
├── 📁 src/                          # Source code utama
│   ├── 📁 api/                      # API endpoints
│   │   ├── 📁 executors/            # Agent executors
│   │   ├── 📁 models/               # Request/Response models
│   │   ├── 📁 tools/                # RAG tools (search, generation, etc.)
│   │   ├── 📁 utils/                # Utilities (config, document processor)
│   │   ├── 📄 simple_api.py         # Simple RAG API (Port 8000)
│   │   ├── 📄 multi_api.py          # Multi-Step RAG API (Port 8001)
│   │   └── 📄 health_api.py         # Health check API
│   ├── 📁 cache/                    # Smart caching system
│   ├── 📁 ingestion/                # Data ingestion scripts
│   ├── 📁 observability/            # Langfuse tracking
│   └── 📁 processing/               # Document processing
├── 📁 data/                         # Data files
│   ├── 📁 processed/                # Processed CSV files
│   └── 📁 raw/                      # Raw PDF documents
├── 📁 benchmarks/                   # Benchmark scripts
├── 📁 benchmark_results/            # Benchmark output
├── 📁 docs/                         # Documentation
├── 📁 notebooks/                    # Jupyter notebooks
├── 📄 requirements.txt              # Python dependencies
├── 📄 pyproject.toml               # Project configuration
├── 📄 Makefile                     # Build commands
├── 📄 docker-compose.yml           # Docker configuration
├── 📄 Dockerfile                   # Docker image
└── 📄 README.md                    # Main documentation
```

## 🚀 Quick Start Commands

### Installation

```bash
# Install dependencies
make install

# Or manually
pip install -r requirements.txt
```

### Running APIs

```bash
# Run Simple RAG API (Port 8000)
make run-simple-api

# Run Multi-Step RAG API (Port 8001)
make run-multi-api
```

### Docker

```bash
# Build and run with Docker
make docker-build
make docker-run
```

### Maintenance

```bash
# Clean cache files
make clean

# Setup virtual environment
make setup-env
```

## 📋 Core Dependencies (Cleaned)

### Framework & API

-   `fastapi==0.115.7` - Web framework
-   `uvicorn[standard]==0.34.0` - ASGI server

### LangChain & LLM

-   `langchain==0.3.19` - LLM framework
-   `langchain-community==0.3.17` - Community tools
-   `langchain-openai==0.3.6` - OpenAI integration
-   `openai==1.63.2` - OpenAI API

### Database & Storage

-   `supabase==2.13.0` - Vector database
-   `redis==5.2.1` - Caching

### Document Processing

-   `pypdf==5.3.0` - PDF processing
-   `bs4==0.0.2` - HTML parsing

### Utilities

-   `python-dotenv==1.0.1` - Environment variables
-   `pydantic==2.10.5` - Data validation
-   `rapidfuzz==3.12.1` - String matching
-   `scikit-learn==1.5.2` - ML utilities

### Observability

-   `langfuse==2.60.1` - LLM observability

## 🗑️ Removed Items

### Files Deleted:

-   ❌ `pytest.ini` - Testing configuration
-   ❌ `setup.py` - Replaced by pyproject.toml
-   ❌ `requirements-full-backup.txt` - Backup file
-   ❌ `view_embedding_results.py` - Standalone script
-   ❌ `run_ragas_benchmark.py` - Duplicate benchmark
-   ❌ `run_benchmark.sh` - Shell script
-   ❌ `.pytest_cache/` - Test cache directory
-   ❌ `src/__pycache__/` - Python cache
-   ❌ `qa-system/` - Old virtual environment

### Dependencies Removed:

-   ❌ `pytest>=6.0` - Testing framework
-   ❌ `black>=21.0` - Code formatter
-   ❌ `flake8>=3.8` - Linter
-   ❌ `isort>=5.0` - Import sorter
-   ❌ `mypy>=0.900` - Type checker

### Makefile Commands Removed:

-   ❌ `install-dev` - Development dependencies
-   ❌ `test` - Run tests
-   ❌ `lint` - Run linting
-   ❌ `format` - Format code
-   ❌ `run-demo` - Demo commands
-   ❌ `run-health` - Health check
-   ❌ `check-deps` - Dependency check

## 🎯 Benefits of Cleanup

1. **Simplified Dependencies**: Hanya dependencies yang benar-benar diperlukan
2. **Cleaner Structure**: Struktur folder yang lebih fokus
3. **Faster Installation**: Instalasi dependencies lebih cepat
4. **Reduced Complexity**: Mengurangi kompleksitas configuration
5. **Focus on Core**: Fokus pada fitur utama RAG system
6. **Better Maintenance**: Lebih mudah untuk maintenance

## 🔧 Available Make Commands

```bash
make help              # Show available commands
make install           # Install dependencies
make clean             # Clean cache files
make run-simple-api    # Run Simple RAG API (Port 8000)
make run-multi-api     # Run Multi-Step RAG API (Port 8001)
make docker-build      # Build Docker image
make docker-run        # Run with Docker Compose
make docker-stop       # Stop Docker containers
make setup-env         # Setup virtual environment
make ingest-data       # Run data ingestion
make process-data      # Run data processing
```

## ✅ Status: CLEAN & ORGANIZED

Project structure telah dibersihkan dan dirapikan. Sekarang lebih fokus pada core functionality dengan dependencies yang minimal dan struktur yang clean.
