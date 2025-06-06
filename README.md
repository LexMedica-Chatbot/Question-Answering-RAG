# Question Answering RAG System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Sistem Question Answering berbasis Retrieval-Augmented Generation (RAG) untuk dokumen hukum Indonesia.

## 📁 Struktur Project

```
question-answering-rag/
├── README.md                    # Dokumentasi utama
├── requirements.txt             # Dependencies Python
├── .gitignore                  # File yang diabaikan Git
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Multi-container setup
├── src/                        # Source code utama
│   ├── api/                    # API endpoints
│   │   ├── multi_api.py        # Enhanced Multi-Step RAG
│   │   └── simple_api.py       # Basic RAG Pipeline
│   ├── ingestion/              # Data ingestion modules
│   │   ├── ingest_in_db.py     # Database ingestion
│   │   ├── ingest_in_csv_db.py # CSV database ingestion
│   │   └── ingest_in_csv_db_small.py
│   ├── processing/             # Data processing modules
│   │   ├── export_pasal_csv.py # Export articles to CSV
│   │   └── export_pasal_tanpa_bab.py
│   └── demo/                   # Demo applications
│       ├── demo_simple.py      # Basic RAG Pipeline demo
│       └── main.py             # Main application
├── data/                       # Data directory
│   ├── raw/                    # Raw PDF files
│   └── processed/              # Processed CSV files
├── notebooks/                  # Jupyter notebooks
│   └── exploratory/            # Exploratory analysis
└── docs/                       # Documentation
```

## 🚀 Quick Start

### Prerequisites

1. Python 3.8 atau lebih tinggi
2. Virtual environment (recommended)

### Installation

1. **Clone repository:**

    ```bash
    git clone <repository-url>
    cd question-answering-rag
    ```

2. **Buat virtual environment:**

    ```bash
    python -m venv qa-system
    ```

3. **Aktivasi virtual environment:**

    **Windows:**

    ```bash
    qa-system\Scripts\activate
    ```

    **Linux/Mac:**

    ```bash
    source qa-system/bin/activate
    ```

4. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

#### API Server

```bash
# Basic RAG Pipeline
python src/api/simple_api.py

# Enhanced Multi-Step RAG
python src/api/multi_api.py
```

#### Demo Application

```bash
python src/demo/demo_simple.py
```

#### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 📊 Data Processing

### Ingestion

-   `src/ingestion/ingest_in_db.py` - Ingest data ke database
-   `src/ingestion/ingest_in_csv_db.py` - Ingest data ke CSV database

### Processing

-   `src/processing/export_pasal_csv.py` - Export pasal ke format CSV
-   `src/processing/export_pasal_tanpa_bab.py` - Export pasal tanpa bab

## 🔧 Development

### Project Structure Guidelines

-   `src/` - Semua source code
-   `data/raw/` - File PDF asli
-   `data/processed/` - File hasil processing
-   `notebooks/` - Jupyter notebooks untuk eksplorasi
-   `docs/` - Dokumentasi tambahan

### Contributing

1. Fork repository
2. Buat feature branch
3. Commit changes
4. Push ke branch
5. Create Pull Request

## 📝 License

[Tambahkan informasi license di sini]

## 🎯 Features

-   ✅ **Modular Architecture** - Kode terorganisir dalam modules yang jelas
-   ✅ **Multiple APIs** - Basic RAG Pipeline dan Enhanced Multi-Step RAG endpoints
-   ✅ **Docker Support** - Containerized deployment
-   ✅ **Data Pipeline** - Automated ingestion dan processing
-   ✅ **Testing Framework** - Unit tests dengan pytest
-   ✅ **Code Quality** - Black, flake8, isort, mypy
-   ✅ **Documentation** - Comprehensive docs dan API reference
-   ✅ **Development Tools** - Makefile untuk automation
-   ✅ **LangFuse Observability** - Cost tracking, token monitoring, performance analytics

## 📊 LangFuse Observability - RAG Analytics

Sistem menggunakan **LangFuse** untuk monitoring yang fokus pada **cost, token usage, dan RAG performance**.

### 🎯 **Metrics yang Ditrack**

-   **💰 Cost tracking**: Biaya real-time per request (USD)
-   **🔢 Token usage**: Input/output tokens setiap call
-   **📄 Document retrieval**: Jumlah dokumen yang diambil
-   **⚡ Response time**: End-to-end processing time
-   **🎨 Model comparison**: Large vs small embedding models

### 🚀 **5-Minute Setup**

```bash
# 1. Daftar di https://cloud.langfuse.com (gratis)
# 2. Buat project baru
# 3. Copy API keys ke .env:
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-here

# 4. Restart application
docker compose up --build -d

# 5. Check status
curl http://localhost:8080/api/observability
```

### 🖥️ **Beautiful Dashboard**

-   **Real-time cost monitoring** - Track spending per request
-   **Token usage analytics** - Optimize model efficiency
-   **Performance benchmarks** - Response time analysis
-   **Query pattern analysis** - Most asked questions
-   **Perfect untuk research insights!** 📊

📖 **Setup guide lengkap**: [SETUP_LANGFUSE.md](SETUP_LANGFUSE.md)

## 🔄 Migration dari Struktur Lama

Jika Anda memiliki kode yang menggunakan struktur lama:

```python
# Lama
from Final.simple_api import app
from Final.ingest_in_db import ingest_data

# Baru - Basic RAG Pipeline
from src.api.simple_api import app
from src.ingestion.ingest_in_db import ingest_data
```

## 🤝 Contributing

Lihat [CONTRIBUTING.md](docs/CONTRIBUTING.md) untuk panduan kontribusi.

## 📚 Documentation

-   [API Documentation](docs/API.md)
-   [Contributing Guide](docs/CONTRIBUTING.md)
