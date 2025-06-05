# API Documentation

## Overview

Question Answering RAG System menyediakan dua jenis API:

1. **Simple API** (`src/api/simple_api.py`) - API sederhana untuk basic QA
2. **Multi API** (`src/api/multi_api.py`) - API dengan multiple models dan advanced features

## Simple API

### Endpoints

#### POST /ask

Mengajukan pertanyaan ke sistem QA.

**Request:**

```json
{
    "question": "Apa itu pasal 1 UU No. 4 Tahun 2024?",
    "context": "optional context"
}
```

**Response:**

```json
{
    "answer": "Jawaban dari sistem",
    "confidence": 0.85,
    "sources": [
        {
            "document": "UU_Nomor_4_Tahun_2024.pdf",
            "page": 1,
            "relevance_score": 0.92
        }
    ]
}
```

#### GET /health

Health check endpoint.

**Response:**

```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

### Usage Example

```python
import requests

# Ask a question
response = requests.post("http://localhost:8000/ask", json={
    "question": "Apa definisi kesehatan menurut UU?"
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

## Multi API

### Features

-   Multiple model support
-   Batch processing
-   Advanced filtering
-   Custom embeddings

### Endpoints

#### POST /ask/batch

Process multiple questions at once.

**Request:**

```json
{
    "questions": ["Pertanyaan 1", "Pertanyaan 2"],
    "model": "default"
}
```

#### POST /ask/advanced

Advanced question answering with filters.

**Request:**

```json
{
    "question": "Pertanyaan",
    "filters": {
        "document_type": "UU",
        "year_range": [2020, 2024]
    },
    "model": "advanced"
}
```

## Running the APIs

### Simple API

```bash
python src/api/simple_api.py
# atau
make run-api
```

### Multi API

```bash
python src/api/multi_api.py
# atau
make run-multi-api
```

## Configuration

API configuration dapat diatur melalui environment variables:

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_PATH=/path/to/model
export DATA_PATH=/path/to/data
```

## Error Handling

### Common Error Codes

-   `400` - Bad Request (invalid input)
-   `404` - Not Found (document not found)
-   `500` - Internal Server Error
-   `503` - Service Unavailable (model not loaded)

### Error Response Format

```json
{
    "error": {
        "code": "INVALID_INPUT",
        "message": "Question cannot be empty",
        "details": {}
    }
}
```

## Rate Limiting

-   Simple API: 100 requests/minute
-   Multi API: 50 requests/minute
-   Batch API: 10 requests/minute

## Authentication

Currently, the APIs are open. For production deployment, implement:

-   API key authentication
-   JWT tokens
-   Rate limiting per user
-   Request logging

## Monitoring

### Metrics Endpoints

-   `/metrics` - Prometheus metrics
-   `/stats` - Usage statistics
-   `/logs` - Recent logs (admin only)

### Health Checks

-   `/health` - Basic health check
-   `/health/detailed` - Detailed system status
