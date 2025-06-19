FROM python:3.11-slim

# Direktori kerja di dalam container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements terlebih dahulu (agar layer caching lebih efektif)
COPY requirements.txt ./

# Instal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin .env file terlebih dahulu jika ada
COPY .env* ./

# Salin source code aplikasi
COPY . .

# Environment variables
ENV PORT=8080 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Perintah start menggunakan Enhanced Multi-Step RAG
CMD ["sh", "-c", "python -m uvicorn src.api.multi_api:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info"]

# # Perintah start menggunakan Basic RAG Pipeline
# CMD ["sh", "-c", "uvicorn src.api.simple_api:app --host 0.0.0.0 --port ${PORT:-8080}"] 