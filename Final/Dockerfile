FROM python:3.11-slim

# Install Redis server untuk Cloud Run deployment
RUN apt-get update && apt-get install -y \
    redis-server \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Mengatur variabel lingkungan dasar
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Direktori kerja di dalam container
WORKDIR /app

# Salin requirements terlebih dahulu (agar layer caching lebih efektif)
COPY requirements.txt ./

# Instal dependensi
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Salin source code aplikasi
COPY . .

# Cloud Run mewajibkan aplikasi mendengarkan $PORT (default 8080)
ENV PORT=8080
ENV REDIS_URL=redis://localhost:6379

# Script untuk start Redis + FastAPI (untuk Cloud Run)
RUN echo '#!/bin/bash\n\
if [ "$NODE_ENV" = "production" ]; then\n\
  echo "🚀 Starting production mode with Redis..."\n\
  redis-server --daemonize yes --maxmemory 256mb --maxmemory-policy allkeys-lru\n\
  sleep 2\n\
  echo "✅ Redis started"\n\
fi\n\
echo "🌐 Starting FastAPI..."\n\
exec "$@"' > start.sh \
    && chmod +x start.sh

# Health check untuk Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Perintah start menggunakan multi_api.py
CMD ["sh", "-c", "uvicorn multi_api:app --host 0.0.0.0 --port ${PORT:-8080}"]

# # Perintah start menggunakan simple_api.py
# CMD ["sh", "-c", "uvicorn simple_api:app --host 0.0.0.0 --port ${PORT:-8080}"] 