FROM python:3.11-slim

# Direktori kerja di dalam container
WORKDIR /app

# Salin requirements terlebih dahulu (agar layer caching lebih efektif)
COPY requirements.txt ./

# Instal dependensi
RUN pip install -r requirements.txt

# Salin .env file terlebih dahulu jika ada
COPY .env* ./

# Salin source code aplikasi
COPY . .

# Environment variables
ENV PORT=8080 \
    PYTHONPATH=/app

# # Perintah start menggunakan Enhanced Multi-Step RAG
# CMD ["sh", "-c", "uvicorn src.api.multi_api:app --host 0.0.0.0 --port ${PORT:-8080}"]

# Perintah start menggunakan Basic RAG Pipeline
CMD ["sh", "-c", "uvicorn src.api.simple_api:app --host 0.0.0.0 --port ${PORT:-8080}"] 