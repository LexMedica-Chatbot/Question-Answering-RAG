FROM python:3.11-slim

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

# Perintah start menggunakan agentic_rag_executor.py
CMD ["sh", "-c", "uvicorn agentic_rag_executor:app --host 0.0.0.0 --port ${PORT:-8080}"] 


# # Perintah start menggunakan simple_api.py
# CMD ["sh", "-c", "uvicorn simple_api:app --host 0.0.0.0 --port ${PORT:-8080}"] 